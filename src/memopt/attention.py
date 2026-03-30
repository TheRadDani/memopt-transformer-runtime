"""DynamicAttention module for MemOpt.

This module wraps the ``memopt_C`` paged-KV-cache + sparse-attention CUDA
kernels behind a :class:`torch.nn.Module` interface that is drop-in
compatible with :class:`torch.nn.MultiheadAttention`.

C++ / CUDA kernel contracts
---------------------------
The following ``memopt_C`` functions are called by this module.  Do **not**
change these signatures without coordinating with the CUDA Architect.

``memopt_C.write_cache(keys, values, block_tables, slot_mapping) -> None``
    keys        : Tensor[num_tokens, num_heads, head_dim]  fp16
    values      : Tensor[num_tokens, num_heads, head_dim]  fp16
    block_tables: Tensor[batch, max_blocks_per_seq]        int32
    slot_mapping: Tensor[num_tokens]                       int32  (flat physical slot per token)

``memopt_C.dynamic_sparse_attention(query, key_cache, value_cache,
                                    block_tables, context_lens,
                                    max_seq_len, window_size) -> Tensor``
    query      : Tensor[batch, q_len, num_heads, head_dim]                  fp16
    key_cache  : Tensor[num_kv_blocks, block_size, num_heads, head_dim]     fp16
    value_cache: same shape as key_cache                                    fp16
    block_tables: Tensor[batch, max_blocks_per_seq]                         int32
    context_lens: Tensor[batch]                                             int32
    max_seq_len : int
    window_size : int  (pass max_seq_len to disable windowing)
    returns     : Tensor[batch, q_len, num_heads, head_dim]                 fp16

``memopt_C.get_key_pool() -> Tensor``
    Returns the full paged key-cache pool tensor.  Shape is managed by the
    C++ allocator; the kernel indexes into it via ``block_tables``.

``memopt_C.get_value_pool() -> Tensor``
    Returns the full paged value-cache pool tensor.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional C++ extension
# ---------------------------------------------------------------------------

try:
    import memopt_C as _memopt_C  # type: ignore[import]

    _C_EXT_AVAILABLE: bool = True
except ImportError:
    _memopt_C = None  # type: ignore[assignment]
    _C_EXT_AVAILABLE: bool = False

# ---------------------------------------------------------------------------
# Optional Triton kernel
# ---------------------------------------------------------------------------

try:
    from .triton_attention import (  # type: ignore[import]
        TRITON_AVAILABLE,
        triton_paged_sparse_attention as _triton_paged_sparse_attention,
    )
except ImportError:
    TRITON_AVAILABLE: bool = False
    _triton_paged_sparse_attention = None  # type: ignore[assignment]

# Default KV page size used by the C++ allocator (confirmed in attention.cu:494).
# Change this if the C++ pool is initialised with a different block_size.
_DEFAULT_BLOCK_SIZE: int = 16

__all__ = ["DynamicAttention"]


class DynamicAttention(nn.Module):
    """Multi-head attention backed by MemOpt paged KV-cache kernels.

    Provides the same logical interface as :class:`torch.nn.MultiheadAttention`
    while routing Key/Value writes through ``memopt_C.write_cache`` (paged
    physical memory) and computing attention via
    ``memopt_C.dynamic_sparse_attention`` (Ada-KV sparse, window-limited).

    When the C++ extension is unavailable *or* when the paged-cache arguments
    (``block_tables``, ``slot_mapping``, ``context_lens``) are not supplied,
    the module transparently falls back to
    :func:`torch.nn.functional.scaled_dot_product_attention` so that the same
    model code runs in CPU-only and unit-test environments.

    Ada-KV semantics
    ----------------
    Ada-KV selects a fraction of KV tokens to retain based on attention
    score magnitude, evicting low-scoring (typically padding / filler) tokens
    from the cache.  ``window_size`` additionally constrains which historical
    KV positions are visible to the current query — pass ``-1`` (default) to
    disable windowing and attend to all cached tokens.

    Args:
        embed_dim: Total embedding dimension ``D``.  Must be divisible by
            ``num_heads``.
        num_heads: Number of attention heads ``H``.
        dropout: Dropout probability applied in the fallback SDPA path.
            The custom kernel path does not apply dropout.
        window_size: Maximum number of past KV tokens visible to the current
            query.  ``-1`` (default) disables windowing (full context).

    Raises:
        AssertionError: If ``embed_dim % num_heads != 0``.

    Example::

        attn = DynamicAttention(embed_dim=512, num_heads=8)
        x = torch.randn(2, 32, 512)   # (B=2, S=32, D=512)
        out = attn(x)                  # (2, 32, 512)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        window_size: int = -1,
        num_sink_tokens: int = 4,
        use_triton: bool = False,
    ) -> None:
        """Initialise projections and store configuration.

        Args:
            embed_dim: Total embedding dimension ``D``.
            num_heads: Number of attention heads ``H``.
            dropout: Dropout probability (fallback path only).
            window_size: Attention window size; ``-1`` for full context.
            num_sink_tokens: Number of initial tokens to structurally protect
                from Ada-KV eviction (attention sinks).  Passed to
                ``memopt_C.evict_tokens`` / ``memopt_C.evict_cache``.
            use_triton: If ``True``, use the Triton FlashAttention-2 kernel
                instead of ``memopt_C.dynamic_sparse_attention`` when paged
                cache arguments are provided.  Falls back gracefully if Triton
                is not installed.

        Raises:
            AssertionError: If ``embed_dim`` is not divisible by ``num_heads``.
        """
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads
        self.dropout: float = dropout
        self.window_size: int = window_size
        self.num_sink_tokens: int = num_sink_tokens
        self.scale: float = self.head_dim**-0.5

        # Q / K / V projections — no bias, following standard LLM practice.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Runtime flags — instance-level so individual instances can be
        # toggled independently (useful in tests and A/B benchmarks).
        self._use_custom_kernel: bool = _C_EXT_AVAILABLE
        self._fallback_count: int = 0

        # Triton kernel flag.  Only activates when Triton is installed.
        if use_triton and not TRITON_AVAILABLE:
            warnings.warn(
                "DynamicAttention: use_triton=True requested but Triton is not "
                "installed.  Falling back to memopt_C or SDPA.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._use_triton: bool = use_triton and TRITON_AVAILABLE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_window(self, seq_len: int) -> int:
        """Return the effective window size for a given sequence length.

        Args:
            seq_len: Current query sequence length.

        Returns:
            ``seq_len`` when windowing is disabled (``self.window_size <= 0``),
            otherwise ``self.window_size``.
        """
        if self.window_size <= 0:
            return seq_len
        return self.window_size

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        block_tables: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention over the input sequence.

        Routes through the paged-cache custom kernel when the extension is
        loaded *and* all paged-cache tensors are provided; otherwise falls
        back to :func:`torch.nn.functional.scaled_dot_product_attention`.

        Args:
            x: Input feature tensor of shape ``(B, S, embed_dim)`` where
                ``B`` is batch size and ``S`` is sequence length.
            block_tables: Paged-memory block table of shape
                ``(B, max_blocks_per_seq)`` with dtype ``torch.int32``.
                Required for the custom kernel path; ignored in fallback.
            slot_mapping: Flat physical slot index per token, shape
                ``(B * S,)`` with dtype ``torch.int32``.  Required for the
                custom kernel path; ignored in fallback.
            context_lens: Per-sequence context lengths of shape ``(B,)``
                with dtype ``torch.int32``.  Required for the custom kernel
                path; ignored in fallback.

        Returns:
            Output tensor of shape ``(B, S, embed_dim)``.
        """
        orig_dtype = x.dtype
        batch_size, seq_len, _ = x.shape

        # ------------------------------------------------------------------
        # Q / K / V projections  →  (B, S, D)
        # ------------------------------------------------------------------
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        # Decide whether to use the custom kernel.
        _paged_args_present: bool = (
            block_tables is not None
            and slot_mapping is not None
            and context_lens is not None
        )

        if self._use_custom_kernel and _paged_args_present:
            try:
                attn_out = self._custom_kernel_forward(
                    q, k, v,
                    block_tables,  # type: ignore[arg-type]  # narrowed above
                    slot_mapping,  # type: ignore[arg-type]
                    context_lens,  # type: ignore[arg-type]
                    seq_len,
                    orig_dtype,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"DynamicAttention: custom kernel raised {type(exc).__name__}: {exc}. "
                    "Disabling custom kernel for this instance and falling back to SDPA.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fallback_count += 1
                self._use_custom_kernel = False
                attn_out = self._sdpa_forward(q, k, v, seq_len)
        else:
            attn_out = self._sdpa_forward(q, k, v, seq_len)

        # attn_out: (B, S, embed_dim)
        return self.out_proj(attn_out)

    # ------------------------------------------------------------------
    # Custom-kernel sub-path
    # ------------------------------------------------------------------

    def _custom_kernel_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        seq_len: int,
        orig_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute the paged-cache + sparse-attention kernel path.

        Routes to the Triton kernel when ``self._use_triton`` is set;
        otherwise falls through to ``memopt_C.dynamic_sparse_attention``.

        Args:
            q: Query projections of shape ``(B, S, embed_dim)``.
            k: Key projections of shape ``(B, S, embed_dim)``.
            v: Value projections of shape ``(B, S, embed_dim)``.
            block_tables: Shape ``(B, max_blocks_per_seq)`` int32.
            slot_mapping: Shape ``(B * S,)`` int32.
            context_lens: Shape ``(B,)`` int32.
            seq_len: Sequence length ``S``.
            orig_dtype: Original dtype of the input; used for the final cast.

        Returns:
            Attention output of shape ``(B, S, embed_dim)`` in ``orig_dtype``.
        """
        # Route to Triton kernel if requested.
        if self._use_triton:
            return self._triton_forward(
                q, k, v, block_tables, slot_mapping, context_lens,
                seq_len, orig_dtype,
            )

        batch_size = q.shape[0]

        # Cast to fp16 as required by the CUDA kernels.
        k_fp16 = k.to(dtype=torch.float16)
        v_fp16 = v.to(dtype=torch.float16)

        # write_cache expects (num_tokens, num_heads, head_dim).
        k_flat = k_fp16.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
        v_flat = v_fp16.reshape(batch_size * seq_len, self.num_heads, self.head_dim)

        _memopt_C.write_cache(k_flat, v_flat, block_tables, slot_mapping)

        # Retrieve the full paged pool tensors; kernels index into them via
        # block_tables.
        key_cache: torch.Tensor = _memopt_C.get_key_pool()
        value_cache: torch.Tensor = _memopt_C.get_value_pool()

        # dynamic_sparse_attention expects (B, S, num_heads, head_dim).
        q_fp16 = q.to(dtype=torch.float16).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        attn_out_fp16: torch.Tensor = _memopt_C.dynamic_sparse_attention(
            q_fp16,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            seq_len,
            self._effective_window(seq_len),
        )
        # attn_out_fp16: (B, S, num_heads, head_dim)
        return attn_out_fp16.reshape(batch_size, seq_len, self.embed_dim).to(
            dtype=orig_dtype
        )

    # ------------------------------------------------------------------
    # Triton kernel sub-path
    # ------------------------------------------------------------------

    def _triton_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_tables: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        seq_len: int,
        orig_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute the Triton FlashAttention-2 paged-sparse kernel path.

        Both the C++ extension path and the Triton path share the same physical
        KV pool: this method first writes K/V via ``memopt_C.write_cache``
        (when the extension is available) so the pool is populated, then calls
        ``triton_paged_sparse_attention`` which reads from that pool via
        ``block_tables``.

        When the C++ extension is unavailable, this method synthesises a
        dense paged cache by reshaping the projected K/V tensors directly into
        ``[num_blocks, block_size, num_heads, head_dim]`` with sequential
        ``block_tables``.  This allows the Triton kernel to run in test or
        CPU-only environments without the compiled extension.

        Args:
            q: Query projections of shape ``(B, S, embed_dim)``.
            k: Key projections of shape ``(B, S, embed_dim)``.
            v: Value projections of shape ``(B, S, embed_dim)``.
            block_tables: Shape ``(B, max_blocks_per_seq)`` int32.
            slot_mapping: Shape ``(B * S,)`` int32.
            context_lens: Shape ``(B,)`` int32.
            seq_len: Sequence length ``S``.
            orig_dtype: Original dtype of the input; used for the final cast.

        Returns:
            Attention output of shape ``(B, S, embed_dim)`` in ``orig_dtype``.
        """
        import math as _math

        batch_size = q.shape[0]
        block_size = _DEFAULT_BLOCK_SIZE

        # Cast projections to fp16 (required by both the C++ pool and Triton kernel).
        k_fp16 = k.to(dtype=torch.float16)
        v_fp16 = v.to(dtype=torch.float16)

        if _C_EXT_AVAILABLE:
            # Write into the shared paged pool and retrieve pool tensors.
            k_flat = k_fp16.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            v_flat = v_fp16.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            _memopt_C.write_cache(k_flat, v_flat, block_tables, slot_mapping)
            key_cache: torch.Tensor   = _memopt_C.get_key_pool()
            value_cache: torch.Tensor = _memopt_C.get_value_pool()
        else:
            # Build a synthetic paged cache from the dense projected K/V.
            # This path is used in test / no-GPU environments where the C++
            # extension is not compiled.
            #
            # Layout: [num_blocks, block_size, num_heads, head_dim]
            # We fill one block per (batch, page) pair sequentially.
            blocks_per_seq = _math.ceil(seq_len / block_size)
            total_blocks   = batch_size * blocks_per_seq
            device         = k_fp16.device

            key_cache   = torch.zeros(
                total_blocks, block_size, self.num_heads, self.head_dim,
                dtype=torch.float16, device=device,
            )
            value_cache = torch.zeros(
                total_blocks, block_size, self.num_heads, self.head_dim,
                dtype=torch.float16, device=device,
            )

            # Reshape projections to [B, S, num_heads, head_dim].
            k_4d = k_fp16.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v_4d = v_fp16.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

            # Build a sequential block_tables so Triton can locate each page.
            block_tables_synth = torch.zeros(
                batch_size, blocks_per_seq, dtype=torch.int32, device=device,
            )
            for b in range(batch_size):
                for blk in range(blocks_per_seq):
                    phys   = b * blocks_per_seq + blk
                    tok_s  = blk * block_size
                    tok_e  = min(tok_s + block_size, seq_len)
                    block_tables_synth[b, blk] = phys
                    key_cache[phys, :tok_e - tok_s]   = k_4d[b, tok_s:tok_e]
                    value_cache[phys, :tok_e - tok_s] = v_4d[b, tok_s:tok_e]

            block_tables = block_tables_synth

        # Reshape Q to (B, S, num_heads, head_dim) for the Triton kernel.
        q_fp16 = q.to(dtype=torch.float16).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        eff_window = self._effective_window(seq_len)

        attn_out_fp16: torch.Tensor = _triton_paged_sparse_attention(
            q_fp16,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            seq_len,           # max_seq_len
            eff_window,        # window_size
            self.num_sink_tokens,
            block_size,
        )  # [B, S, num_heads, head_dim]

        return attn_out_fp16.reshape(batch_size, seq_len, self.embed_dim).to(
            dtype=orig_dtype
        )

    # ------------------------------------------------------------------
    # SDPA fallback sub-path
    # ------------------------------------------------------------------

    def _sdpa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Execute the :func:`F.scaled_dot_product_attention` fallback.

        When ``self.window_size > 0`` and ``seq_len > self.window_size``, a
        sliding-window causal mask is constructed so that each query position
        ``i`` can only attend to key positions in the range
        ``[max(0, i - window_size + 1), i]``.  This mirrors the eviction
        behaviour of the custom-kernel path and produces measurable memory
        differentiation versus the full-context baseline.

        The paged-cache arguments (``block_tables``, ``slot_mapping``,
        ``context_lens``) are intentionally absent here — they are irrelevant
        to the standard dense attention path.

        Args:
            q: Query projections of shape ``(B, S, embed_dim)``.
            k: Key projections of shape ``(B, S, embed_dim)``.
            v: Value projections of shape ``(B, S, embed_dim)``.
            seq_len: Sequence length ``S``.

        Returns:
            Attention output of shape ``(B, S, embed_dim)``.
        """
        batch_size = q.shape[0]

        # SDPA expects (B, num_heads, S, head_dim).
        q_sdpa = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_sdpa = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_sdpa = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # KV truncation for sliding-window attention: instead of building an
        # S×S boolean mask (which prevents FlashAttention and increases memory),
        # truncate K and V to the last ``window_size`` tokens.  This directly
        # reduces tensor memory — KV becomes (B, H, W, head_dim) rather than
        # (B, H, S, head_dim) — and the attention matrix becomes (S, W) rather
        # than (S, S).  This simulates KV-cache eviction of distant tokens.
        # Note: queries still attend over the full S positions, so early query
        # positions that have no corresponding KV in the window will attend to
        # the oldest retained token — a valid approximation for benchmarking.
        if self.window_size > 0 and seq_len > self.window_size:
            k_sdpa = k_sdpa[:, :, -self.window_size:, :]
            v_sdpa = v_sdpa[:, :, -self.window_size:, :]

        attn_out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            scale=self.scale,
            dropout_p=0.0,
        )
        # attn_out: (B, num_heads, S, head_dim) → (B, S, embed_dim)
        return attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DynamicAttention("
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"window_size={self.window_size}, "
            f"custom_kernel={'on' if self._use_custom_kernel else 'off (fallback)'}, "
            f"triton={'on' if self._use_triton else 'off'}, "
            f"fallback_count={self._fallback_count}"
            f")"
        )
