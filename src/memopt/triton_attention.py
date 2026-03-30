"""Triton FlashAttention-2 kernel with Ada-KV sparsity and paged KV cache.

Kernel contract
---------------
``triton_paged_sparse_attention`` is a drop-in replacement for
``memopt_C.dynamic_sparse_attention``.  It accepts the same paged KV tensors and
returns the same output shape, but is implemented entirely in Triton so it runs
without the compiled ``memopt_C`` C++ extension.

Function signature
~~~~~~~~~~~~~~~~~~
``triton_paged_sparse_attention(query, key_cache, value_cache,
                                block_tables, context_lens,
                                max_seq_len, window_size,
                                num_sink_tokens, block_size) -> Tensor``

    query        : Tensor[batch, q_len, num_heads, head_dim]  fp16
    key_cache    : Tensor[num_kv_blocks, block_size, num_heads, head_dim]  fp16
    value_cache  : same shape as key_cache  fp16
    block_tables : Tensor[batch, max_blocks_per_seq]  int32
    context_lens : Tensor[batch]  int32
    max_seq_len  : int   (used to determine when windowing is disabled)
    window_size  : int   (active window; pass max_seq_len to attend to all tokens)
    num_sink_tokens : int  (first N tokens are always retained, never skipped)
    block_size   : int   (KV page size — must match pool init, typically 16)
    returns      : Tensor[batch, q_len, num_heads, head_dim]  fp16

Ada-KV sparsity semantics
~~~~~~~~~~~~~~~~~~~~~~~~~
A KV block is skipped (not loaded into SRAM) when *all* of its token positions
fall outside both:
  1. The sliding attention window ``[context_len - window_size, context_len)``.
  2. The sink-token region ``[0, num_sink_tokens)``.

Blocks that overlap either region are always processed.  When
``window_size == max_seq_len``, windowing is effectively disabled and all
blocks in the context are processed (only the boundary-mask logic in the
online softmax then determines which individual tokens contribute).

FlashAttention-2 online softmax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For each (batch × head, q-tile) program instance the kernel accumulates:
  - ``m``   : running max of QK scores seen so far  (fp32, per query lane)
  - ``l``   : running softmax normaliser ``sum(exp(QK - m))``  (fp32)
  - ``acc`` : weighted value accumulator  (fp32, per head_dim element)

When the running max increases from ``m_old`` to ``m_new``, the previously
accumulated ``acc`` and ``l`` are rescaled by ``exp(m_old - m_new)`` before
adding the new block's contribution.  At the end, ``acc / l`` gives the
exact softmax-weighted value sum (up to fp16 quantisation error).
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional Triton import
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE: bool = True
except ImportError:
    TRITON_AVAILABLE = False

__all__ = ["triton_paged_sparse_attention", "TRITON_AVAILABLE"]

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _flash_attn_paged_sparse_kernel(
        # --- Pointers ---
        Q_ptr,           # [batch, q_len, num_heads, head_dim]  fp16
        K_cache_ptr,     # [num_kv_blocks, BLOCK_SIZE, num_heads, head_dim]  fp16
        V_cache_ptr,     # same shape as K_cache_ptr  fp16
        Block_tables_ptr,  # [batch, max_blocks_per_seq]  int32
        Context_lens_ptr,  # [batch]  int32
        Out_ptr,         # [batch, q_len, num_heads, head_dim]  fp16  (output)
        # --- Scalar dimensions ---
        batch: int,
        q_len: int,
        num_heads: int,
        max_blocks_per_seq: int,
        window_size: int,
        num_sink_tokens: int,
        scale,           # float32  (1 / sqrt(head_dim))
        # --- Strides for Q  (row-major: batch, q, head, dim) ---
        stride_qb: int,
        stride_qs: int,
        stride_qh: int,
        stride_qd: int,
        # --- Strides for K/V cache  (block, pos, head, dim) ---
        stride_kb: int,
        stride_ks: int,
        stride_kh: int,
        stride_kd: int,
        # --- Strides for block_tables  (batch, block_idx) ---
        stride_btb: int,
        stride_bti: int,
        # --- Strides for output ---
        stride_ob: int,
        stride_os: int,
        stride_oh: int,
        stride_od: int,
        # --- Constexpr tile sizes ---
        BLOCK_SIZE: tl.constexpr,   # KV page size (tokens per physical block)
        Q_TILE_SIZE: tl.constexpr,  # number of query positions per program
        HEAD_DIM: tl.constexpr,     # must be a power of 2
    ) -> None:
        """FlashAttention-2 Triton kernel with paged KV and Ada-KV sparsity.

        Grid: (batch * num_heads, cdiv(q_len, Q_TILE_SIZE))
          - Program axis 0 encodes (batch_idx * num_heads + head_idx).
          - Program axis 1 encodes the q-tile index.
        """
        # ------------------------------------------------------------------
        # Decode program coordinates
        # ------------------------------------------------------------------
        bh_idx = tl.program_id(0)   # flattened (batch, head) index
        q_tile = tl.program_id(1)   # which tile of Q this program handles

        batch_idx = bh_idx // num_heads
        head_idx  = bh_idx  % num_heads

        # Query positions this program is responsible for.
        q_start = q_tile * Q_TILE_SIZE
        q_offs  = q_start + tl.arange(0, Q_TILE_SIZE)        # [Q_TILE_SIZE]
        q_mask  = q_offs < q_len

        # Head-dimension lane indices.
        d_offs = tl.arange(0, HEAD_DIM)                       # [HEAD_DIM]

        # ------------------------------------------------------------------
        # Load Q tile:  [Q_TILE_SIZE, HEAD_DIM]  (fp16 → fp32 for accumulator)
        # ------------------------------------------------------------------
        # Base pointer to Q[batch_idx, 0, head_idx, 0]
        q_base = (
            Q_ptr
            + batch_idx * stride_qb
            + head_idx  * stride_qh
        )
        # q_ptrs[i, d] → Q[batch, q_offs[i], head, d]
        q_ptrs = (
            q_base
            + q_offs[:, None] * stride_qs    # [Q_TILE_SIZE, 1]
            + d_offs[None, :] * stride_qd    # [1, HEAD_DIM]
        )
        # Mask out-of-bounds query rows; fill with 0 (won't affect result).
        q_tile_data = tl.load(
            q_ptrs,
            mask=q_mask[:, None] & (d_offs[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)  # [Q_TILE_SIZE, HEAD_DIM] in fp32

        # ------------------------------------------------------------------
        # Per-query FA-2 running accumulators  (all fp32)
        # ------------------------------------------------------------------
        # m[i]   : running max of QK logits for query i
        # l[i]   : running softmax denominator for query i
        # acc[i] : running weighted value sum for query i
        m_i   = tl.full([Q_TILE_SIZE],         float("-inf"), dtype=tl.float32)
        l_i   = tl.zeros([Q_TILE_SIZE],                       dtype=tl.float32)
        acc_i = tl.zeros([Q_TILE_SIZE, HEAD_DIM],             dtype=tl.float32)

        # ------------------------------------------------------------------
        # Retrieve context length for this batch entry.
        # ------------------------------------------------------------------
        context_len = tl.load(Context_lens_ptr + batch_idx).to(tl.int32)

        # Effective window start: tokens older than this can be skipped *unless*
        # they are sink tokens.  May be negative (no windowing).
        window_start = context_len - window_size

        # ------------------------------------------------------------------
        # Iterate over KV pages
        # ------------------------------------------------------------------
        # Triton's JIT does not support `break` or `continue` statements.
        # We iterate over all max_blocks_per_seq slots and use scalar boolean
        # guards to mask out contributions from:
        #   (a) Slots beyond the actual context  (block_active = False)
        #   (b) Blocks entirely outside the window AND the sink region
        #       (block_relevant = False)
        # When a block is inactive or irrelevant we still load from the
        # block_tables array (with mask=False on K/V so we read 0.0) and then
        # set all QK scores to -inf before the online-softmax step, which makes
        # their softmax weights exp(-inf)=0 — a no-op contribution.
        bt_base = Block_tables_ptr + batch_idx * stride_btb

        for block_idx in range(max_blocks_per_seq):
            # Token positions covered by this logical block.
            tok_start = block_idx * BLOCK_SIZE
            tok_end   = tok_start + BLOCK_SIZE   # exclusive; may exceed context_len

            # Is this block within the actual context?  (replaces the `break`.)
            block_active = tok_start < context_len

            # Ada-KV sparsity gate (replaces the `continue`):
            # Process the block only if it overlaps the window OR the sink region.
            in_window    = tok_end   > window_start          # overlaps sliding window
            in_sinks     = tok_start < num_sink_tokens       # overlaps sink tokens
            block_relevant = in_window | in_sinks

            # Combined mask: we do real work only when both conditions hold.
            do_block = block_active & block_relevant

            # ------------------------------------------------------------------
            # Look up the physical block index via block_tables.
            # We always read block_tables to avoid an out-of-bounds pointer
            # calculation; when do_block is False the result is discarded.
            # ------------------------------------------------------------------
            phys_block_idx = tl.load(bt_base + block_idx * stride_bti).to(tl.int32)

            # ------------------------------------------------------------------
            # Load K tile:  [BLOCK_SIZE, HEAD_DIM]  (fp16)
            # ------------------------------------------------------------------
            # K_cache layout: [num_kv_blocks, BLOCK_SIZE, num_heads, head_dim]
            k_base = (
                K_cache_ptr
                + phys_block_idx * stride_kb
                + head_idx       * stride_kh
            )
            pos_offs = tl.arange(0, BLOCK_SIZE)                    # [BLOCK_SIZE]

            # Per-position validity:
            #   - position must be within the actual context (pos_in_context)
            #   - do_block must be True; if False we load 0.0 → QK = 0 → masked
            pos_in_context = (tok_start + pos_offs) < context_len  # [BLOCK_SIZE]
            load_mask      = do_block & pos_in_context              # [BLOCK_SIZE]

            k_ptrs = (
                k_base
                + pos_offs[:, None] * stride_ks    # [BLOCK_SIZE, 1]
                + d_offs[None, :]   * stride_kd    # [1, HEAD_DIM]
            )
            k_tile = tl.load(
                k_ptrs,
                mask=load_mask[:, None],
                other=0.0,
            ).to(tl.float32)   # [BLOCK_SIZE, HEAD_DIM]

            # ------------------------------------------------------------------
            # QK dot product:  [Q_TILE_SIZE, BLOCK_SIZE]
            #   Use tl.dot for tensor-core acceleration on Ampere+.
            # ------------------------------------------------------------------
            # tl.dot requires both operands to be 2-D and the inner dimension
            # must match.  q_tile_data: [Q_TILE_SIZE, HEAD_DIM];
            # k_tile transposed: [HEAD_DIM, BLOCK_SIZE].
            qk = tl.dot(
                q_tile_data.to(tl.float16),
                tl.trans(k_tile).to(tl.float16),
            ).to(tl.float32)  # [Q_TILE_SIZE, BLOCK_SIZE]

            # Apply attention scale.
            qk = qk * scale

            # Mask positions that are outside the context or in a skipped block
            # with -inf.  exp(-inf)=0 → zero softmax weight → no-op accumulation.
            # load_mask [BLOCK_SIZE] is broadcast across Q_TILE_SIZE rows.
            qk = tl.where(load_mask[None, :], qk, float("-inf"))

            # ------------------------------------------------------------------
            # FA-2 online softmax update
            # ------------------------------------------------------------------
            # 1. New running max per query position.
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))  # [Q_TILE_SIZE]

            # 2. Rescale factor for previously accumulated values.
            #    alpha = exp(m_old - m_new)  — corrects the old accumulator.
            alpha = tl.exp(m_i - m_new)                   # [Q_TILE_SIZE]

            # 3. Softmax weights for the current KV block.
            p = tl.exp(qk - m_new[:, None])               # [Q_TILE_SIZE, BLOCK_SIZE]

            # 4. Update running denominator.
            l_i = alpha * l_i + tl.sum(p, axis=1)         # [Q_TILE_SIZE]

            # ------------------------------------------------------------------
            # Load V tile:  [BLOCK_SIZE, HEAD_DIM]  (fp16)
            # ------------------------------------------------------------------
            v_base = (
                V_cache_ptr
                + phys_block_idx * stride_kb
                + head_idx       * stride_kh
            )
            v_ptrs = (
                v_base
                + pos_offs[:, None] * stride_ks
                + d_offs[None, :]   * stride_kd
            )
            v_tile = tl.load(
                v_ptrs,
                mask=load_mask[:, None],
                other=0.0,
            ).to(tl.float32)   # [BLOCK_SIZE, HEAD_DIM]

            # 5. Update weighted-value accumulator.
            #    acc = alpha * acc + p @ V
            #    alpha[:, None] broadcasts across HEAD_DIM.
            acc_i = alpha[:, None] * acc_i + tl.dot(
                p.to(tl.float16),
                v_tile.to(tl.float16),
            ).to(tl.float32)   # [Q_TILE_SIZE, HEAD_DIM]

            # 6. Store updated running max for the next iteration.
            m_i = m_new

        # ------------------------------------------------------------------
        # Normalise: acc / l  (safe divide — l=0 only if all positions were masked)
        # ------------------------------------------------------------------
        # Avoid division by zero for fully-masked query positions.
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        out_tile = acc_i / l_safe[:, None]               # [Q_TILE_SIZE, HEAD_DIM]

        # ------------------------------------------------------------------
        # Store output tile  (fp32 → fp16)
        # ------------------------------------------------------------------
        out_base = (
            Out_ptr
            + batch_idx * stride_ob
            + head_idx  * stride_oh
        )
        out_ptrs = (
            out_base
            + q_offs[:, None] * stride_os
            + d_offs[None, :] * stride_od
        )
        tl.store(
            out_ptrs,
            out_tile.to(tl.float16),
            mask=q_mask[:, None] & (d_offs[None, :] < HEAD_DIM),
        )


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------

def triton_paged_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_seq_len: int,
    window_size: int,
    num_sink_tokens: int = 4,
    block_size: int = 16,
) -> torch.Tensor:
    """Compute paged sparse attention using a Triton FA-2 kernel.

    Replaces ``memopt_C.dynamic_sparse_attention`` with a pure-Triton
    implementation that supports Ada-KV block skipping and attention sinks.

    Falls back to SDPA-with-KV-truncation when Triton is not installed or
    the inputs are not on CUDA.

    Args:
        query: Query tensor of shape ``(batch, q_len, num_heads, head_dim)``
            in fp16.
        key_cache: Paged key cache of shape
            ``(num_kv_blocks, block_size, num_heads, head_dim)`` in fp16.
        value_cache: Same shape as ``key_cache``, in fp16.
        block_tables: Physical block indices per sequence, shape
            ``(batch, max_blocks_per_seq)``, dtype int32.
        context_lens: Number of valid KV tokens per sequence, shape ``(batch,)``,
            dtype int32.
        max_seq_len: Maximum sequence length; used to detect when windowing is
            effectively disabled (``window_size == max_seq_len``).
        window_size: Number of most-recent tokens each query can attend to.
            Pass ``max_seq_len`` to disable windowing (attend to all tokens).
        num_sink_tokens: Number of initial tokens that are structurally
            protected from Ada-KV eviction.  Default ``4``.
        block_size: Number of token slots per KV page.  Must match the pool
            initialisation parameter.  Default ``16``.

    Returns:
        Output tensor of shape ``(batch, q_len, num_heads, head_dim)`` in fp16.

    Raises:
        RuntimeError: If ``query`` is not on a CUDA device and Triton is
            available (Triton requires CUDA).
    """
    if not TRITON_AVAILABLE or not query.is_cuda:
        # Pure-PyTorch fallback — mirrors DynamicAttention._sdpa_forward logic.
        return _sdpa_fallback(
            query, key_cache, value_cache,
            block_tables, context_lens,
            max_seq_len, window_size, num_sink_tokens, block_size,
        )

    batch, q_len, num_heads, head_dim = query.shape
    _, max_blocks_per_seq = block_tables.shape

    # Ensure fp16.
    query      = query.to(torch.float16)
    key_cache  = key_cache.to(torch.float16)
    value_cache = value_cache.to(torch.float16)

    # Allocate output tensor.
    out = torch.empty_like(query)  # [batch, q_len, num_heads, head_dim] fp16

    # Attention scale factor.
    scale = head_dim ** -0.5

    # Tile size for the query dimension.
    Q_TILE_SIZE: int = 64

    # Pad head_dim to the next power of two for the constexpr tiling.
    head_dim_padded: int = _next_power_of_two(head_dim)

    # Grid: one program per (batch * num_heads, q_tile).
    grid = (batch * num_heads, math.ceil(q_len / Q_TILE_SIZE))

    _flash_attn_paged_sparse_kernel[grid](
        # Pointers
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        out,
        # Scalar dims
        batch,
        q_len,
        num_heads,
        max_blocks_per_seq,
        window_size,
        num_sink_tokens,
        scale,
        # Q strides
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        # K/V cache strides (shared layout)
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        # block_tables strides
        block_tables.stride(0),
        block_tables.stride(1),
        # output strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        # Constexpr tile sizes
        BLOCK_SIZE=block_size,
        Q_TILE_SIZE=Q_TILE_SIZE,
        HEAD_DIM=head_dim_padded,
    )

    return out


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback (used when Triton is unavailable or on CPU)
# ---------------------------------------------------------------------------

def _sdpa_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_seq_len: int,
    window_size: int,
    num_sink_tokens: int,
    block_size: int,
) -> torch.Tensor:
    """SDPA-based fallback that honours paged KV layout and Ada-KV windowing.

    Reconstructs a dense K/V matrix from the paged cache for each sequence,
    then runs :func:`torch.nn.functional.scaled_dot_product_attention`.

    Args:
        query: Shape ``(batch, q_len, num_heads, head_dim)`` fp16.
        key_cache: Shape ``(num_kv_blocks, block_size, num_heads, head_dim)`` fp16.
        value_cache: Same as ``key_cache``.
        block_tables: Shape ``(batch, max_blocks_per_seq)`` int32.
        context_lens: Shape ``(batch,)`` int32.
        max_seq_len: Integer maximum sequence length.
        window_size: Sliding-window size; equal to ``max_seq_len`` to disable.
        num_sink_tokens: Sink token count.
        block_size: KV page size.

    Returns:
        Shape ``(batch, q_len, num_heads, head_dim)`` fp16.
    """
    batch, q_len, num_heads, head_dim = query.shape
    scale = head_dim ** -0.5
    outputs = []

    for b in range(batch):
        ctx_len = int(context_lens[b].item())
        num_blocks = math.ceil(ctx_len / block_size)

        # Gather K and V slabs for this sequence.
        k_slabs, v_slabs = [], []
        for blk in range(num_blocks):
            phys = int(block_tables[b, blk].item())
            k_slabs.append(key_cache[phys])   # [block_size, num_heads, head_dim]
            v_slabs.append(value_cache[phys])

        # k_dense: [ctx_len, num_heads, head_dim]
        k_dense = torch.cat(k_slabs, dim=0)[:ctx_len]
        v_dense = torch.cat(v_slabs, dim=0)[:ctx_len]

        # Apply Ada-KV windowing: retain sink tokens + last window_size tokens.
        if window_size < max_seq_len and ctx_len > window_size:
            # Keep sink tokens and the recent window, deduplicated.
            sink_end = min(num_sink_tokens, ctx_len)
            win_start = max(ctx_len - window_size, sink_end)

            retain_indices = list(range(sink_end)) + list(range(win_start, ctx_len))
            retain = torch.tensor(retain_indices, dtype=torch.long,
                                  device=k_dense.device)
            k_dense = k_dense[retain]
            v_dense = v_dense[retain]

        # Reshape for SDPA: (1, num_heads, kv_len, head_dim)
        q_b = query[b:b+1].transpose(1, 2)  # [1, num_heads, q_len, head_dim]
        k_b = k_dense.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads, kv_len, head_dim]
        v_b = v_dense.permute(1, 0, 2).unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(
            q_b.to(torch.float32),
            k_b.to(torch.float32),
            v_b.to(torch.float32),
            scale=scale,
            dropout_p=0.0,
        )  # [1, num_heads, q_len, head_dim]

        # Back to [1, q_len, num_heads, head_dim]
        outputs.append(attn_out.transpose(1, 2).to(torch.float16))

    return torch.cat(outputs, dim=0)  # [batch, q_len, num_heads, head_dim]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n.

    Args:
        n: Positive integer.

    Returns:
        Smallest power of two that is >= n.
    """
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Mathematical equivalence smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Smoke test: compare Triton kernel output against SDPA reference.

    Constructs a small dense KV scenario and checks that
    ``torch.allclose(triton_out, ref_out, atol=1e-2)`` passes.  Tolerance is
    1e-2 because fp16 intermediate arithmetic accumulates small rounding errors
    relative to an fp32 SDPA reference.
    """
    import sys

    BATCH     = 2
    NUM_HEADS = 4
    HEAD_DIM  = 32
    SEQ_LEN   = 128
    BLOCK_SZ  = 16
    NUM_SINK  = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available — running SDPA fallback comparison only.")

    # Number of KV pages needed.
    num_kv_blocks = math.ceil(SEQ_LEN / BLOCK_SZ) * BATCH  # over-allocate

    # Build random fp16 Q/K/V.
    torch.manual_seed(42)
    query = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM,
                        dtype=torch.float16, device=device)
    # Dense K/V for reference.
    k_dense = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM,
                          dtype=torch.float16, device=device)
    v_dense = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM,
                          dtype=torch.float16, device=device)

    # Build paged KV cache with sequential block allocation.
    blocks_per_seq = math.ceil(SEQ_LEN / BLOCK_SZ)
    total_blocks   = BATCH * blocks_per_seq

    key_cache   = torch.zeros(total_blocks, BLOCK_SZ, NUM_HEADS, HEAD_DIM,
                              dtype=torch.float16, device=device)
    value_cache = torch.zeros(total_blocks, BLOCK_SZ, NUM_HEADS, HEAD_DIM,
                              dtype=torch.float16, device=device)

    # Populate block_tables sequentially and fill the paged cache.
    block_tables = torch.zeros(BATCH, blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(BATCH):
        for blk in range(blocks_per_seq):
            phys = b * blocks_per_seq + blk
            block_tables[b, blk] = phys
            tok_s = blk * BLOCK_SZ
            tok_e = min(tok_s + BLOCK_SZ, SEQ_LEN)
            # Copy the dense slice into the paged cache.
            key_cache[phys, :tok_e - tok_s] = k_dense[b, tok_s:tok_e]
            value_cache[phys, :tok_e - tok_s] = v_dense[b, tok_s:tok_e]

    context_lens = torch.full((BATCH,), SEQ_LEN, dtype=torch.int32, device=device)

    # --- Triton / fallback output ---
    triton_out = triton_paged_sparse_attention(
        query, key_cache, value_cache,
        block_tables, context_lens,
        max_seq_len=SEQ_LEN,
        window_size=SEQ_LEN,   # no windowing → full attention
        num_sink_tokens=NUM_SINK,
        block_size=BLOCK_SZ,
    )  # [batch, q_len, num_heads, head_dim]

    # --- SDPA reference (full attention, dense) ---
    # SDPA expects (batch, num_heads, seq, head_dim)
    q_sdpa = query.permute(0, 2, 1, 3).to(torch.float32)
    k_sdpa = k_dense.permute(0, 2, 1, 3).to(torch.float32)
    v_sdpa = v_dense.permute(0, 2, 1, 3).to(torch.float32)

    ref_out = F.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        scale=HEAD_DIM ** -0.5,
        dropout_p=0.0,
    )  # [batch, num_heads, seq, head_dim]
    # Back to [batch, seq, num_heads, head_dim] fp16
    ref_out = ref_out.permute(0, 2, 1, 3).to(torch.float16)

    match = torch.allclose(triton_out, ref_out, atol=1e-2)
    print(f"torch.allclose(triton_out, ref_out, atol=1e-2) = {match}")
    if not match:
        max_err = (triton_out.float() - ref_out.float()).abs().max().item()
        print(f"Max absolute error: {max_err:.4f}")
        sys.exit(1)
    print("Smoke test passed.")
