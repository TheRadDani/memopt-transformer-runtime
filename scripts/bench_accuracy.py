"""Accuracy / perplexity benchmark: Baseline vs StreamingLLM vs MemOpt (Full).

Location: scripts/bench_accuracy.py
Summary: Sliding-window perplexity evaluation suite that compares three
    Transformer configurations on WikiText-2 and C4 datasets.  Supports both
    HuggingFace pretrained models (GPT-2, etc.) and a self-contained tiny
    decoder for smoke-testing without network access.

Configurations evaluated
------------------------
* ``baseline``       — Standard causal SDPA; no KV eviction or masking.
* ``streaming_llm``  — Attention-sink masking (arXiv 2309.17453): keeps first
                        ``num_sink_tokens`` + last ``window_size`` tokens.
* ``memopt_full``    — DynamicAttention (Ada-KV pruning + W4A16 quant) from
                        ``src/memopt/`` + MemoryScheduler.

Used with / by
--------------
- src/memopt/attention.py  — DynamicAttention module (memopt_full config)
- src/memopt/scheduler.py  — MemoryScheduler (memopt_full config)
- scripts/bench_memory.py  — shares architectural conventions and style

Usage
-----
    # Smoke-test: no network, tiny sequences
    python scripts/bench_accuracy.py --dry-run --no-plot --device cpu

    # Real HuggingFace GPT-2, WikiText-2 only
    python scripts/bench_accuracy.py --model-name gpt2 --evaluate-dataset wikitext2

    # Full benchmark
    python scripts/bench_accuracy.py --model-name gpt2 --context-lengths 1024 4096 8192

    # See all options
    python scripts/bench_accuracy.py --help
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve the project src/ directory so the script works when run from the
# repo root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional memopt imports — degrade gracefully when C++ extension is absent.
# ---------------------------------------------------------------------------
try:
    from memopt.attention import DynamicAttention
    from memopt.scheduler import MemoryScheduler, SchedulerConfig

    _MEMOPT_AVAILABLE: bool = True
except Exception as exc:  # noqa: BLE001
    warnings.warn(
        f"Could not import memopt Python layer ({exc}). "
        "memopt_full config will use an inline SDPA stub.",
        ImportWarning,
        stacklevel=1,
    )
    _MEMOPT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional HuggingFace transformers + datasets
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    _TRANSFORMERS_AVAILABLE: bool = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from datasets import load_dataset  # type: ignore[import]

    _DATASETS_AVAILABLE: bool = True
except ImportError:
    _DATASETS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional matplotlib / seaborn
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend — always works
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE: bool = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "matplotlib is not installed — plots will be skipped. "
        "Install with: pip install matplotlib",
        ImportWarning,
        stacklevel=1,
    )

try:
    import seaborn as sns  # type: ignore[import]

    _SEABORN_AVAILABLE: bool = True
except ImportError:
    _SEABORN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("bench_accuracy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VOCAB_SIZE: int = 50257  # GPT-2 vocab size; used by the tiny self-contained decoder
_TINY_D_MODEL: int = 256
_TINY_N_HEADS: int = 4
_TINY_N_LAYERS: int = 4
_MAX_SEQ_LEN: int = 32768  # positional-embedding upper bound for tiny decoder

_DEFAULT_CONTEXT_LENGTHS: List[int] = [1024, 4096, 8192]
_DRY_RUN_CONTEXT_LENGTHS: List[int] = [64, 128]
_DEFAULT_STRIDE: int = 512
_DEFAULT_NUM_SINK_TOKENS: int = 4
_DEFAULT_WINDOW_SIZE: int = 256  # StreamingLLM recency window

# Config names — canonical strings used as JSON keys and in logging.
_CFG_BASELINE: str = "baseline"
_CFG_STREAMING: str = "streaming_llm"
_CFG_MEMOPT: str = "memopt_full"

# Plot style per config — consistent across all subplots.
_CONFIG_STYLES: Dict[str, dict] = {
    _CFG_BASELINE: {
        "color": "C0",
        "marker": "o",
        "linestyle": "-",
        "label": "Baseline (SDPA)",
    },
    _CFG_STREAMING: {
        "color": "C1",
        "marker": "s",
        "linestyle": "--",
        "label": "StreamingLLM (sinks + recency)",
    },
    _CFG_MEMOPT: {
        "color": "C2",
        "marker": "^",
        "linestyle": ":",
        "label": "MemOpt Full (Ada-KV + W4A16)",
    },
}


# ===========================================================================
# Self-contained tiny decoder (no HuggingFace required)
# ===========================================================================


class _BaselineAttention(nn.Module):
    """Standard causal SDPA — no KV eviction or masking.

    Args:
        d_model: Embedding dimension ``D``.
        n_heads: Number of attention heads ``H``.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal SDPA forward.

        Args:
            x: Shape ``(B, S, D)``.

        Returns:
            Shape ``(B, S, D)``.
        """
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, self.d_model))


class _StreamingLLMAttention(nn.Module):
    """StreamingLLM attention-sink masking (arXiv 2309.17453).

    Retains the first ``num_sink_tokens`` (attention sinks) and the last
    ``window_size`` (recency window) KV positions.  All positions outside these
    two sets receive an additive ``-inf`` mask before softmax, replicating the
    mathematical output of StreamingLLM without altering memory layout.

    This is implemented as an attention-level mask so the simulation operates
    correctly at the attention-score level, not by truncating the input sequence.

    Args:
        d_model: Embedding dimension ``D``.
        n_heads: Number of attention heads ``H``.
        num_sink_tokens: Number of leading sink tokens to always attend to.
        window_size: Number of most-recent tokens to retain in the KV window.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_sink_tokens: int,
        window_size: int,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _build_sink_mask(self, S: int, device: torch.device) -> torch.Tensor:
        """Build an additive attention mask for sink + recency retention.

        A key position ``j`` is visible to query ``i`` when:
          - ``j <= i`` (causal), AND
          - ``j < num_sink_tokens`` (sink) OR ``j >= i - window_size + 1`` (recent).

        Masked positions receive ``-inf``; visible positions receive ``0``.

        Args:
            S: Sequence length.
            device: Target device for the mask tensor.

        Returns:
            Float mask of shape ``(S, S)`` with ``0`` or ``-inf`` entries.
        """
        i_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

        is_causal = j_idx <= i_idx  # (S, S)
        is_sink = j_idx < self.num_sink_tokens  # (S, S)
        is_recent = j_idx >= (i_idx - self.window_size + 1)  # (S, S)

        visible = is_causal & (is_sink | is_recent)  # (S, S)

        mask = torch.zeros(S, S, device=device)
        mask = mask.masked_fill(~visible, float("-inf"))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """StreamingLLM-masked attention forward.

        Args:
            x: Shape ``(B, S, D)``.

        Returns:
            Shape ``(B, S, D)``.
        """
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        mask = self._build_sink_mask(S, x.device)  # (S, S)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S) — broadcast over B, H

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, self.d_model))


class _SDPABlock(nn.Module):
    """Single Transformer block with injected attention module.

    Args:
        d_model: Embedding dimension ``D``.
        n_heads: Number of attention heads ``H``.
        attn_module: An ``nn.Module`` with signature ``(x: Tensor[B,S,D])
            -> Tensor[B,S,D]``.  When ``None``, falls back to
            ``nn.MultiheadAttention`` (MHA API).
        d_ff: Feed-forward hidden dimension (defaults to ``4 * d_model``).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_module: Optional[nn.Module] = None,
        d_ff: Optional[int] = None,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        if attn_module is not None:
            self.attn: nn.Module = attn_module
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=0.0,
                batch_first=True,
            )
        self._uses_mha: bool = isinstance(self.attn, nn.MultiheadAttention)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm residual forward.

        Args:
            x: Shape ``(B, S, D)``.

        Returns:
            Shape ``(B, S, D)``.
        """
        residual = x
        x_norm = self.norm1(x)
        if self._uses_mha:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        else:
            attn_out = self.attn(x_norm)
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class _TinyDecoder(nn.Module):
    """Minimal autoregressive Transformer decoder.

    Used when HuggingFace models are not available (dry-run, no-network).
    All three configs share this backbone and differ only in the attention
    module supplied via ``attn_factory``.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of stacked Transformer blocks.
        attn_factory: Callable ``(layer_idx: int) -> nn.Module`` returning the
            attention module for each block.
        vocab_size: Token vocabulary size.
        max_seq_len: Maximum sequence length for positional embedding.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        attn_factory: Callable[[int], nn.Module],
        vocab_size: int = _VOCAB_SIZE,
        max_seq_len: int = _MAX_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                _SDPABlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_module=attn_factory(layer_idx),
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning next-token logits.

        Args:
            input_ids: Integer token ids of shape ``(B, S)``.

        Returns:
            Logits of shape ``(B, S, vocab_size)``.
        """
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ===========================================================================
# Model / wrapper factories for each config
# ===========================================================================


def _build_tiny_baseline(d_model: int, n_heads: int, n_layers: int) -> _TinyDecoder:
    """Build the Baseline tiny decoder using standard causal SDPA.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of layers.

    Returns:
        Baseline ``_TinyDecoder`` instance.
    """

    def _factory(_: int) -> nn.Module:
        return _BaselineAttention(d_model=d_model, n_heads=n_heads)

    return _TinyDecoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        attn_factory=_factory,
    )


def _build_tiny_streamingllm(
    d_model: int,
    n_heads: int,
    n_layers: int,
    num_sink_tokens: int,
    window_size: int,
) -> _TinyDecoder:
    """Build the StreamingLLM tiny decoder with attention-sink masking.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of layers.
        num_sink_tokens: Number of leading sink tokens.
        window_size: Recency window size in tokens.

    Returns:
        StreamingLLM ``_TinyDecoder`` instance.
    """

    def _factory(_: int) -> nn.Module:
        return _StreamingLLMAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_sink_tokens=num_sink_tokens,
            window_size=window_size,
        )

    return _TinyDecoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        attn_factory=_factory,
    )


def _build_tiny_memopt(d_model: int, n_heads: int, n_layers: int) -> _TinyDecoder:
    """Build the MemOpt tiny decoder using DynamicAttention (SDPA fallback).

    Uses ``DynamicAttention`` when the memopt Python layer is available;
    otherwise falls back to the inline ``_BaselineAttention`` stub so the
    benchmark still runs without the C++ extension.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of layers.

    Returns:
        MemOpt ``_TinyDecoder`` instance.
    """
    if _MEMOPT_AVAILABLE:

        def _factory(_: int) -> nn.Module:
            return DynamicAttention(embed_dim=d_model, num_heads=n_heads)

    else:

        def _factory(_: int) -> nn.Module:  # type: ignore[misc]
            return _BaselineAttention(d_model=d_model, n_heads=n_heads)

    return _TinyDecoder(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        attn_factory=_factory,
    )


# ---------------------------------------------------------------------------
# HuggingFace model wrappers
# ---------------------------------------------------------------------------


def _build_sink_mask_2d(
    S: int,
    num_sink_tokens: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a 2-D additive attention mask for sink + recency retention.

    A key position ``j`` is visible to query ``i`` when:
      - ``j <= i`` (causal), AND
      - ``j < num_sink_tokens`` (sink) OR ``j >= i - window_size + 1`` (recent).

    Masked positions receive ``-inf``; visible positions receive ``0``.

    Args:
        S: Sequence length.
        num_sink_tokens: Number of leading sink tokens to always attend to.
        window_size: Number of most-recent tokens to retain.
        device: Target device for the mask tensor.

    Returns:
        Float mask of shape ``(S, S)`` with ``0`` or ``-inf`` entries.
    """
    i_idx = torch.arange(S, device=device).unsqueeze(1)  # (S, 1)
    j_idx = torch.arange(S, device=device).unsqueeze(0)  # (1, S)

    is_causal = j_idx <= i_idx
    is_sink = j_idx < num_sink_tokens
    is_recent = j_idx >= (i_idx - window_size + 1)

    visible = is_causal & (is_sink | is_recent)

    mask = torch.zeros(S, S, dtype=torch.float32, device=device)
    mask = mask.masked_fill(~visible, float("-inf"))
    return mask


class _HFStreamingLLMWrapper(nn.Module):
    """Thin wrapper around a HuggingFace GPT-2 CausalLM applying StreamingLLM masking.

    Monkey-patches each ``GPT2Attention.forward`` method in the model to
    intercept the ``attention_mask`` argument and combine it with a sink +
    recency additive mask before delegating to the original forward.

    This avoids the 4-D ``attention_mask`` shape mismatch that occurs when
    passing a float mask via ``model(input_ids, attention_mask=mask)``, and is
    compatible with newer transformers versions that use an
    ``attention_interface`` dispatch rather than a ``_attn`` helper.

    The original ``forward`` method is saved per-layer and restored when this
    wrapper is garbage-collected, so the deep-copied ``hf_model`` is not left
    in a permanently patched state.

    Args:
        hf_model: A deep-copied ``AutoModelForCausalLM`` instance (GPT-2 family).
        num_sink_tokens: Number of leading sink tokens.
        window_size: Recency window size in tokens.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        num_sink_tokens: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.model = hf_model
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size

        # Save (attn_layer, original_forward) pairs for unpatching on __del__.
        self._patched_attn_layers: list = []
        self._patch_gpt2_attention_layers()

    def _patch_gpt2_attention_layers(self) -> None:
        """Monkey-patch each GPT2Attention.forward to inject sink masking.

        For each transformer block in ``self.model.transformer.h``, wraps the
        ``forward()`` method to build a ``(1, 1, S, S)`` additive sink+recency
        mask and merge it into the ``attention_mask`` argument that arrives at
        the layer (already 4-D and in additive float form after GPT2Model's
        pre-processing).
        """
        try:
            blocks = self.model.transformer.h
        except AttributeError:
            logger.warning(
                "_HFStreamingLLMWrapper: model has no .transformer.h — "
                "sink-mask patching skipped; StreamingLLM will behave like baseline."
            )
            return

        num_sink = self.num_sink_tokens
        window = self.window_size

        for block in blocks:
            attn_layer = block.attn
            original_forward = attn_layer.forward

            # Build a closure that captures original_forward per layer.
            def _make_patched_forward(orig_fwd):
                def _patched_forward(hidden_states, attention_mask=None, **kwargs):
                    """Inject sink+recency mask into attention_mask before calling forward."""
                    S = hidden_states.shape[1]  # sequence length from (B, S, D)
                    sink_mask = _build_sink_mask_2d(
                        S=S,
                        num_sink_tokens=num_sink,
                        window_size=window,
                        device=hidden_states.device,
                    )  # (S, S)
                    sink_mask_4d = sink_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

                    # GPT2Model pre-processes attention_mask to additive float form
                    # (0 for visible positions, very negative for masked positions).
                    # Combine our sink mask additively so masked positions stay masked.
                    if attention_mask is not None:
                        combined_mask = attention_mask + sink_mask_4d
                    else:
                        combined_mask = sink_mask_4d

                    return orig_fwd(hidden_states, attention_mask=combined_mask, **kwargs)

                return _patched_forward

            patched_fwd = _make_patched_forward(original_forward)
            attn_layer.forward = patched_fwd
            self._patched_attn_layers.append((attn_layer, original_forward))

        logger.info(
            "_HFStreamingLLMWrapper: patched forward on %d GPT2Attention layers "
            "(num_sink_tokens=%d, window_size=%d).",
            len(self._patched_attn_layers),
            num_sink,
            window,
        )

    def _unpatch_gpt2_attention_layers(self) -> None:
        """Restore original ``forward`` methods on all patched layers."""
        for attn_layer, original_fwd in self._patched_attn_layers:
            attn_layer.forward = original_fwd
        self._patched_attn_layers.clear()

    def __del__(self) -> None:
        """Restore original forward methods when this wrapper is garbage-collected."""
        try:
            self._unpatch_gpt2_attention_layers()
        except Exception:  # noqa: BLE001
            pass  # Best-effort cleanup; do not raise in __del__

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the HF model with StreamingLLM sink masking applied via monkey-patch.

        Args:
            input_ids: Token ids of shape ``(B, S)``.

        Returns:
            Logits of shape ``(B, S, vocab_size)``.
        """
        out = self.model(input_ids)
        return out.logits


class _HFBaselineWrapper(nn.Module):
    """Thin wrapper around a HuggingFace CausalLM with standard causal masking.

    Args:
        hf_model: A loaded ``AutoModelForCausalLM`` instance.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the HF model and return logits.

        Args:
            input_ids: Token ids of shape ``(B, S)``.

        Returns:
            Logits of shape ``(B, S, vocab_size)``.
        """
        out = self.model(input_ids)
        return out.logits


# ===========================================================================
# Dataset loading and tokenization
# ===========================================================================


def _load_wikitext2_tokens(
    dry_run: bool,
    tokenizer: Optional[object] = None,
) -> List[int]:
    """Load WikiText-2 validation text and tokenize.

    When ``tokenizer`` is provided uses HuggingFace tokenization (BPE/WordPiece
    etc.); otherwise falls back to a simple word-level vocabulary mapping clipped
    to ``[0, _VOCAB_SIZE - 1]``.  In dry-run mode, returns a small random sequence.

    Args:
        dry_run: When ``True``, returns a tiny random token sequence (no I/O).
        tokenizer: Optional HuggingFace tokenizer instance.

    Returns:
        Flat list of integer token IDs.

    Raises:
        RuntimeError: If ``datasets`` is not installed and ``dry_run`` is False.
    """
    if dry_run:
        return _generate_random_tokens(n=2048)

    if not _DATASETS_AVAILABLE:
        raise RuntimeError(
            "The 'datasets' package is required to load WikiText-2.  "
            "Install with: pip install datasets"
        )

    logger.info("Loading WikiText-2 validation split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = " ".join(row["text"] for row in dataset if row["text"].strip())
    return _encode_text(text, tokenizer)


def _load_c4_tokens(
    dry_run: bool,
    tokenizer: Optional[object] = None,
) -> List[int]:
    """Load C4 English validation text (first 1 000 examples) and tokenize.

    Args:
        dry_run: When ``True``, returns a tiny random token sequence (no I/O).
        tokenizer: Optional HuggingFace tokenizer instance.

    Returns:
        Flat list of integer token IDs.

    Raises:
        RuntimeError: If ``datasets`` is not installed and ``dry_run`` is False.
    """
    if dry_run:
        return _generate_random_tokens(n=2048)

    if not _DATASETS_AVAILABLE:
        raise RuntimeError(
            "The 'datasets' package is required to load C4.  "
            "Install with: pip install datasets"
        )

    logger.info("Loading C4 English validation split (streaming, first 1 000 examples)...")
    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts: List[str] = []
    for i, row in enumerate(dataset):
        if i >= 1000:
            break
        texts.append(row["text"])
    text = " ".join(texts)
    return _encode_text(text, tokenizer)


def _encode_text(text: str, tokenizer: Optional[object]) -> List[int]:
    """Encode ``text`` using ``tokenizer`` if available, else word-level fallback.

    HuggingFace tokenizers produce real BPE/WordPiece IDs that are valid inputs
    to pretrained models.  When no tokenizer is provided, uses a stable word-
    to-int mapping clipped to ``[0, _VOCAB_SIZE - 1]`` — suitable for the
    self-contained tiny decoder.

    Args:
        text: Raw corpus text string.
        tokenizer: Optional HuggingFace tokenizer instance.

    Returns:
        Flat list of integer token IDs.
    """
    if tokenizer is not None:
        encoded = tokenizer.encode(text)
        if isinstance(encoded, list):
            return [int(t) for t in encoded]
        # Some tokenizers return a tensor or BatchEncoding.
        if hasattr(encoded, "input_ids"):
            return [int(t) for t in encoded.input_ids]
        return [int(t) for t in list(encoded)]

    words = text.split()
    if not words:
        return []
    vocab = {w: i % _VOCAB_SIZE for i, w in enumerate(sorted(set(words)))}
    return [vocab[w] for w in words]


def _generate_random_tokens(n: int) -> List[int]:
    """Generate ``n`` random token IDs in ``[0, _VOCAB_SIZE)``.

    Args:
        n: Number of tokens to generate.

    Returns:
        List of ``n`` random integer token IDs (seeded for reproducibility).
    """
    import random

    rng = random.Random(42)
    return [rng.randrange(_VOCAB_SIZE) for _ in range(n)]


# ===========================================================================
# Perplexity evaluation
# ===========================================================================


def _iter_chunks(
    tokens: List[int],
    context_length: int,
    stride: int,
) -> Iterator[Tuple[List[int], int]]:
    """Yield ``(chunk, n_predict_tokens)`` pairs using a sliding window.

    Follows the Hugging Face sliding-window PPL methodology:
    - First window: predicts all ``context_length - 1`` non-seed positions.
    - Subsequent windows: advance by ``stride``; predict only the ``stride``
      new tokens at the window's trailing edge.

    Args:
        tokens: Flat list of all token IDs.
        context_length: Window size ``S``.
        stride: Number of new tokens per step.

    Yields:
        Tuple of ``(token_window, n_predictable_tokens)``.
    """
    total = len(tokens)
    start = 0
    first = True
    while start < total - 1:
        end = min(start + context_length, total)
        chunk = tokens[start:end]
        if len(chunk) < 2:
            break
        n_predict = len(chunk) - 1 if first else min(stride, len(chunk) - 1)
        yield chunk, n_predict
        first = False
        if end >= total:
            break
        start += stride


def evaluate_perplexity(
    model: nn.Module,
    tokens: List[int],
    context_length: int,
    stride: int,
    device: torch.device,
    max_chunks: int = 50,
) -> float:
    """Compute sliding-window perplexity of ``model`` on ``tokens``.

    PPL = exp( (1 / N) * sum_i NLL_i )

    where ``NLL_i`` is the cross-entropy loss summed over the ``n_predict``
    non-context positions in each window.

    OOM is caught per-chunk and results in the chunk being skipped (logged
    at WARNING level).  If all chunks fail, ``float("nan")`` is returned.

    Args:
        model: A module returning logits ``(B, S, V)`` from input_ids ``(B, S)``.
        tokens: Flat list of integer token IDs.
        context_length: Sliding-window size ``S``.
        stride: Number of new tokens per window step.
        device: Torch device for forward passes.
        max_chunks: Maximum number of chunks to evaluate (caps wall-clock time).

    Returns:
        Perplexity as a Python float.  ``float("nan")`` if evaluation failed.
    """
    model.eval()
    total_loss: float = 0.0
    total_tokens: int = 0

    for chunk_idx, (chunk, n_predict) in enumerate(
        _iter_chunks(tokens, context_length, stride)
    ):
        if chunk_idx >= max_chunks:
            break
        if len(chunk) < 2:
            continue

        # shape: (1, S)
        input_ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)

        try:
            with torch.no_grad():
                logits = model(input_ids)  # (1, S, V)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "PPL eval OOM at context_length=%d chunk=%d — skipping chunk.",
                context_length,
                chunk_idx,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PPL forward pass failed (context_length=%d chunk=%d): %s — skipping.",
                context_length,
                chunk_idx,
                exc,
            )
            continue

        vocab_size = logits.shape[-1]
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)  # (S-1, V)
        shift_labels = input_ids[:, 1:].reshape(-1)  # (S-1,)

        # Evaluate only the trailing n_predict positions (sliding-window semantics).
        if n_predict < shift_labels.shape[0]:
            shift_logits = shift_logits[-n_predict:]
            shift_labels = shift_labels[-n_predict:]

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.shape[0]

    if total_tokens == 0:
        logger.warning(
            "No tokens evaluated for context_length=%d — returning nan.", context_length
        )
        return float("nan")

    return math.exp(total_loss / total_tokens)


# ===========================================================================
# Benchmark orchestration
# ===========================================================================


def run_config_benchmark(
    config_name: str,
    model: nn.Module,
    tokens_by_dataset: Dict[str, List[int]],
    context_lengths: List[int],
    stride: int,
    device: torch.device,
    max_chunks: int,
    scheduler: Optional["MemoryScheduler"] = None,
) -> Dict[str, Dict[str, object]]:
    """Evaluate one model config across all datasets and context lengths.

    For each (dataset, context_length) pair, computes sliding-window PPL.
    Catches ``torch.cuda.OutOfMemoryError`` at the top level (per context
    length) and records ``{"status": "OOM"}`` so the benchmark continues.

    Args:
        config_name: Human-readable name used in logging (e.g. ``"baseline"``).
        model: The decoder model to evaluate.
        tokens_by_dataset: Dict mapping dataset name to flat token list.
        context_lengths: List of context lengths to evaluate.
        stride: Sliding-window stride for PPL computation.
        device: Torch device.
        max_chunks: Maximum chunks per (dataset, context_length) pair.
        scheduler: Optional ``MemoryScheduler`` to start/stop around evaluation.

    Returns:
        Nested dict: ``{dataset_name: {str(ctx_len): ppl_float | {"status": "OOM"}}}``.
    """
    model = model.to(device)
    model.eval()

    if scheduler is not None:
        scheduler.start()

    results: Dict[str, Dict[str, object]] = {}

    try:
        for dataset_name, tokens in tokens_by_dataset.items():
            results[dataset_name] = {}
            for ctx_len in context_lengths:
                logger.info(
                    "[%s] dataset=%s  context_length=%d  stride=%d",
                    config_name,
                    dataset_name,
                    ctx_len,
                    stride,
                )
                try:
                    ppl = evaluate_perplexity(
                        model=model,
                        tokens=tokens,
                        context_length=ctx_len,
                        stride=stride,
                        device=device,
                        max_chunks=max_chunks,
                    )
                    results[dataset_name][str(ctx_len)] = round(ppl, 4)
                    logger.info(
                        "[%s] dataset=%s  ctx=%d  ppl=%.4f",
                        config_name,
                        dataset_name,
                        ctx_len,
                        ppl,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "[%s] dataset=%s  ctx=%d  OOM — recording and continuing.",
                        config_name,
                        dataset_name,
                        ctx_len,
                    )
                    results[dataset_name][str(ctx_len)] = {"status": "OOM"}
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "[%s] dataset=%s  ctx=%d  unexpected error: %s",
                        config_name,
                        dataset_name,
                        ctx_len,
                        exc,
                    )
                    results[dataset_name][str(ctx_len)] = {"status": "ERROR", "msg": str(exc)}

                gc.collect()
    finally:
        if scheduler is not None:
            scheduler.stop()

    return results


# ===========================================================================
# Reporting
# ===========================================================================


def save_json(
    results: Dict[str, Dict[str, Dict[str, object]]],
    metadata: dict,
    output_path: Path,
) -> None:
    """Persist benchmark results and metadata to a JSON file.

    Output format::

        {
          "metadata": {"model": "...", "date": "...", "context_lengths": [...]},
          "results": {
            "wikitext2": {
              "baseline":      {"1024": 12.3},
              "streaming_llm": {"1024": 14.1},
              "memopt_full":   {"1024": 12.5}
            }
          }
        }

    Args:
        results: Nested ``{dataset: {config: {str(ctx_len): ppl | OOM_dict}}}``.
        metadata: Benchmark configuration / environment metadata.
        output_path: Destination ``.json`` file path.
    """
    payload = {"metadata": metadata, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Results saved to %s", output_path)


def print_summary_table(
    results: Dict[str, Dict[str, Dict[str, object]]],
    context_lengths: List[int],
) -> None:
    """Print a formatted PPL comparison table to stdout.

    Args:
        results: Nested ``{dataset: {config: {str(ctx_len): ppl | OOM_dict}}}``.
        context_lengths: Ordered context lengths for column headers.
    """
    col_w = 14
    configs = [_CFG_BASELINE, _CFG_STREAMING, _CFG_MEMOPT]

    for dataset_name, dataset_results in results.items():
        print()
        print(f"Perplexity Results — {dataset_name}")
        print("-" * (col_w + col_w * len(context_lengths)))
        header = f"{'config':<{col_w}}" + "".join(
            f"{cl:>{col_w}}" for cl in context_lengths
        )
        print(header)
        print("-" * len(header))

        for cfg in configs:
            if cfg not in dataset_results:
                continue
            cfg_results = dataset_results[cfg]
            row = f"{cfg:<{col_w}}"
            for cl in context_lengths:
                val = cfg_results.get(str(cl), "N/A")
                if isinstance(val, dict):
                    cell = val.get("status", "ERR")
                elif isinstance(val, float) and math.isnan(val):
                    cell = "nan"
                elif isinstance(val, float):
                    cell = f"{val:.2f}"
                else:
                    cell = str(val)
                row += f"{cell:>{col_w}}"
            print(row)

        print("-" * len(header))
    print()


def plot_dataset_results(
    dataset_name: str,
    context_lengths: List[int],
    results: Dict[str, Dict[str, object]],
    output_path: Path,
) -> None:
    """Save a PPL vs context-length line plot for one dataset.

    Args:
        dataset_name: Dataset label used in the figure title.
        context_lengths: X-axis values.
        results: ``{config_name: {str(ctx_len): ppl}}`` for this dataset.
        output_path: Destination PNG file path.
    """
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available — skipping plot for %s.", dataset_name)
        return

    if _SEABORN_AVAILABLE:
        sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))

    for config_name, ctx_ppl in results.items():
        style = _CONFIG_STYLES.get(config_name, {})
        y_vals: List[float] = []
        for cl in context_lengths:
            val = ctx_ppl.get(str(cl), float("nan"))
            if isinstance(val, dict):
                y_vals.append(float("nan"))
            else:
                y_vals.append(float(val))

        ax.plot(
            context_lengths,
            y_vals,
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=2,
            label=style.get("label", config_name),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title(f"Perplexity vs Context Length — {dataset_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", output_path)


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Accuracy / perplexity benchmark: Baseline vs StreamingLLM vs MemOpt. "
            "Evaluates sliding-window PPL on WikiText-2 and/or C4."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    def _dataset_choice(value: str) -> str:
        """Normalise dataset name; accept 'wikitext-2' as alias for 'wikitext2'."""
        normalised = value.lower().replace("-", "")
        valid = {"wikitext2", "c4", "both"}
        if normalised not in valid:
            raise argparse.ArgumentTypeError(
                f"invalid choice: '{value}' (choose from {sorted(valid)})"
            )
        return normalised

    parser.add_argument(
        "--evaluate-dataset",
        type=_dataset_choice,
        default="both",
        metavar="{wikitext2,c4,both}",
        help="Which dataset(s) to evaluate.",
    )

    def _context_length_list(value: str) -> List[int]:
        """Accept comma-separated integers."""
        parts = [v.strip() for v in value.split(",") if v.strip()]
        result: List[int] = []
        for p in parts:
            try:
                result.append(int(p))
            except ValueError:
                raise argparse.ArgumentTypeError(f"invalid int value: '{p}'")
        return result

    parser.add_argument(
        "--context-lengths",
        type=_context_length_list,
        nargs="+",
        default=_DEFAULT_CONTEXT_LENGTHS,
        metavar="N",
        help="Space- or comma-separated context lengths to benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for forward passes (PPL evaluation uses batch size 1).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        metavar="HF_MODEL_ID",
        help=(
            "HuggingFace model ID to use as the backbone (e.g. 'gpt2', "
            "'gpt2-medium').  When transformers is not installed, the script "
            "falls back to the self-contained tiny decoder automatically."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(_REPO_ROOT) / "results" / "accuracy_results.json",
        metavar="FILE",
        help="Path for JSON results output.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=_DEFAULT_STRIDE,
        metavar="N",
        help="Stride for sliding-window PPL evaluation.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of sliding-window chunks per (dataset, ctx_len) pair.",
    )
    parser.add_argument(
        "--num-sink-tokens",
        type=int,
        default=_DEFAULT_NUM_SINK_TOKENS,
        metavar="K",
        help="Number of attention sink tokens for the StreamingLLM config.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=_DEFAULT_WINDOW_SIZE,
        metavar="W",
        help="Recency window size (in tokens) for the StreamingLLM config.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving matplotlib figures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Use tiny fake data at seq_len=64,128 for smoke-testing.  "
            "No network or GPU required.  Overrides --context-lengths."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (e.g. 'cuda', 'cpu').",
    )
    return parser.parse_args()


# ===========================================================================
# GPT-2 DynamicAttention patching
# ===========================================================================


class _GPT2DynamicAttentionAdapter(nn.Module):
    """Adapter that wraps ``DynamicAttention`` with GPT-2's calling convention.

    GPT-2's ``GPT2Attention`` layer is called by the block as::

        attn(hidden_states, layer_past=..., attention_mask=..., head_mask=...,
             use_cache=..., output_attentions=...)

    and returns a tuple ``(attn_output, present, attn_weights_or_None)``.

    ``DynamicAttention`` expects ``forward(x: Tensor[B,S,D])`` and returns
    ``Tensor[B,S,D]``.  This adapter bridges the two conventions, discarding
    ``layer_past`` / ``use_cache`` (no KV-cache pass-through needed for batch
    perplexity evaluation) and returning the tuple format GPT-2 expects.

    Args:
        dynamic_attn: A ``DynamicAttention`` instance.
    """

    def __init__(self, dynamic_attn: "DynamicAttention") -> None:
        super().__init__()
        self.dynamic_attn = dynamic_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward through DynamicAttention, returning GPT-2 tuple format.

        Args:
            hidden_states: ``(B, S, D)`` input to the attention layer.
            layer_past: Ignored (no KV-cache pass-through).
            attention_mask: Ignored (DynamicAttention uses its own mask).
            head_mask: Ignored.
            encoder_hidden_states: Ignored (self-attention only).
            encoder_attention_mask: Ignored.
            use_cache: Ignored.
            output_attentions: Ignored.
            **kwargs: Any additional kwargs passed by newer transformers versions.

        Returns:
            Tuple of ``(attn_output,)`` — single-element tuple matching the
            minimum expected by ``GPT2Block``.
        """
        attn_output = self.dynamic_attn(hidden_states)
        # GPT2Block unpacks: outputs = attn(hidden, ...) → outputs[0] is attn_output
        # present = outputs[1] when use_cache=True; None otherwise.
        return (attn_output,)


def patch_gpt2_with_dynamic_attention(
    model: nn.Module,
    window_size: int,
    num_sink_tokens: int,
) -> nn.Module:
    """Replace GPT-2 attention layers with ``DynamicAttention`` instances.

    Iterates over each transformer block in ``model.transformer.h`` and
    replaces ``block.attn`` (a ``GPT2Attention``) with a ``DynamicAttention``
    instance whose weights are copied from the original layer.

    GPT-2 uses ``Conv1D`` (weights stored as ``(in_features, out_features)``)
    rather than ``nn.Linear`` (``(out_features, in_features)``).  All weight
    copies therefore apply ``.t().contiguous()`` to transpose the layout.

    Weight mapping
    --------------
    ``GPT2Attention.c_attn.weight``  shape ``(embed_dim, 3 * embed_dim)``
        Split along dim-1 into Q, K, V projections of shape
        ``(embed_dim, embed_dim)`` each, then transpose to
        ``(embed_dim, embed_dim)`` for ``nn.Linear``.
    ``GPT2Attention.c_proj.weight``  shape ``(embed_dim, embed_dim)``
        Transpose to ``(embed_dim, embed_dim)`` for ``nn.Linear``.

    Args:
        model: A loaded HuggingFace GPT-2 ``AutoModelForCausalLM`` instance.
        window_size: Attention window size passed to ``DynamicAttention``.
        num_sink_tokens: Number of sink tokens passed to ``DynamicAttention``.

    Returns:
        The same ``model`` with attention layers replaced in-place.

    Raises:
        AttributeError: If ``model`` does not have a ``transformer.h``
            attribute (i.e. is not a GPT-2 family model).
        ImportError: If ``DynamicAttention`` is not available (memopt not
            installed); callers should guard with ``_MEMOPT_AVAILABLE``.
    """
    embed_dim: int = model.config.n_embd
    num_heads: int = model.config.n_head

    for block_idx, block in enumerate(model.transformer.h):
        original_attn = block.attn

        # Create a DynamicAttention with the same dimensions and window params.
        new_attn = DynamicAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            num_sink_tokens=num_sink_tokens,
        )
        new_attn = new_attn.to(next(original_attn.parameters()).device)

        # Copy weights from GPT2Attention (Conv1D) to DynamicAttention (nn.Linear).
        # Conv1D stores weight as (in_features, out_features); nn.Linear stores
        # (out_features, in_features).  Transpose with .t().contiguous().
        with torch.no_grad():
            # c_attn: combined QKV projection — shape (embed_dim, 3 * embed_dim)
            # Split along the output dimension (dim=1) into Q, K, V chunks.
            c_attn_weight = original_attn.c_attn.weight  # (embed_dim, 3*embed_dim)
            q_weight, k_weight, v_weight = c_attn_weight.split(embed_dim, dim=1)
            new_attn.q_proj.weight.copy_(q_weight.t().contiguous())
            new_attn.k_proj.weight.copy_(k_weight.t().contiguous())
            new_attn.v_proj.weight.copy_(v_weight.t().contiguous())

            # c_proj: output projection — shape (embed_dim, embed_dim)
            new_attn.out_proj.weight.copy_(original_attn.c_proj.weight.t().contiguous())

        # Wrap in adapter so GPT-2 block's calling convention is satisfied.
        block.attn = _GPT2DynamicAttentionAdapter(new_attn)
        logger.debug(
            "Replaced GPT2Attention with _GPT2DynamicAttentionAdapter(DynamicAttention) "
            "at layer %d (embed_dim=%d, num_heads=%d, window_size=%d, num_sink_tokens=%d).",
            block_idx,
            embed_dim,
            num_heads,
            window_size,
            num_sink_tokens,
        )

    logger.info(
        "patch_gpt2_with_dynamic_attention: replaced %d attention layers "
        "(window_size=%d, num_sink_tokens=%d).",
        len(model.transformer.h),
        window_size,
        num_sink_tokens,
    )
    return model


class _HFMemOptWrapper(nn.Module):
    """Thin wrapper around a HuggingFace CausalLM patched with DynamicAttention.

    Identical in interface to ``_HFBaselineWrapper`` but documents the intent
    that the underlying model's attention layers have been replaced with
    ``DynamicAttention`` instances (window-limited KV truncation active).

    Args:
        hf_model: A patched ``AutoModelForCausalLM`` instance.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the patched HF model and return logits.

        ``use_cache=False`` is required because the ``DynamicAttention``
        adapter does not return KV-cache ``present`` tuples.  If
        ``use_cache=True`` (GPT-2 default), ``GPT2Model`` tries to access
        ``outputs[1]`` as the present key-values and raises
        ``IndexError: tuple index out of range``.

        Args:
            input_ids: Token ids of shape ``(B, S)``.

        Returns:
            Logits of shape ``(B, S, vocab_size)``.
        """
        out = self.model(input_ids, use_cache=False)
        return out.logits


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Main entry point: load data, build models, evaluate PPL, save results."""
    args = _parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Flatten potentially nested lists produced by nargs="+" + comma-split.
    # ------------------------------------------------------------------
    raw_ctx = args.context_lengths
    if raw_ctx and isinstance(raw_ctx[0], list):
        flat_ctx: List[int] = [n for sublist in raw_ctx for n in sublist]
    else:
        flat_ctx = list(raw_ctx)

    context_lengths: List[int] = (
        _DRY_RUN_CONTEXT_LENGTHS if args.dry_run else sorted(set(flat_ctx))
    )

    # ------------------------------------------------------------------
    # Log environment info
    # ------------------------------------------------------------------
    logger.info("PyTorch version  : %s", torch.__version__)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "CUDA device      : %s  (%.1f GiB)",
            props.name,
            props.total_memory / (1024**3),
        )
        logger.info("CUDA version     : %s", torch.version.cuda)
    else:
        logger.info("Device           : CPU (VRAM measurements will be 0)")

    logger.info("Config           : model=%s  device=%s  ctx_lens=%s  stride=%d",
                args.model_name, device, context_lengths, args.stride)

    # ------------------------------------------------------------------
    # Load HuggingFace tokenizer (if available and not dry-run)
    # ------------------------------------------------------------------
    tokenizer: Optional[object] = None
    use_hf_models: bool = False

    if _TRANSFORMERS_AVAILABLE and not args.dry_run:
        try:
            logger.info("Loading HuggingFace tokenizer for '%s'...", args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            use_hf_models = True
            logger.info("HuggingFace tokenizer loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load HuggingFace tokenizer for '%s': %s.  "
                "Falling back to tiny self-contained decoder.",
                args.model_name,
                exc,
            )
    elif not _TRANSFORMERS_AVAILABLE:
        logger.warning(
            "transformers not installed — using tiny self-contained decoder.  "
            "Install with: pip install transformers"
        )

    # ------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------
    if args.evaluate_dataset == "both":
        dataset_names: List[str] = ["wikitext2", "c4"]
    else:
        dataset_names = [args.evaluate_dataset]

    # ------------------------------------------------------------------
    # Load and tokenize datasets
    # ------------------------------------------------------------------
    tokens_by_dataset: Dict[str, List[int]] = {}
    for name in dataset_names:
        try:
            if name == "wikitext2":
                tokens_by_dataset[name] = _load_wikitext2_tokens(
                    dry_run=args.dry_run, tokenizer=tokenizer
                )
            else:
                tokens_by_dataset[name] = _load_c4_tokens(
                    dry_run=args.dry_run, tokenizer=tokenizer
                )
            logger.info("Loaded %s: %d tokens.", name, len(tokens_by_dataset[name]))
        except RuntimeError as exc:
            logger.error("Failed to load %s: %s — skipping.", name, exc)

    if not tokens_by_dataset:
        logger.error(
            "No datasets could be loaded.  "
            "Install datasets with: pip install datasets"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build model configs
    # ------------------------------------------------------------------
    scheduler: Optional[MemoryScheduler] = None  # type: ignore[assignment]
    all_results: Dict[str, Dict[str, Dict[str, object]]] = {}

    if use_hf_models:
        # ----------------------------------------------------------------
        # HuggingFace model path: load weights once, wrap per config.
        # ----------------------------------------------------------------
        logger.info("Loading HuggingFace model '%s'...", args.model_name)
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load HuggingFace model '%s': %s.  "
                "Falling back to tiny self-contained decoder.",
                args.model_name,
                exc,
            )
            hf_model = None
            use_hf_models = False

        if use_hf_models and hf_model is not None:
            # ------------------------------------------------------------------
            # Guard: filter context lengths that exceed the model's positional
            # embedding limit.  GPT-2 and similar models raise a CUDA index
            # out-of-bounds error when sequence length >= max_position_embeddings.
            # ------------------------------------------------------------------
            max_pos = getattr(hf_model.config, "max_position_embeddings", None)
            if max_pos is not None:
                valid_ctx = [cl for cl in context_lengths if cl <= max_pos]
                oversized_ctx = [cl for cl in context_lengths if cl > max_pos]

                if oversized_ctx:
                    logger.warning(
                        "WARNING: The following requested context lengths exceed "
                        "max_position_embeddings=%d for model '%s' and will be skipped: %s.  "
                        "Pass --model with a larger model (e.g. 'gpt2-xl', 'gpt-neo-1.3B') "
                        "or use --context-lengths with values <= %d.",
                        max_pos,
                        args.model_name,
                        oversized_ctx,
                        max_pos,
                    )

                if not valid_ctx:
                    raise SystemExit(
                        f"\nERROR: All requested context lengths {context_lengths} exceed "
                        f"max_position_embeddings={max_pos} for model '{args.model_name}'.\n"
                        f"Pass --model with a larger model (e.g. 'gpt2-xl', 'gpt-neo-1.3B', "
                        f"'EleutherAI/gpt-neo-2.7B') or use --context-lengths with values "
                        f"<= {max_pos}."
                    )

                context_lengths = valid_ctx

            # Baseline — wraps hf_model directly (no weight changes).
            baseline_model: nn.Module = _HFBaselineWrapper(hf_model)
            # StreamingLLM — monkey-patches GPT2Attention._attn on a deep copy so
            # the baseline wrapper's underlying model is not affected.
            streaming_model: nn.Module = _HFStreamingLLMWrapper(
                hf_model=copy.deepcopy(hf_model),
                num_sink_tokens=args.num_sink_tokens,
                window_size=args.window_size,
            )
            # MemOpt Full — patch the HF model's attention layers in-place with
            # DynamicAttention (window-limited KV truncation + Ada-KV).  A deep
            # copy of hf_model is made so the baseline and streaming configs are
            # not affected by the weight replacement.
            if _MEMOPT_AVAILABLE:
                logger.info(
                    "MemOpt available — patching HF model copy with "
                    "DynamicAttention (window_size=%d, num_sink_tokens=%d) "
                    "and starting MemoryScheduler.",
                    args.window_size,
                    args.num_sink_tokens,
                )
                hf_model_for_memopt = copy.deepcopy(hf_model)
                try:
                    hf_model_for_memopt = patch_gpt2_with_dynamic_attention(
                        model=hf_model_for_memopt,
                        window_size=args.window_size,
                        num_sink_tokens=args.num_sink_tokens,
                    )
                    memopt_model_final: nn.Module = _HFMemOptWrapper(hf_model_for_memopt)
                except AttributeError as exc:
                    logger.error(
                        "patch_gpt2_with_dynamic_attention failed (%s) — "
                        "memopt_full will fall back to baseline wrapper.",
                        exc,
                    )
                    memopt_model_final = _HFBaselineWrapper(hf_model)
                scheduler = MemoryScheduler(SchedulerConfig(poll_interval_sec=0.05))
            else:
                logger.info(
                    "MemOpt not available — memopt_full wraps HF model "
                    "(scheduler not started; no weight replacement)."
                )
                memopt_model_final = _HFBaselineWrapper(hf_model)

            configs = [
                (_CFG_BASELINE, baseline_model, None),
                (_CFG_STREAMING, streaming_model, None),
                (_CFG_MEMOPT, memopt_model_final, scheduler),
            ]

    if not use_hf_models:
        # ----------------------------------------------------------------
        # Tiny self-contained decoder path (dry-run, no-network, no-HF).
        # ----------------------------------------------------------------
        logger.info(
            "Building tiny self-contained decoder (d_model=%d, n_heads=%d, n_layers=%d).",
            _TINY_D_MODEL,
            _TINY_N_HEADS,
            _TINY_N_LAYERS,
        )

        baseline_model = _build_tiny_baseline(
            d_model=_TINY_D_MODEL,
            n_heads=_TINY_N_HEADS,
            n_layers=_TINY_N_LAYERS,
        )
        streaming_model = _build_tiny_streamingllm(
            d_model=_TINY_D_MODEL,
            n_heads=_TINY_N_HEADS,
            n_layers=_TINY_N_LAYERS,
            num_sink_tokens=args.num_sink_tokens,
            window_size=args.window_size,
        )
        memopt_model_final = _build_tiny_memopt(
            d_model=_TINY_D_MODEL,
            n_heads=_TINY_N_HEADS,
            n_layers=_TINY_N_LAYERS,
        )
        if _MEMOPT_AVAILABLE:
            scheduler = MemoryScheduler(SchedulerConfig(poll_interval_sec=0.05))

        configs = [
            (_CFG_BASELINE, baseline_model, None),
            (_CFG_STREAMING, streaming_model, None),
            (_CFG_MEMOPT, memopt_model_final, scheduler),
        ]

    # ------------------------------------------------------------------
    # Run evaluations
    # ------------------------------------------------------------------
    for config_name, model, sched in configs:
        logger.info("=== Evaluating config: %s ===", config_name)
        config_results = run_config_benchmark(
            config_name=config_name,
            model=model,
            tokens_by_dataset=tokens_by_dataset,
            context_lengths=context_lengths,
            stride=args.stride,
            device=device,
            max_chunks=args.max_chunks,
            scheduler=sched,
        )
        # Merge into per-dataset structure:
        # all_results[dataset][config_name] = {ctx: ppl}
        for ds_name, ctx_ppls in config_results.items():
            if ds_name not in all_results:
                all_results[ds_name] = {}
            all_results[ds_name][config_name] = ctx_ppls

        # Release model memory before the next config.
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save JSON (schema: metadata + results)
    # ------------------------------------------------------------------
    metadata: dict = {
        "model": args.model_name,
        "date": datetime.now(timezone.utc).isoformat(),
        "context_lengths": context_lengths,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "max_chunks": args.max_chunks,
        "device": str(device),
        "num_sink_tokens": args.num_sink_tokens,
        "window_size": args.window_size,
        "dry_run": args.dry_run,
        "pytorch_version": torch.__version__,
        "memopt_available": _MEMOPT_AVAILABLE,
        "hf_models_used": use_hf_models,
    }
    save_json(results=all_results, metadata=metadata, output_path=args.output_file)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print_summary_table(all_results, context_lengths)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if not args.no_plot:
        scripts_dir = Path(__file__).parent
        for ds_name, ds_results in all_results.items():
            plot_path = scripts_dir / f"bench_accuracy_{ds_name}.png"
            plot_dataset_results(
                dataset_name=ds_name,
                context_lengths=context_lengths,
                results=ds_results,
                output_path=plot_path,
            )

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
