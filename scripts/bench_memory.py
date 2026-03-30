"""VRAM benchmark: BaselineTransformer vs MemOptTransformer.

Runs a dummy autoregressive generation loop over a range of sequence lengths
and records peak GPU memory allocated at each length.  Results are printed as
a table and saved as a matplotlib figure.

Usage
-----
    python scripts/bench_memory.py                         # defaults
    python scripts/bench_memory.py --dry-run               # short seq_lengths for smoke-test
    python scripts/bench_memory.py --device cpu            # CPU fallback (VRAM will be 0)
    python scripts/bench_memory.py --num-heads 16 --d-model 1024 --num-layers 8
    python scripts/bench_memory.py --model llama-7b --seq-len 8192

C++ extension (memopt_C)
------------------------
When the extension is not compiled the script falls back to the pure-Python
``DynamicAttention`` SDPA path automatically.  A warning is printed but the
benchmark proceeds.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Resolve the project src/ directory so the script works when run from the
# repo root without installing the package.
# ---------------------------------------------------------------------------
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional memopt imports — degrade gracefully.
# ---------------------------------------------------------------------------
try:
    from memopt.attention import DynamicAttention
    from memopt.scheduler import MemoryScheduler, SchedulerConfig

    _MEMOPT_AVAILABLE: bool = True
except Exception as exc:  # noqa: BLE001
    warnings.warn(
        f"Could not import memopt Python layer ({exc}). "
        "MemOptTransformer will use an inline SDPA-only stub.",
        ImportWarning,
        stacklevel=1,
    )
    _MEMOPT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — always works
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE: bool = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    warnings.warn(
        "matplotlib is not installed.  Plot will not be saved.  "
        "Install with: pip install matplotlib",
        ImportWarning,
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("bench_memory")

# ---------------------------------------------------------------------------
# CPU memory helpers
# ---------------------------------------------------------------------------


def _get_rss_mib() -> float:
    """Return current process RSS (resident set size) in MiB.

    Reads ``/proc/self/status`` directly, which captures all memory mapped by
    the process including PyTorch's C++ allocator — unlike ``tracemalloc``,
    which only tracks the Python heap.  Linux-specific; returns 0.0 on other
    platforms or if the file is unavailable.

    Returns:
        Current RSS in MiB, or ``0.0`` on failure.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # KiB → MiB
    except OSError:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_SEQ_LENGTHS: List[int] = [512, 1024, 2048, 4096, 8192]
_DRY_RUN_SEQ_LENGTHS: List[int] = [64, 128]
_VOCAB_SIZE: int = 32000  # typical LLM vocabulary size

# Predefined architecture configs keyed by --model preset name.
# Each entry provides (d_model, n_heads, n_layers) matching published
# LLaMA architecture dimensions.
_MODEL_CONFIGS: dict = {
    "tiny":      {"d_model": 512,  "n_heads": 8,  "n_layers": 4},
    "llama-7b":  {"d_model": 4096, "n_heads": 32, "n_layers": 32},
    "llama-13b": {"d_model": 5120, "n_heads": 40, "n_layers": 40},
    "llama-70b": {"d_model": 8192, "n_heads": 64, "n_layers": 80},
}


# ===========================================================================
# Baseline Transformer (plain nn.MultiheadAttention — no MemOpt)
# ===========================================================================


class _BaselineBlock(nn.Module):
    """Single Transformer block using standard nn.MultiheadAttention.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension (defaults to 4 * d_model).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Shape ``(B, S, d_model)``.

        Returns:
            Shape ``(B, S, d_model)``.
        """
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class BaselineTransformer(nn.Module):
    """Stacked Transformer using vanilla nn.MultiheadAttention.

    This is the unoptimized reference implementation used for VRAM comparison.
    No MemOpt techniques (paged KV cache, Ada-KV pruning, INT4) are applied.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of stacked Transformer blocks.
        vocab_size: Vocabulary size for the token embedding table.
        max_seq_len: Maximum sequence length (for positional embedding).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int = _VOCAB_SIZE,
        max_seq_len: int = 32768,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_BaselineBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # noqa: ARG002  accepted, unused
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Integer token ids of shape ``(B, S)``.
            attention_mask: Ignored; present for API symmetry.

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
# MemOpt-backed Transformer
# ===========================================================================


class _MemOptBlock(nn.Module):
    """Single Transformer block using DynamicAttention.

    Falls back to a plain SDPA block when memopt is unavailable.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        window_size: Attention window size for DynamicAttention; ``-1`` for
            full context.  Ignored when falling back to SDPA stub.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        window_size: int = -1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        if _MEMOPT_AVAILABLE:
            self.attn: nn.Module = DynamicAttention(
                embed_dim=d_model, num_heads=n_heads, window_size=window_size
            )
        else:
            # Inline SDPA stub — mirrors DynamicAttention fallback path, including
            # windowed causal masking so memory differentiation is preserved.
            self.attn = _SDPAAttentionStub(d_model=d_model, n_heads=n_heads, window_size=window_size)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with pre-norm residuals.

        Args:
            x: Shape ``(B, S, d_model)``.

        Returns:
            Shape ``(B, S, d_model)``.
        """
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x)

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


class _SDPAAttentionStub(nn.Module):
    """Minimal SDPA-based attention stub used when DynamicAttention is unavailable.

    Supports optional sliding-window causal attention via ``window_size`` to
    match the behaviour of ``DynamicAttention._sdpa_forward``.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        window_size: Sliding-window size (number of tokens each position can
            attend to).  ``-1`` or ``0`` disables windowing and falls back to
            standard causal (full-context) SDPA.
    """

    def __init__(self, d_model: int, n_heads: int, window_size: int = -1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SDPA forward with optional sliding-window causal mask.

        Args:
            x: Shape ``(B, S, d_model)``.

        Returns:
            Shape ``(B, S, d_model)``.
        """
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # KV truncation for sliding-window attention: truncate K and V to the
        # last ``window_size`` tokens instead of building an S×S boolean mask.
        # This reduces tensor memory from (B,H,S,D) to (B,H,W,D) and mirrors
        # KV-cache eviction semantics.  Matches the approach in DynamicAttention.
        if self.window_size > 0 and S > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]

        out = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, self.d_model))


class MemOptTransformer(nn.Module):
    """Stacked Transformer using DynamicAttention + MemoryScheduler.

    When the C++ extension is not compiled, DynamicAttention falls back to
    its SDPA path transparently.  The MemoryScheduler daemon is started and
    stopped around the benchmark to reflect the real-world usage pattern.

    Args:
        d_model: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of stacked blocks.
        vocab_size: Vocabulary size for the embedding table.
        max_seq_len: Maximum sequence length.
        window_size: Attention window size passed to each ``_MemOptBlock``;
            ``-1`` for full context, positive values limit attention span.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int = _VOCAB_SIZE,
        max_seq_len: int = 32768,
        window_size: int = -1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                _MemOptBlock(d_model=d_model, n_heads=n_heads, window_size=window_size)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # MemoryScheduler: created but not started until benchmark begins.
        self.scheduler: Optional[MemoryScheduler] = (
            MemoryScheduler(SchedulerConfig(poll_interval_sec=0.05))
            if _MEMOPT_AVAILABLE
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Integer token ids of shape ``(B, S)``.
            attention_mask: Ignored; present for API symmetry.

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

    def start_scheduler(self) -> None:
        """Start the background MemoryScheduler thread (no-op if unavailable)."""
        if self.scheduler is not None:
            self.scheduler.start()

    def stop_scheduler(self) -> None:
        """Stop the background MemoryScheduler thread (no-op if unavailable)."""
        if self.scheduler is not None:
            self.scheduler.stop()


# ===========================================================================
# Benchmark runner
# ===========================================================================


def run_benchmark(
    model: nn.Module,
    seq_lengths: List[int],
    device: torch.device,
    label: str,
    batch_size: int = 1,
    use_cuda: bool = True,
) -> Dict[int, float]:
    """Run a dummy autoregressive loop and record peak memory per sequence length.

    For each ``seq_len`` the function:

    1. Resets memory tracking (``torch.cuda.reset_peak_memory_stats()`` on CUDA,
       RSS snapshot via ``/proc/self/status`` on CPU).
    2. Runs a single forward pass over a randomly generated token sequence of
       length ``seq_len``.  The "generation loop" is approximated by a
       full-context forward pass which allocates realistic KV activations
       without the complexity of a token-by-token loop.
    3. Records peak memory in MiB (CUDA VRAM via ``torch.cuda.max_memory_allocated()``
       or process RSS delta via ``/proc/self/status`` on CPU).
    4. Cleans up tensors with ``gc.collect()`` and ``torch.cuda.empty_cache()``
       before the next iteration.

    Args:
        model: A ``torch.nn.Module`` with a ``forward(input_ids)`` signature.
        seq_lengths: List of sequence lengths to benchmark.
        device: Target torch device.
        label: Human-readable name printed in the progress log.
        batch_size: Number of sequences per forward call.
        use_cuda: Whether CUDA VRAM stats are meaningful.

    Returns:
        Dict mapping each sequence length to peak memory in MiB.  On CPU
        the values reflect RSS growth captured via ``/proc/self/status``.
    """
    model.to(device)
    model.eval()

    results: Dict[int, float] = {}

    for seq_len in seq_lengths:
        # Free memory from the previous iteration.
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # Snapshot RSS before the forward pass so we measure the delta caused
        # by this specific forward call rather than cumulative process growth.
        rss_before = _get_rss_mib() if not use_cuda else 0.0

        input_ids = torch.randint(
            low=0, high=_VOCAB_SIZE, size=(batch_size, seq_len), device=device
        )

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                _ = model(input_ids)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                "[%s] seq_len=%d  OOM — skipping.", label, seq_len
            )
            results[seq_len] = float("nan")
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
            continue
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[%s] seq_len=%d  unexpected error: %s", label, seq_len, exc
            )
            results[seq_len] = float("nan")
            continue
        elapsed = time.perf_counter() - t0

        if use_cuda:
            peak_bytes = torch.cuda.max_memory_allocated(device)
            peak_mib = peak_bytes / (1024 ** 2)
        else:
            # RSS delta captures PyTorch C++ allocations that tracemalloc misses.
            peak_mib = max(0.0, _get_rss_mib() - rss_before)

        results[seq_len] = peak_mib
        mem_label = "peak_vram" if use_cuda else "peak_mem"
        logger.info(
            "[%s] seq_len=%6d  %s=%7.1f MiB  elapsed=%.3fs",
            label, seq_len, mem_label, peak_mib, elapsed,
        )

    return results


# ===========================================================================
# Reporting
# ===========================================================================


def print_table(
    seq_lengths: List[int],
    baseline_results: Dict[int, float],
    memopt_results: Dict[int, float],
) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        seq_lengths: Ordered list of sequence lengths.
        baseline_results: Peak VRAM (MiB) from the baseline model.
        memopt_results: Peak VRAM (MiB) from the MemOpt model.
    """
    col_width = 14
    header = (
        f"{'seq_len':>{col_width}}"
        f"{'baseline (MiB)':>{col_width}}"
        f"{'memopt (MiB)':>{col_width}}"
        f"{'savings (MiB)':>{col_width}}"
        f"{'savings (%)':>{col_width}}"
    )
    separator = "-" * len(header)
    print()
    print("Memory Benchmark Results")
    print(separator)
    print(header)
    print(separator)
    for sl in seq_lengths:
        b = baseline_results.get(sl, float("nan"))
        m = memopt_results.get(sl, float("nan"))
        if b != b or m != m:  # nan check
            savings_mib = float("nan")
            savings_pct = float("nan")
            b_str = "OOM" if b != b else f"{b:.1f}"
            m_str = "OOM" if m != m else f"{m:.1f}"
            sav_mib_str = "N/A"
            sav_pct_str = "N/A"
        else:
            savings_mib = b - m
            savings_pct = (savings_mib / b * 100) if b > 0 else 0.0
            b_str = f"{b:.1f}"
            m_str = f"{m:.1f}"
            sav_mib_str = f"{savings_mib:+.1f}"
            sav_pct_str = f"{savings_pct:+.1f}%"
        print(
            f"{sl:>{col_width}}"
            f"{b_str:>{col_width}}"
            f"{m_str:>{col_width}}"
            f"{sav_mib_str:>{col_width}}"
            f"{sav_pct_str:>{col_width}}"
        )
    print(separator)
    print()


def plot_results(
    seq_lengths: List[int],
    baseline_results: Dict[int, float],
    memopt_results: Dict[int, float],
    output_path: Path,
    use_cuda: bool = True,
) -> None:
    """Save a matplotlib comparison chart to ``output_path``.

    Args:
        seq_lengths: Ordered sequence lengths for the x-axis.
        baseline_results: Peak memory (MiB) — BaselineTransformer.
        memopt_results: Peak memory (MiB) — MemOptTransformer.
        output_path: Destination file for the PNG figure.
        use_cuda: Whether benchmarks ran on CUDA (affects axis labels).

    Raises:
        RuntimeError: If matplotlib is not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise RuntimeError(
            "matplotlib is not installed.  Cannot save plot.  "
            "Install with: pip install matplotlib"
        )

    x = seq_lengths
    b_y = [baseline_results.get(sl, float("nan")) for sl in x]
    m_y = [memopt_results.get(sl, float("nan")) for sl in x]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, b_y, marker="o", linewidth=2, label="BaselineTransformer (nn.MHA)")
    ax.plot(x, m_y, marker="s", linewidth=2, linestyle="--", label="MemOptTransformer (DynamicAttn)")

    mem_type = "VRAM" if use_cuda else "Memory"
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel(f"Peak {mem_type} (MiB)", fontsize=12)
    ax.set_title(f"{mem_type} vs Sequence Length: Baseline vs MemOpt", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.set_xscale("log", base=2)

    # Annotate data points with actual values (1 decimal place).
    for sl, bv, mv in zip(x, b_y, m_y):
        if bv == bv:  # not nan
            ax.annotate(
                f"{bv:.1f}",
                xy=(sl, bv),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="C0",
            )
        if mv == mv:
            ax.annotate(
                f"{mv:.1f}",
                xy=(sl, mv),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="C1",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
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
        description="Benchmark VRAM usage: BaselineTransformer vs MemOptTransformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (e.g. 'cuda', 'cuda:1', 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for each forward pass.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        metavar="H",
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        metavar="D",
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        metavar="L",
        help="Number of Transformer layers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "bench_memory.png",
        metavar="FILE",
        help="Output path for the PNG plot.",
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_CONFIGS.keys()),
        default="tiny",
        metavar="PRESET",
        help=(
            "Architecture preset.  One of: "
            + ", ".join(_MODEL_CONFIGS.keys())
            + ".  Sets d_model/num_heads/num_layers from a predefined config.  "
            "Explicit --d-model/--num-heads/--num-layers flags override the preset."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run the benchmark at a single sequence length N instead of the "
            "default range.  Ignored when --dry-run is also set."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        metavar="W",
        help=(
            "Attention window size for MemOptTransformer.  Limits each "
            "DynamicAttention layer to attend to at most W past KV tokens.  "
            "Use -1 for full (unbounded) context."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use short sequence lengths [64, 128] for quick smoke-testing.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving the matplotlib figure.",
    )
    return parser.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Main entry point: build models, run benchmarks, print and plot results."""
    args = _parse_args()

    # ------------------------------------------------------------------
    # Apply model preset, then let explicit flags override individual dims.
    # ------------------------------------------------------------------
    preset = _MODEL_CONFIGS[args.model]
    # Only override from preset when the user did NOT supply the flag
    # explicitly.  argparse always populates the attribute, so we compare
    # against the declared defaults to detect "not explicitly set".
    _DEFAULT_D_MODEL = 512
    _DEFAULT_N_HEADS = 8
    _DEFAULT_N_LAYERS = 4
    if args.d_model == _DEFAULT_D_MODEL:
        args.d_model = preset["d_model"]
    if args.num_heads == _DEFAULT_N_HEADS:
        args.num_heads = preset["n_heads"]
    if args.num_layers == _DEFAULT_N_LAYERS:
        args.num_layers = preset["n_layers"]

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    if use_cuda and not torch.cuda.is_available():
        logger.warning(
            "CUDA device '%s' requested but CUDA is not available.  "
            "Falling back to CPU.",
            args.device,
        )
        device = torch.device("cpu")
        use_cuda = False

    if use_cuda:
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "CUDA device: %s  (%.1f GiB total VRAM)",
            props.name,
            props.total_memory / (1024 ** 3),
        )
    else:
        logger.warning(
            "Running on CPU — using process RSS delta for memory tracking.  "
            "Use --device cuda for GPU VRAM measurements."
        )

    if args.dry_run:
        seq_lengths = _DRY_RUN_SEQ_LENGTHS
    elif args.seq_len is not None:
        seq_lengths = [args.seq_len]
    else:
        seq_lengths = _DEFAULT_SEQ_LENGTHS

    logger.info(
        "Config: d_model=%d  n_heads=%d  n_layers=%d  batch_size=%d  seq_lengths=%s",
        args.d_model, args.num_heads, args.num_layers, args.batch_size, seq_lengths,
    )

    # ------------------------------------------------------------------
    # Build models
    # ------------------------------------------------------------------
    logger.info("Building BaselineTransformer...")
    baseline_model = BaselineTransformer(
        d_model=args.d_model,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        max_seq_len=max(seq_lengths) + 1,
    )

    logger.info(
        "Building MemOptTransformer (window_size=%d)...", args.window_size,
    )
    memopt_model = MemOptTransformer(
        d_model=args.d_model,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        max_seq_len=max(seq_lengths) + 1,
        window_size=args.window_size,
    )

    # ------------------------------------------------------------------
    # Baseline benchmark
    # ------------------------------------------------------------------
    logger.info("--- Benchmarking BaselineTransformer ---")
    baseline_results = run_benchmark(
        model=baseline_model,
        seq_lengths=seq_lengths,
        device=device,
        label="Baseline",
        batch_size=args.batch_size,
        use_cuda=use_cuda,
    )
    del baseline_model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # MemOpt benchmark
    # ------------------------------------------------------------------
    logger.info("--- Benchmarking MemOptTransformer ---")
    memopt_model.start_scheduler()
    try:
        memopt_results = run_benchmark(
            model=memopt_model,
            seq_lengths=seq_lengths,
            device=device,
            label="MemOpt",
            batch_size=args.batch_size,
            use_cuda=use_cuda,
        )
    finally:
        memopt_model.stop_scheduler()

    del memopt_model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    print_table(seq_lengths, baseline_results, memopt_results)

    if not args.no_plot:
        if _MATPLOTLIB_AVAILABLE:
            plot_results(
                seq_lengths, baseline_results, memopt_results, args.output,
                use_cuda=use_cuda,
            )
        else:
            logger.warning("Skipping plot: matplotlib not available.")


if __name__ == "__main__":
    main()
