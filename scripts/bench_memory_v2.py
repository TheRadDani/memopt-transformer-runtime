"""Fine-grained VRAM breakdown benchmark: BaselineTransformer vs MemOptTransformer.

Location: scripts/bench_memory_v2.py

Separately tracks four memory categories per (model, seq_len):
  1. weights_mib     — model parameter bytes allocated after model.to(device)
  2. kv_cache_mib    — analytical estimate: 2*B*L*H*S*head_dim*dtype_bytes
                       NOTE: this is an estimate; actual KV allocation is
                       interleaved with activations at the PyTorch allocator level.
  3. activation_mib  — peak_active - weights_mib - kv_cache_mib (clamped ≥ 0)
  4. active_mib      — torch.cuda.max_memory_allocated() / 1024**2
  5. reserved_mib    — torch.cuda.max_memory_reserved() / 1024**2

Outputs: CSV (csv.DictWriter, stdlib only) + optional two-subplot matplotlib figure.
Results are written to results/memory_v2_{timestamp}.csv by default.
Predecessor: scripts/bench_memory.py (single active_mib metric).

Usage:
    python scripts/bench_memory_v2.py --dry-run --no-plot --device cpu
    python scripts/bench_memory_v2.py --model medium --seq-lengths 512 1024 2048
    python scripts/bench_memory_v2.py --output-dir /tmp/results
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn

try:
    from memopt.attention import DynamicAttention
    from memopt.scheduler import MemoryScheduler, SchedulerConfig
    _MEMOPT_AVAILABLE: bool = True
except Exception as exc:  # noqa: BLE001
    warnings.warn(f"Could not import memopt ({exc}). MemOpt uses SDPA stub.", ImportWarning, stacklevel=1)
    _MEMOPT_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE: bool = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("bench_memory_v2")

_VOCAB_SIZE: int = 32_000
_DEFAULT_SEQ_LENGTHS: List[int] = [512, 1024, 2048, 4096, 8192]
_DRY_RUN_SEQ_LENGTHS: List[int] = [64, 128]
_BYTES_PER_MIB: float = 1024.0 ** 2
_MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":     {"d_model": 256,  "n_heads": 4,  "n_layers": 4},
    "medium":   {"d_model": 512,  "n_heads": 8,  "n_layers": 6},
    "llama-7b": {"d_model": 4096, "n_heads": 32, "n_layers": 32},
}
_CSV_FIELDNAMES: List[str] = [
    "model",
    "seq_len",
    "active_mib",
    "reserved_mib",
    "weights_mib",
    "kv_cache_mib",
    "activation_mib",
]

# Internal-only extended fieldnames (written in full result dicts but not to CSV).
_INTERNAL_FIELDNAMES: List[str] = [
    "model", "seq_len", "batch_size",
    "weights_mib", "kv_cache_mib", "activation_mib",
    "active_mib", "reserved_mib", "reserved_minus_active_mib",
]


# ===========================================================================
# Model architectures
# ===========================================================================


class _BaselineBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        nx = self.norm1(x)
        attn_out, _ = self.attn(nx, nx, nx, need_weights=False)
        x = res + attn_out
        res = x
        return res + self.ff(self.norm2(x))


class BaselineTransformer(nn.Module):
    """Stacked Transformer using vanilla nn.MultiheadAttention (no MemOpt)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 vocab_size: int = _VOCAB_SIZE, max_seq_len: int = 32_768) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([_BaselineBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embedding(torch.arange(S, device=input_ids.device).unsqueeze(0))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


class _SDPAStub(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads, self.head_dim, self.d_model = n_heads, d_model // n_heads, d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        def proj_reshape(p): return p(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        out = nn.functional.scaled_dot_product_attention(proj_reshape(self.q_proj), proj_reshape(self.k_proj), proj_reshape(self.v_proj))
        return self.out_proj(out.transpose(1, 2).reshape(B, S, self.d_model))


class _MemOptBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn: nn.Module = (DynamicAttention(embed_dim=d_model, num_heads=n_heads)
                                if _MEMOPT_AVAILABLE else _SDPAStub(d_model, n_heads))
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = res + self.attn(self.norm1(x))
        res = x
        return res + self.ff(self.norm2(x))


class MemOptTransformer(nn.Module):
    """Stacked Transformer using DynamicAttention + MemoryScheduler."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 vocab_size: int = _VOCAB_SIZE, max_seq_len: int = 32_768) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([_MemOptBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.scheduler: Optional[MemoryScheduler] = (
            MemoryScheduler(SchedulerConfig(poll_interval_sec=0.05)) if _MEMOPT_AVAILABLE else None
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embedding(input_ids) + self.pos_embedding(torch.arange(S, device=input_ids.device).unsqueeze(0))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def start_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.start()

    def stop_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.stop()


# ===========================================================================
# Memory helpers
# ===========================================================================


def estimate_kv_cache_bytes(n_layers: int, n_heads: int, head_dim: int,
                             seq_len: int, batch_size: int,
                             dtype: torch.dtype = torch.float32) -> int:
    """KV cache = 2 (K+V) * batch * layers * heads * seq_len * head_dim * bytes_per_elem."""
    bpe = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    return 2 * batch_size * n_layers * n_heads * seq_len * head_dim * bpe


def _to_mib(byte_count: int) -> float:
    return round(byte_count / _BYTES_PER_MIB, 3)


def _cuda_clear(device: torch.device) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _nan_row(config_name: str, seq_len: int, batch_size: int) -> Dict:
    """Build a row dict representing an OOM or failed measurement.

    Numeric fields are set to ``float('nan')`` (printed as ``"OOM"`` in the
    console table and written as empty string ``""`` to CSV to avoid downstream
    ``"nan"`` string misparses).

    Args:
        config_name: Model label (``"baseline"`` or ``"memopt"``).
        seq_len: Sequence length that caused the failure.
        batch_size: Batch size used.

    Returns:
        Result dict with NaN numeric fields.
    """
    nan = float("nan")
    return {
        "model": config_name,
        "config": config_name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "weights_mib": nan,
        "kv_cache_mib": nan,
        "activation_mib": nan,
        "active_mib": nan,
        "reserved_mib": nan,
        "reserved_minus_active_mib": nan,
    }


# ===========================================================================
# Benchmark runner
# ===========================================================================


def run_breakdown_benchmark(
    model: nn.Module, config_name: str, seq_lengths: List[int],
    batch_size: int, device: torch.device, use_cuda: bool,
    n_layers: int, n_heads: int, head_dim: int,
    model_dtype: torch.dtype = torch.float32,
) -> List[Dict]:
    """Run per-seq-len fine-grained memory breakdown.

    Memory isolation rationale:
    - Weight bytes: allocator delta after model.to(device) with empty cache → only parameters land.
    - Activation bytes: peak_allocated − mem_before_forward. mem_before_forward captures the
      weight+buffer baseline; the delta isolates only forward-pass intermediates (Q/K/V tensors,
      attention scores, FFN activations). The KV cache is NOT separately allocated by the fallback
      SDPA path — it is analytically estimated to avoid contaminating the activation signal.
    - KV cache: analytical estimate (theoretical full-sequence FP32 cost) kept independent of
      measured allocations, which interleave K/V temporaries with other activations and cannot
      be cleanly separated at the PyTorch allocator level without custom hooks.
    """
    model.eval()
    rows: List[Dict] = []

    # Pre-flight VRAM check: estimate weight bytes on CPU before touching the GPU.
    # model.to(device) does NOT raise OutOfMemoryError cleanly when the model is
    # too large — it causes the driver to thrash/swap and can freeze the system.
    # Use the target dtype byte-width so fp16 models get the correct estimate.
    if use_cuda:
        dtype_bytes = torch.finfo(model_dtype).bits // 8
        cpu_weight_bytes = sum(p.numel() for p in model.parameters()) * dtype_bytes
        free_vram, total_vram = torch.cuda.mem_get_info(device)
        if cpu_weight_bytes > free_vram * 0.98:
            logger.warning(
                "[%s] SKIPPED — model weights (~%.1f GiB in %s) exceed 98%% of free VRAM "
                "(%.1f GiB free / %.1f GiB total). Cannot safely load onto GPU.",
                config_name,
                cpu_weight_bytes / (1024 ** 3),
                model_dtype,
                free_vram / (1024 ** 3),
                total_vram / (1024 ** 3),
            )
            return [_nan_row(config_name, sl, batch_size) for sl in seq_lengths]

    # Measure weight footprint once — constant across seq_lengths.
    gc.collect()
    if use_cuda:
        _cuda_clear(device)
        mem_before = torch.cuda.memory_allocated(device)
        try:
            model.to(dtype=model_dtype, device=device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("[%s] OOM while loading model weights — skipping.", config_name)
            gc.collect()
            torch.cuda.empty_cache()
            return [_nan_row(config_name, sl, batch_size) for sl in seq_lengths]
        torch.cuda.synchronize(device)
        weight_bytes = max(0, torch.cuda.memory_allocated(device) - mem_before)
    else:
        model.to(device)
        weight_bytes = 0
    weight_mib = _to_mib(weight_bytes)
    logger.info("[%s] weight footprint: %.3f MiB", config_name, weight_mib)

    for seq_len in seq_lengths:
        gc.collect()
        if use_cuda:
            _cuda_clear(device)

        kv_mib = _to_mib(estimate_kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, batch_size, dtype=model_dtype))
        input_ids = torch.randint(0, _VOCAB_SIZE, (batch_size, seq_len), device=device)

        try:
            if use_cuda:
                _cuda_clear(device)
                mem_before_fwd = torch.cuda.memory_allocated(device)
            else:
                mem_before_fwd = 0

            with torch.no_grad():
                _ = model(input_ids)

            if use_cuda:
                torch.cuda.synchronize(device)
                peak_alloc = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
            else:
                peak_alloc = peak_reserved = 0

        except torch.cuda.OutOfMemoryError:
            logger.warning("[%s] seq_len=%d OOM — writing NaN row.", config_name, seq_len)
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
            rows.append(_nan_row(config_name, seq_len, batch_size))
            continue
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] seq_len=%d error: %s", config_name, seq_len, exc)
            rows.append(_nan_row(config_name, seq_len, batch_size))
            continue
        finally:
            del input_ids

        # activation_mib: active minus the static weight footprint and analytical
        # KV cache estimate. Clamped to ≥0 to handle estimation error where the
        # analytical KV size exceeds the actual allocator delta.
        active_mib = _to_mib(peak_alloc)
        reserved_mib_val = _to_mib(peak_reserved)
        activation_mib = _to_mib(max(0, peak_alloc - mem_before_fwd))

        logger.info(
            "[%s] seq_len=%6d  weights=%.1f  kv_est=%.1f  activations=%.1f  "
            "active=%.1f  reserved=%.1f MiB",
            config_name, seq_len, weight_mib, kv_mib, activation_mib,
            active_mib, reserved_mib_val,
        )
        rows.append({
            "model": config_name,
            "config": config_name,      # legacy alias used by plot_breakdown
            "seq_len": seq_len,
            "batch_size": batch_size,
            "weights_mib": weight_mib,
            "kv_cache_mib": kv_mib,
            "activation_mib": activation_mib,
            "active_mib": active_mib,
            "reserved_mib": reserved_mib_val,
            "reserved_minus_active_mib": round(reserved_mib_val - active_mib, 3),
            # Legacy aliases kept so plot_breakdown can still read the old keys:
            "weight_mb": weight_mib,
            "kv_cache_mb_estimated": kv_mib,
            "activation_mb": activation_mib,
            "peak_allocated_mb": active_mib,
            "peak_reserved_mb": reserved_mib_val,
            "reserved_minus_allocated_mb": round(reserved_mib_val - active_mib, 3),
        })

        # Move to CPU between seq_lens to free VRAM before next measurement.
        model.cpu()
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        model.to(device)

    model.cpu()
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return rows


# ===========================================================================
# CSV and plot
# ===========================================================================


def write_csv(rows: List[Dict], output_path: Path) -> None:
    """Write benchmark rows to CSV using csv.DictWriter (stdlib only).

    OOM / NaN rows are written with empty string ``""`` for numeric fields so
    that downstream tools do not receive literal ``"nan"`` strings.

    Args:
        rows: List of measurement result dicts.
        output_path: Destination ``.csv`` file path.
    """
    import math

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Map legacy "config" key to "model" when needed.
            if "config" in row and "model" not in row:
                row = dict(row, model=row["config"])
            # Write numeric NaN fields as empty string.
            cleaned: Dict = {"model": row.get("model", row.get("config", ""))}
            cleaned["seq_len"] = row.get("seq_len", "")
            for field in ("active_mib", "reserved_mib", "weights_mib",
                          "kv_cache_mib", "activation_mib"):
                val = row.get(field, float("nan"))
                try:
                    fval = float(val)  # type: ignore[arg-type]
                    cleaned[field] = "" if math.isnan(fval) else fval
                except (TypeError, ValueError):
                    cleaned[field] = ""
            writer.writerow(cleaned)
    logger.info("CSV written to %s", output_path)


def _safe_float(val: object) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return float("nan")


def _is_plottable_row(row: Dict) -> bool:
    """Return True if a row contains real measurement data (not a skipped/NaN sentinel).

    Skipped rows (from _nan_row or OOM paths) lack the legacy alias keys added
    only in the successful measurement path of run_breakdown_benchmark.
    """
    import math
    val = row.get("peak_allocated_mb")
    if val is None:
        return False
    try:
        return not math.isnan(float(val))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def plot_breakdown(rows: List[Dict], output_path: Path) -> None:
    """Two-subplot figure: stacked bar breakdown (left) + peak line chart (right).

    Rows that represent skipped or OOM measurements (sentinel rows from _nan_row)
    are silently excluded from both subplots.  If a config has no plottable rows
    after filtering it is omitted entirely — no crash, no empty bars.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib not installed. pip install matplotlib")

    # Filter to plottable rows only; skip sentinel/NaN rows that lack legacy keys.
    plottable_rows = [r for r in rows if _is_plottable_row(r)]
    if not plottable_rows:
        logger.warning("plot_breakdown: no plottable rows after filtering skipped/OOM entries — skipping figure.")
        return

    configs: Dict[str, List[Dict]] = {}
    for row in plottable_rows:
        configs.setdefault(row["config"], []).append(row)
    for v in configs.values():
        v.sort(key=lambda r: r["seq_len"])

    config_names = list(configs.keys())
    all_seq_lens = sorted({r["seq_len"] for r in plottable_rows})
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Stacked bar chart ----
    bar_w = 0.35
    stack_fields = ["weight_mb", "kv_cache_mb_estimated", "activation_mb"]
    stack_labels = ["Weights", "KV Cache (est.)", "Activations"]
    stack_colors = ["#4C72B0", "#DD8452", "#55A868"]
    alphas = [0.90, 0.55]
    x_base = list(range(len(all_seq_lens)))
    seq_to_xi = {sl: i for i, sl in enumerate(all_seq_lens)}

    for ci, cfg_name in enumerate(config_names):
        cfg_rows = configs[cfg_name]
        offset = (ci - (len(config_names) - 1) / 2.0) * bar_w
        bottoms = [0.0] * len(cfg_rows)
        for field, label, color in zip(stack_fields, stack_labels, stack_colors):
            heights = [_safe_float(r.get(field, "nan")) for r in cfg_rows]
            xs = [seq_to_xi[r["seq_len"]] + offset for r in cfg_rows]
            ax_bar.bar(xs, heights, bar_w * 0.92, bottom=bottoms, color=color,
                       alpha=alphas[ci % 2], edgecolor="white", linewidth=0.5,
                       label=label if ci == 0 else None)
            bottoms = [b + max(0, h) for b, h in zip(bottoms, heights)]

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels([str(sl) for sl in all_seq_lens], rotation=30, ha="right")
    ax_bar.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax_bar.set_ylabel("Memory (MiB)", fontsize=11)
    ax_bar.set_title("Memory Breakdown by Category", fontsize=12)
    ax_bar.grid(axis="y", alpha=0.35)
    ax_bar.legend(fontsize=9, loc="upper left")

    # ---- Line chart: peak allocated + reserved ----
    fmt_map = {"baseline": "-o", "memopt": "--s"}
    color_map = {"baseline": "#4C72B0", "memopt": "#DD8452"}
    for cfg_name, cfg_rows in configs.items():
        xs = [r["seq_len"] for r in cfg_rows]
        alloc = [_safe_float(r.get("peak_allocated_mb", "nan")) for r in cfg_rows]
        reserved = [_safe_float(r.get("peak_reserved_mb", "nan")) for r in cfg_rows]
        color = color_map.get(cfg_name)
        fmt = fmt_map.get(cfg_name, "-o")
        ax_line.plot(xs, alloc, fmt, color=color, linewidth=2, label=f"{cfg_name} allocated")
        ax_line.plot(xs, reserved, fmt, color=color, linewidth=1.5, linestyle=":",
                     alpha=0.6, label=f"{cfg_name} reserved")

    ax_line.set_xlabel("Sequence Length (tokens)", fontsize=11)
    ax_line.set_ylabel("Peak Memory (MiB)", fontsize=11)
    ax_line.set_title("Peak Allocated vs Reserved", fontsize=12)
    ax_line.legend(fontsize=9)
    ax_line.grid(alpha=0.35)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Figure saved to %s", output_path)


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-grained VRAM breakdown benchmark: BaselineTransformer vs MemOptTransformer. "
            "Outputs structured CSV and optional matplotlib figure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (e.g. 'cuda', 'cuda:1', 'cpu').")
    parser.add_argument("--seq-lengths", "--seq-len", type=int, nargs="+",
                        default=_DEFAULT_SEQ_LENGTHS, metavar="N",
                        dest="seq_lengths",
                        help="Space-separated sequence length(s) to benchmark.")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N",
                        help="Batch size for each forward pass.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "CSV output path.  When omitted a timestamped file is created in "
            "--output-dir (e.g. results/memory_v2_20260101_120000.csv)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        metavar="DIR",
        dest="output_dir",
        help="Directory for timestamped CSV output (default: <repo>/results/).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip saving the matplotlib figure.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use short seq_lengths [64, 128] for quick smoke-testing.")
    parser.add_argument("--model", choices=list(_MODEL_PRESETS.keys()), default="tiny",
                        help=("Model preset. tiny=d_model=256,n_heads=4,n_layers=4; "
                              "medium=d_model=512,n_heads=8,n_layers=6."))
    parser.add_argument("--fp32", action="store_true",
                        help="Keep model in float32 on CUDA (default: auto-cast to float16 to halve VRAM).")
    return parser.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    """Orchestrate benchmark: build models, run measurements, write CSV, plot."""
    args = _parse_args()
    seq_lengths = _DRY_RUN_SEQ_LENGTHS if args.dry_run else args.seq_lengths

    # Resolve output path (timestamp-based when --output not explicitly set).
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(args.output_dir) / f"memory_v2_{timestamp}.csv"

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        device = torch.device("cpu")
        use_cuda = False

    if use_cuda:
        props = torch.cuda.get_device_properties(device)
        logger.info("CUDA device: %s  (%.1f GiB total VRAM)", props.name,
                    props.total_memory / (1024 ** 3))
    else:
        logger.warning("Running on CPU — all VRAM values will be 0.0 MiB.")

    preset = _MODEL_PRESETS[args.model]
    d_model, n_heads, n_layers = preset["d_model"], preset["n_heads"], preset["n_layers"]
    head_dim = d_model // n_heads
    max_seq_len = max(seq_lengths) + 1

    logger.info("Config: model=%s  d_model=%d  n_heads=%d  n_layers=%d  head_dim=%d  "
                "batch_size=%d  seq_lengths=%s",
                args.model, d_model, n_heads, n_layers, head_dim, args.batch_size, seq_lengths)

    # Use fp16 on CUDA by default to halve weight memory; opt out with --fp32.
    model_dtype = torch.float32 if (args.fp32 or not use_cuda) else torch.float16
    if use_cuda and model_dtype == torch.float16:
        logger.info("Using float16 on CUDA (pass --fp32 to disable).")

    all_rows: List[Dict] = []
    bench_kwargs = dict(seq_lengths=seq_lengths, batch_size=args.batch_size,
                        device=device, use_cuda=use_cuda,
                        n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
                        model_dtype=model_dtype)

    # Baseline.
    logger.info("=== Baseline (nn.MultiheadAttention) ===")
    baseline = BaselineTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                                   max_seq_len=max_seq_len)
    all_rows.extend(run_breakdown_benchmark(baseline, "baseline", **bench_kwargs))
    del baseline
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # MemOpt.
    logger.info("=== MemOpt (DynamicAttention + MemoryScheduler) ===")
    memopt = MemOptTransformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                               max_seq_len=max_seq_len)
    memopt.start_scheduler()
    try:
        all_rows.extend(run_breakdown_benchmark(memopt, "memopt", **bench_kwargs))
    finally:
        memopt.stop_scheduler()
    del memopt
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    write_csv(all_rows, args.output)
    print(f"Results saved to: {args.output}")

    if not args.no_plot:
        if _MATPLOTLIB_AVAILABLE:
            has_plottable = any(_is_plottable_row(r) for r in all_rows)
            if has_plottable:
                plot_breakdown(all_rows, args.output.with_suffix(".png"))
            else:
                print("No data to plot — all configurations were skipped.")
        else:
            logger.warning("matplotlib not available — skipping plot.")

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
