"""bench_latency_v2.py — Fine-grained generation latency benchmark for MemOpt.

Location: scripts/bench_latency_v2.py
Summary: Separates TTFT (Time To First Token / prefill) from TPOT (Time Per Output
    Token / decode) across variable batch sizes.  Compares Baseline
    (nn.MultiheadAttention) vs MemOpt (DynamicAttention + MemoryScheduler) and
    writes a structured CSV + optional matplotlib figure.

Used by:
    Researchers comparing KV-cache and attention strategies.
    Related to: scripts/bench_latency.py (predecessor — whole-loop timer),
    src/memopt/attention.py (DynamicAttention), src/memopt/scheduler.py
    (MemoryScheduler).

Usage::

    # Quick smoke test — batch=[1,2], tokens=10
    python scripts/bench_latency_v2.py --dry-run

    # Minimal CPU smoke test with explicit args
    python scripts/bench_latency_v2.py \\
        --device cpu --batch-sizes 1 4 --prompt-len 16 --gen-len 8 \\
        --warmup 1 --no-plot

    # Full GPU run
    python scripts/bench_latency_v2.py --device cuda

Performance notes
-----------------
MemOpt advantage: DynamicAttention uses a sliding-window KV truncation that
reduces the attention matrix from (S, S) to (S, window_size).  This benefit
only activates when window_size > 0 and seq_len > window_size.  The default
MEMOPT_WINDOW_SIZE of 256 means savings kick in once the sequence exceeds 256
tokens (prompt_len + generated tokens).

Without a positive window_size the fallback _sdpa_forward path adds overhead
(extra reshape/transpose/window-check ops) without any KV savings, causing
MemOpt to run slower than baseline — especially at large batch sizes where
tensor sizes amplify the overhead difference.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure src/ is importable when running from the repo root
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from memopt import DynamicAttention, MemoryScheduler  # noqa: E402

# ---------------------------------------------------------------------------
# Architecture presets
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 256
MAX_SEQ_LEN: int = 2048
SEED: int = 42

# Sliding-window size for DynamicAttention in the MemOpt model.
# KV truncation in _sdpa_forward activates when seq_len > MEMOPT_WINDOW_SIZE,
# shrinking the attention matrix from (S, S) to (S, W).  At prompt_len=128 and
# gen_len=200 (total up to 328 tokens) a window of 256 ensures savings kick in
# partway through generation.  Tune down for more aggressive savings at the
# cost of slightly lower quality approximation.
MEMOPT_WINDOW_SIZE: int = 256

_PRESETS: Dict[str, Dict[str, int]] = {
    "tiny":   {"d_model": 256, "n_heads": 4, "n_layers": 4},
    "medium": {"d_model": 512, "n_heads": 8, "n_layers": 6},
}

# ---------------------------------------------------------------------------
# Model architecture — mirrors _TinyTransformerDecoder in bench_latency.py
# ---------------------------------------------------------------------------


class _TransformerLayer(nn.Module):
    """Single decoder layer: LayerNorm + attention + LayerNorm + FFN.

    Args:
        attn_module: Attention module (MultiheadAttention or DynamicAttention).
        d_model: Hidden dimension.
        d_ff: Feed-forward intermediate dimension.
        use_dynamic_attention: Selects between DynamicAttention single-arg
            signature and nn.MultiheadAttention three-arg signature.
    """

    def __init__(
        self,
        attn_module: nn.Module,
        d_model: int,
        d_ff: int,
        use_dynamic_attention: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attn_module
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self._use_dynamic_attention = use_dynamic_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, S, D)``.

        Returns:
            Output tensor of shape ``(B, S, D)``.
        """
        residual = x
        normed = self.norm1(x)
        if self._use_dynamic_attention:
            attn_out = self.attn(normed)
        else:
            attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = residual + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class _TinyTransformerDecoder(nn.Module):
    """Minimal autoregressive Transformer decoder.

    Architecture: Embedding → N × _TransformerLayer → LayerNorm → LM head.
    forward(input_ids) → logits of shape (B, S, vocab_size).

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        d_ff: Feed-forward intermediate size.
        max_seq_len: Maximum positional embedding length.
        use_dynamic_attention: Use DynamicAttention when True.
        window_size: Sliding-window size passed to DynamicAttention.  ``-1``
            disables windowing (full context).  Ignored when
            ``use_dynamic_attention=False``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        use_dynamic_attention: bool,
        window_size: int = -1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Pre-allocate the full position index buffer once so that forward()
        # can slice into it instead of calling torch.arange() on every step.
        # Registered as a buffer so it moves with the model when .to(device)
        # is called.  Shape: (1, max_seq_len) — sliced to (1, S) at runtime.
        self.register_buffer(
            "_pos_ids",
            torch.arange(max_seq_len).unsqueeze(0),  # (1, max_seq_len)
            persistent=False,
        )

        layers = []
        for _ in range(n_layers):
            if use_dynamic_attention:
                # window_size activates KV truncation in _sdpa_forward when
                # seq_len > window_size, shrinking the attention tensor from
                # (B, H, S, S) to (B, H, S, W).  Pass -1 to disable (full context).
                attn: nn.Module = DynamicAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    window_size=window_size,
                )
            else:
                attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    batch_first=True,
                )
            layers.append(
                _TransformerLayer(
                    attn_module=attn,
                    d_model=d_model,
                    d_ff=d_ff,
                    use_dynamic_attention=use_dynamic_attention,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for every position.

        Args:
            input_ids: Token indices of shape ``(B, S)``.

        Returns:
            Logits of shape ``(B, S, vocab_size)``.
        """
        B, S = input_ids.shape
        # Slice the pre-allocated buffer instead of allocating a new arange tensor
        # on every forward call.  expand() is zero-copy (returns a view).
        positions = self._pos_ids[:, :S].expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    """Synchronize CUDA stream if on GPU, no-op on CPU.

    Args:
        device: The current compute device.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def _measure_ttft(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """Time a single full prefill forward pass (TTFT).

    Args:
        model: Decoder model in eval mode on ``device``.
        prompt_ids: Token IDs of shape ``(B, prompt_len)``.
        device: Compute device.

    Returns:
        TTFT in milliseconds.
    """
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(prompt_ids)
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0


def _measure_decode_loop(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    gen_len: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Run the token-by-token decode loop and capture per-step latencies.

    Reasoning chain: The decode loop appends one greedy token per step.
    We time each step individually with synchronize() barriers so CUDA ops
    don't overlap across steps.  We exclude the first step from TPOT because
    it overlaps with prefill warm-up effects (cache cold-start), consistent
    with standard inference profiling practice.  Throughput uses the *total*
    decode wall time (all steps included) so it reflects real-world token rate.

    Args:
        model: Decoder model in eval mode on ``device``.
        prompt_ids: Token IDs of shape ``(B, prompt_len)``.
        gen_len: Number of tokens to generate.
        device: Compute device.

    Returns:
        Tuple of (tpot_ms, throughput_tokens_per_sec, total_decode_wall_ms).
        tpot_ms is the mean per-step latency excluding the first step.
    """
    ids = prompt_ids.clone()
    step_latencies_sec: List[float] = []

    with torch.no_grad():
        for _ in range(gen_len):
            _sync(device)
            t0 = time.perf_counter()
            logits = model(ids)                                     # (B, S, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            ids = torch.cat([ids, next_token], dim=1)
            _sync(device)
            step_latencies_sec.append(time.perf_counter() - t0)

    total_decode_wall_ms = sum(step_latencies_sec) * 1000.0
    # Exclude first step from TPOT to avoid prefill overlap bias
    tpot_steps = step_latencies_sec[1:] if len(step_latencies_sec) > 1 else step_latencies_sec
    tpot_ms = (sum(tpot_steps) / len(tpot_steps)) * 1000.0

    batch_size = ids.shape[0]
    total_decode_sec = sum(step_latencies_sec)
    throughput = (batch_size * gen_len) / total_decode_sec if total_decode_sec > 0 else 0.0

    return tpot_ms, throughput, total_decode_wall_ms


# ---------------------------------------------------------------------------
# Per-configuration benchmark runner
# ---------------------------------------------------------------------------


def _build_model(
    preset: Dict[str, int],
    use_dynamic_attention: bool,
    max_seq_len: int = MAX_SEQ_LEN,
    window_size: int = -1,
) -> _TinyTransformerDecoder:
    """Construct a decoder model for the given preset.

    Args:
        preset: Architecture dict with keys d_model, n_heads, n_layers.
        use_dynamic_attention: Use DynamicAttention when True.
        max_seq_len: Maximum sequence length for positional embeddings.
        window_size: Sliding-window size for DynamicAttention.  ``-1``
            disables windowing.  Ignored when ``use_dynamic_attention=False``.

    Returns:
        Freshly-initialised decoder model.
    """
    d_model = preset["d_model"]
    return _TinyTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_heads=preset["n_heads"],
        n_layers=preset["n_layers"],
        d_ff=d_model * 4,
        max_seq_len=max_seq_len,
        use_dynamic_attention=use_dynamic_attention,
        window_size=window_size,
    )


def _run_config(
    config_name: str,
    model: nn.Module,
    batch_sizes: List[int],
    prompt_len: int,
    gen_len: int,
    device: torch.device,
    scheduler: Optional[MemoryScheduler],
) -> List[Dict[str, object]]:
    """Benchmark one model configuration across all batch sizes.

    Args:
        config_name: Human-readable label (e.g. "baseline", "memopt").
        model: Decoder model already moved to ``device`` and in eval mode.
        batch_sizes: List of batch sizes to test.
        prompt_len: Number of prompt tokens.
        gen_len: Number of tokens to generate per step.
        device: Compute device.
        scheduler: Active MemoryScheduler (already started) or None.

    Returns:
        List of result dicts, one per batch size.  Failed rows contain "OOM"
        for numeric fields.
    """
    rows: List[Dict[str, object]] = []

    for bs in batch_sizes:
        prompt_ids = torch.zeros(bs, prompt_len, dtype=torch.long, device=device)

        try:
            ttft_ms = _measure_ttft(model, prompt_ids, device)
            tpot_ms, throughput, total_decode_wall_ms = _measure_decode_loop(
                model, prompt_ids, gen_len, device
            )
            total_latency_ms = ttft_ms + tpot_ms * gen_len
            total_time_s = round(total_latency_ms / 1000.0, 6)

            rows.append({
                "model": config_name,
                "config": config_name,  # legacy alias
                "batch_size": bs,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "ttft_ms": round(ttft_ms, 3),
                "tpot_ms": round(tpot_ms, 3),
                "throughput_tok_per_sec": round(throughput, 3),
                "throughput_tokens_per_sec": round(throughput, 3),  # legacy alias
                "total_time_s": total_time_s,
                "total_latency_ms": round(total_latency_ms, 3),  # legacy alias
            })
            print(
                f"  [{config_name}] bs={bs:>2}  TTFT={ttft_ms:7.1f}ms  "
                f"TPOT={tpot_ms:6.2f}ms/tok  "
                f"Throughput={throughput:8.1f} tok/s  "
                f"TotalLatency={total_latency_ms:8.1f}ms"
            )

        except torch.cuda.OutOfMemoryError:
            warnings.warn(
                f"[{config_name}] batch_size={bs}: CUDA OOM — skipping this row.",
                RuntimeWarning,
                stacklevel=2,
            )
            rows.append({
                "model": config_name,
                "config": config_name,
                "batch_size": bs,
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "ttft_ms": "OOM",
                "tpot_ms": "OOM",
                "throughput_tok_per_sec": "OOM",
                "throughput_tokens_per_sec": "OOM",
                "total_time_s": "OOM",
                "total_latency_ms": "OOM",
            })

    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "model",
    "batch_size",
    "ttft_ms",
    "tpot_ms",
    "throughput_tok_per_sec",
    "total_time_s",
]

# Legacy extended fields retained for internal result dicts (not written to CSV).
_INTERNAL_EXTRA_FIELDS = ["prompt_len", "gen_len", "total_latency_ms"]


def _write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write benchmark results to a CSV file.

    OOM entries are written with empty string values for numeric columns so that
    downstream tools do not misparse ``"OOM"`` as a float.

    Args:
        rows: List of result dicts.  Must contain ``model`` and ``batch_size``
            keys.  Numeric fields must be present or the row is treated as OOM.
        output_path: Destination ``.csv`` file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Normalise: map legacy "config" key → "model" if present.
            if "config" in row and "model" not in row:
                row = dict(row, model=row["config"])
            # Normalise: map legacy throughput key.
            if "throughput_tokens_per_sec" in row and "throughput_tok_per_sec" not in row:
                row = dict(row, throughput_tok_per_sec=row["throughput_tokens_per_sec"])
            # Normalise: map total_latency_ms → total_time_s (convert ms → s).
            if "total_latency_ms" in row and "total_time_s" not in row:
                tl = row["total_latency_ms"]
                if tl == "OOM":
                    row = dict(row, total_time_s="")
                else:
                    try:
                        row = dict(row, total_time_s=round(float(tl) / 1000.0, 6))
                    except (ValueError, TypeError):
                        row = dict(row, total_time_s="")
            # Write OOM rows with empty numeric fields.
            if row.get("ttft_ms") == "OOM":
                writer.writerow({
                    "model": row.get("model", row.get("config", "")),
                    "batch_size": row["batch_size"],
                    "ttft_ms": "",
                    "tpot_ms": "",
                    "throughput_tok_per_sec": "",
                    "total_time_s": "",
                })
            else:
                writer.writerow(row)
    print(f"\nCSV written to: {output_path}")


# ---------------------------------------------------------------------------
# Matplotlib figure
# ---------------------------------------------------------------------------


def _plot_results(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Generate a two-panel matplotlib figure (TTFT and Throughput vs batch size).

    Args:
        rows: List of result dicts from the benchmark.
        output_path: Destination PNG path.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "matplotlib is not installed — skipping figure generation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    # Partition rows by config, filtering out OOM entries
    configs: Dict[str, Dict[int, Dict[str, object]]] = {}
    for row in rows:
        cfg = str(row["config"])
        bs = int(row["batch_size"])
        if row["ttft_ms"] == "OOM":
            continue
        if cfg not in configs:
            configs[cfg] = {}
        configs[cfg][bs] = row

    if not configs:
        print("No non-OOM rows to plot.")
        return

    all_batch_sizes = sorted(
        {bs for cfg_data in configs.values() for bs in cfg_data.keys()}
    )
    bs_labels = [str(bs) for bs in all_batch_sizes]

    fig, (ax_ttft, ax_tp) = plt.subplots(1, 2, figsize=(12, 5))
    bar_width = 0.35
    n_configs = len(configs)
    offsets = [i * bar_width - bar_width * (n_configs - 1) / 2 for i in range(n_configs)]

    for idx, (cfg_name, cfg_data) in enumerate(configs.items()):
        ttft_vals = [float(cfg_data[bs]["ttft_ms"]) if bs in cfg_data else 0.0
                     for bs in all_batch_sizes]
        tp_vals = [float(cfg_data[bs]["throughput_tokens_per_sec"]) if bs in cfg_data else 0.0
                   for bs in all_batch_sizes]
        x_positions = [i + offsets[idx] for i in range(len(all_batch_sizes))]

        ax_ttft.bar(x_positions, ttft_vals, width=bar_width, label=cfg_name)
        ax_tp.bar(x_positions, tp_vals, width=bar_width, label=cfg_name)

    ax_ttft.set_title("TTFT vs Batch Size")
    ax_ttft.set_xlabel("Batch Size")
    ax_ttft.set_ylabel("TTFT (ms)")
    ax_ttft.set_xticks(range(len(all_batch_sizes)))
    ax_ttft.set_xticklabels(bs_labels)
    ax_ttft.legend()
    ax_ttft.grid(axis="y", alpha=0.3)

    ax_tp.set_title("Throughput vs Batch Size")
    ax_tp.set_xlabel("Batch Size")
    ax_tp.set_ylabel("Throughput (tokens/sec)")
    ax_tp.set_xticks(range(len(all_batch_sizes)))
    ax_tp.set_xticklabels(bs_labels)
    ax_tp.legend()
    ax_tp.grid(axis="y", alpha=0.3)

    fig.suptitle("MemOpt Latency Benchmark: TTFT and Throughput vs Batch Size")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Parsed namespace with all benchmark parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "MemOpt fine-grained latency benchmark. "
            "Measures TTFT and TPOT separately, compares Baseline vs MemOpt "
            "across configurable batch sizes."
        )
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--batch-sizes", "--batch-size",
        nargs="+",
        type=str,
        default=["1", "4", "16", "64"],
        metavar="N",
        dest="batch_sizes",
        help="Batch sizes to test — space-separated or comma-separated (default: 1 4 16 64).",
    )
    parser.add_argument(
        "--prompt-len", "--seq-len",
        type=int,
        default=128,
        metavar="N",
        dest="prompt_len",
        help="Prompt length in tokens (default: 128).",
    )
    parser.add_argument(
        "--gen-len",
        "--tokens",
        type=int,
        default=200,
        metavar="N",
        dest="gen_len",
        help="Number of tokens to generate per decode phase (default: 200).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        metavar="N",
        help="Warmup forward passes at batch_size=1 before timing (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "CSV output path.  When omitted, a timestamped file is written to "
            "--output-dir (e.g. results/latency_v2_20260101_120000.csv)."
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )
    parser.add_argument(
        "--model",
        choices=list(_PRESETS.keys()),
        default="tiny",
        help="Architecture preset: tiny (d=256,h=4,L=4) or medium (d=512,h=8,L=6). Default: tiny.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Smoke test: use batch_sizes=[1, 2] and gen_len=10 to verify "
            "no import/shape errors."
        ),
    )
    args = parser.parse_args()

    # Dry-run overrides.
    if args.dry_run:
        args.batch_sizes = [1, 2]
        args.gen_len = 10
    else:
        # Flatten comma-separated entries so both "1,4,16" and "1 4 16" work.
        flat: List[int] = []
        for item in args.batch_sizes:
            flat.extend(int(x) for x in item.split(",") if x.strip())
        args.batch_sizes = flat

    # Resolve output path.
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(args.output_dir) / f"latency_v2_{timestamp}.csv"

    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and execute the latency benchmark."""
    args = _parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda specified but torch.cuda.is_available() is False."
        )

    device = torch.device(args.device)
    preset = _PRESETS[args.model]

    print(f"Device        : {device}")
    print(f"Model preset  : {args.model}  "
          f"(d_model={preset['d_model']}, n_heads={preset['n_heads']}, "
          f"n_layers={preset['n_layers']})")
    print(f"Batch sizes   : {args.batch_sizes}")
    print(f"Prompt length : {args.prompt_len} tokens")
    print(f"Generate len  : {args.gen_len} tokens")
    print(f"Warmup passes : {args.warmup}")

    # ------------------------------------------------------------------
    # Build both models with identical seeds for a fair comparison
    # ------------------------------------------------------------------
    max_seq_len = args.prompt_len + args.gen_len

    torch.manual_seed(SEED)
    baseline_model = _build_model(
        preset,
        use_dynamic_attention=False,
        max_seq_len=max_seq_len,
    )
    baseline_model.eval()
    baseline_model.to(device)

    torch.manual_seed(SEED)
    # Pass MEMOPT_WINDOW_SIZE so DynamicAttention's KV truncation activates
    # once seq_len exceeds the window.  Without a positive window_size the
    # _sdpa_forward path adds overhead (extra reshapes, window checks) with no
    # KV savings, causing MemOpt to regress vs baseline at large batch sizes.
    memopt_model = _build_model(
        preset,
        use_dynamic_attention=True,
        max_seq_len=max_seq_len,
        window_size=MEMOPT_WINDOW_SIZE,
    )
    memopt_model.eval()
    memopt_model.to(device)

    print(f"MemOpt window : {MEMOPT_WINDOW_SIZE} tokens "
          f"(KV savings activate when seq_len > {MEMOPT_WINDOW_SIZE})")

    # ------------------------------------------------------------------
    # Warmup — run at each distinct batch size to avoid JIT compilation
    # overhead biasing TTFT measurements.  A single pass per batch size
    # is sufficient to trigger kernel specialisation.
    # ------------------------------------------------------------------
    print(f"\nWarming up ({args.warmup} passes per batch size) ...")
    warmup_batch_sizes = sorted(set([1] + args.batch_sizes))
    for wbs in warmup_batch_sizes:
        warmup_ids = torch.zeros(wbs, args.prompt_len, dtype=torch.long, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = baseline_model(warmup_ids)
                _ = memopt_model(warmup_ids)
            _sync(device)

    # ------------------------------------------------------------------
    # Baseline benchmark
    # ------------------------------------------------------------------
    print("\n--- Baseline (nn.MultiheadAttention) ---")
    baseline_rows = _run_config(
        config_name="baseline",
        model=baseline_model,
        batch_sizes=args.batch_sizes,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        device=device,
        scheduler=None,
    )

    # ------------------------------------------------------------------
    # MemOpt benchmark (with MemoryScheduler lifecycle)
    # ------------------------------------------------------------------
    print("\n--- MemOpt (DynamicAttention + MemoryScheduler) ---")
    scheduler = MemoryScheduler()
    scheduler.start()
    try:
        memopt_rows = _run_config(
            config_name="memopt",
            model=memopt_model,
            batch_sizes=args.batch_sizes,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            device=device,
            scheduler=scheduler,
        )
    finally:
        scheduler.stop()

    # ------------------------------------------------------------------
    # Collect and persist results
    # ------------------------------------------------------------------
    all_rows = baseline_rows + memopt_rows

    _write_csv(all_rows, args.output)
    print(f"Results saved to: {args.output}")

    if not args.no_plot:
        plot_path = args.output.with_suffix(".png")
        _plot_results(all_rows, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
