"""run_ablations.py — Ablation harness for MemOpt benchmark scripts.

Orchestrates bench_memory_v2.py, bench_latency_v2.py, and bench_accuracy.py
to produce a unified ablation JSON covering four configurations:

  A — Full MemOpt       : Ada-KV ON,  W4A16 ON,  AutoChunk ON
  B — No Pruning        : Ada-KV OFF, W4A16 ON,  AutoChunk ON
  C — No Quantization   : Ada-KV ON,  W4A16 OFF, AutoChunk ON
  D — No Chunking       : Ada-KV ON,  W4A16 ON,  AutoChunk OFF

Each configuration is approximated from 4 shared benchmark runs:

  1. bench_memory_v2.py  (fp16)   → baseline row, memopt row
  2. bench_memory_v2.py  (fp32)   → memopt fp32 row
  3. bench_latency_v2.py          → baseline row, memopt row (batch_size=1)
  4. bench_accuracy.py            → baseline, streaming_llm, memopt_full PPL

Mapping rationale:
  Config A — VRAM=memopt(fp16),   TPOT=memopt,    PPL=memopt_full
  Config B — VRAM=baseline(fp16), TPOT=baseline,  PPL=baseline
  Config C — VRAM=memopt(fp32),   TPOT=memopt,    PPL=memopt_full
  Config D — VRAM=memopt(fp16),   TPOT=baseline,  PPL=streaming_llm

Usage::

    # Smoke test (seq_len=128)
    python scripts/run_ablations.py --dry-run

    # Standard run at default seq_len=8192
    python scripts/run_ablations.py

    # Custom output
    python scripts/run_ablations.py --output-file results/my_ablation.json

    # Keep temp files
    python scripts/run_ablations.py --keep-tmp
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

_SCRIPTS_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _SCRIPTS_DIR.parent
_RESULTS_DIR: Path = _REPO_ROOT / "results"
_TMP_DIR: Path = _RESULTS_DIR / "ablation_tmp"


# ---------------------------------------------------------------------------
# Ablation config descriptors
# ---------------------------------------------------------------------------

# Internal extraction keys (not written to output JSON):
#   "_mem_key"  : (model_name: str, use_fp32: bool)
#   "_lat_key"  : (model_name: str,)
#   "_acc_key"  : config_name in accuracy JSON

_ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "config_a_full_memopt": {
        "description": "Full MemOpt: Ada-KV ON, W4A16 ON, AutoChunk ON",
        "flags": {"ada_kv": True, "w4a16": True, "autochunk": True},
        "source_model_memory": "memopt (fp16)",
        "source_model_latency": "memopt",
        "source_model_accuracy": "memopt_full",
        "_mem_key": ("memopt", False),
        "_lat_key": ("memopt",),
        "_acc_key": "memopt_full",
    },
    "config_b_no_pruning": {
        "description": "No Pruning: Ada-KV OFF, W4A16 ON, AutoChunk ON",
        "flags": {"ada_kv": False, "w4a16": True, "autochunk": True},
        "source_model_memory": "baseline (fp16)",
        "source_model_latency": "baseline",
        "source_model_accuracy": "baseline",
        "approximation_note": (
            "Ada-KV OFF approximated as baseline: no KV eviction → standard "
            "dense attention memory footprint and latency"
        ),
        "_mem_key": ("baseline", False),
        "_lat_key": ("baseline",),
        "_acc_key": "baseline",
    },
    "config_c_no_quant": {
        "description": "No Quantization: Ada-KV ON, W4A16 OFF, AutoChunk ON",
        "flags": {"ada_kv": True, "w4a16": False, "autochunk": True},
        "source_model_memory": "memopt (fp32)",
        "source_model_latency": "memopt",
        "source_model_accuracy": "memopt_full",
        "approximation_note": (
            "W4A16 OFF approximated by fp32 model dtype (--fp32 flag): "
            "no INT4 KV cache → ~2× higher VRAM than fp16 baseline"
        ),
        "_mem_key": ("memopt", True),
        "_lat_key": ("memopt",),
        "_acc_key": "memopt_full",
    },
    "config_d_no_chunking": {
        "description": "No Chunking: Ada-KV ON, W4A16 ON, AutoChunk OFF",
        "flags": {"ada_kv": True, "w4a16": True, "autochunk": False},
        "source_model_memory": "memopt (fp16)",
        "source_model_latency": "baseline",
        "source_model_accuracy": "streaming_llm",
        "approximation_note": (
            "AutoChunk OFF: VRAM unchanged (memory impact minimal); "
            "TPOT approximated as baseline (full-context attention without "
            "chunking ≈ baseline attention cost); PPL approximated via "
            "streaming_llm (windowed attention is the closest accuracy proxy "
            "for disabled AutoChunk)"
        ),
        "_mem_key": ("memopt", False),
        "_lat_key": ("baseline",),
        "_acc_key": "streaming_llm",
    },
}


# ---------------------------------------------------------------------------
# Safe float parsing
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    """Parse ``value`` to float; return ``None`` on failure, empty string, or NaN.

    Args:
        value: Any value that may be numeric, a string, ``None``, or empty.

    Returns:
        Parsed float, or ``None`` when the value represents a missing/OOM
        measurement (empty string, ``"nan"``, ``"OOM"``, actual NaN, or
        non-numeric content).
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


# ---------------------------------------------------------------------------
# CSV / JSON parsing helpers
# ---------------------------------------------------------------------------


def _find_latest_csv(pattern: str) -> Optional[Path]:
    """Return the most recently modified file matching ``pattern``.

    Args:
        pattern: Glob pattern string, e.g.
            ``"results/ablation_tmp/memory_v2_*.csv"``.

    Returns:
        Path to the most recently modified match, or ``None`` if none found.
    """
    matches = glob.glob(pattern)
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


def _parse_memory_csv(
    csv_path: Path,
    seq_len: int,
    model_name: str,
) -> Optional[float]:
    """Extract ``active_mib`` from a memory_v2 CSV row.

    Locates the row where ``model == model_name`` and
    ``seq_len == seq_len``.  OOM rows have an empty string in
    ``active_mib`` and are returned as ``None``.

    Args:
        csv_path: Path to the ``memory_v2_*.csv`` file.
        seq_len: Target sequence length to filter on.
        model_name: Model label to filter on (``"baseline"`` or ``"memopt"``).

    Returns:
        ``active_mib`` as float, or ``None`` if unavailable.
    """
    try:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    row_seq = int(row.get("seq_len", -1))
                except (ValueError, TypeError):
                    continue
                if row.get("model") == model_name and row_seq == seq_len:
                    return _safe_float(row.get("active_mib", ""))
    except OSError as exc:
        print(f"  WARNING: could not read {csv_path}: {exc}")
    return None


def _parse_latency_csv(
    csv_path: Path,
    batch_size: int,
    model_name: str,
) -> Optional[float]:
    """Extract ``tpot_ms`` from a latency_v2 CSV row.

    Locates the row where ``model == model_name`` and
    ``batch_size == batch_size``.  OOM rows have an empty string and are
    returned as ``None``.

    Args:
        csv_path: Path to the ``latency_v2_*.csv`` file.
        batch_size: Target batch size to filter on.
        model_name: Model label to filter on (``"baseline"`` or ``"memopt"``).

    Returns:
        ``tpot_ms`` as float, or ``None`` if unavailable.
    """
    try:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    row_bs = int(row.get("batch_size", -1))
                except (ValueError, TypeError):
                    continue
                if row.get("model") == model_name and row_bs == batch_size:
                    return _safe_float(row.get("tpot_ms", ""))
    except OSError as exc:
        print(f"  WARNING: could not read {csv_path}: {exc}")
    return None


def _parse_accuracy_json(
    json_path: Path,
    config_name: str,
    context_length: int,
    dataset: str = "wikitext2",
) -> Optional[float]:
    """Extract perplexity from an accuracy benchmark JSON file.

    Expected JSON structure::

        {
          "results": {
            "wikitext2": {
              "baseline":      {"1024": 12.3, "8192": 15.0},
              "streaming_llm": {"1024": 14.1},
              "memopt_full":   {"1024": 13.0}
            }
          }
        }

    A ``{"status": "OOM"}`` or ``{"status": "ERROR", ...}`` entry is treated
    as ``None``.

    Args:
        json_path: Path to the ``accuracy.json`` file.
        config_name: One of ``"baseline"``, ``"streaming_llm"``,
            ``"memopt_full"``.
        context_length: The context length key (stored as a string in JSON).
        dataset: Dataset name key, e.g. ``"wikitext2"``.

    Returns:
        Perplexity as float, or ``None`` if not available.
    """
    try:
        with json_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  WARNING: could not read {json_path}: {exc}")
        return None

    results = data.get("results", {})
    dataset_results = results.get(dataset, {})
    config_results = dataset_results.get(config_name, {})
    val = config_results.get(str(context_length))
    if val is None:
        return None
    # OOM / ERROR sentinels are dicts
    if isinstance(val, dict):
        return None
    return _safe_float(val)


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _run_subprocess(
    cmd: List[str],
    label: str,
    step: int,
    total_steps: int,
) -> Tuple[bool, str]:
    """Execute a subprocess synchronously; print progress to stdout.

    Non-zero return codes are logged as warnings but do NOT abort the harness.
    The caller records ``null`` for affected metrics and continues.

    Args:
        cmd: Full command list including interpreter path.
        label: Human-readable description printed in the progress header.
        step: Current 1-indexed step number.
        total_steps: Total number of steps in this harness run.

    Returns:
        Tuple of ``(success: bool, stderr: str)``.
        ``success`` is ``False`` when the return code is non-zero or the
        subprocess could not be launched.
    """
    print(f"\n[{step}/{total_steps}] Running {label}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        print(f"  WARNING: subprocess launch failed: {exc}")
        return False, str(exc)

    if result.returncode != 0:
        stderr_tail = (result.stderr or "").strip()[-1200:]
        print(f"  WARNING: {label} exited with code {result.returncode}")
        if stderr_tail:
            print(f"  stderr (tail):\n{stderr_tail}")
        return False, result.stderr or ""

    return True, result.stderr or ""


# ---------------------------------------------------------------------------
# Output JSON builder
# ---------------------------------------------------------------------------


def _build_output(
    seq_len: int,
    dry_run: bool,
    model_preset: str,
    mem_fp16_csv: Optional[Path],
    mem_fp32_csv: Optional[Path],
    lat_csv: Optional[Path],
    acc_json: Optional[Path],
) -> Dict[str, Any]:
    """Aggregate raw benchmark files into the ablation output structure.

    Extracts raw metric values from each file and maps them to the four
    ablation configurations.  Missing or OOM values are represented as
    ``None`` (serialised as JSON ``null``).

    Args:
        seq_len: Sequence length used for all subprocess runs.
        dry_run: Whether this was invoked as a dry-run.
        model_preset: Model preset label forwarded to bench_memory_v2 (e.g.
            ``"tiny"``).
        mem_fp16_csv: Path to the fp16 memory CSV, or ``None`` on failure.
        mem_fp32_csv: Path to the fp32 memory CSV, or ``None`` on failure.
        lat_csv: Path to the latency CSV, or ``None`` on failure.
        acc_json: Path to the accuracy JSON, or ``None`` on failure.

    Returns:
        Full ablation output dict ready for JSON serialisation.
    """
    # ---- Extract raw metrics from files -----------------------------------
    vram_memopt_fp16: Optional[float] = (
        _parse_memory_csv(mem_fp16_csv, seq_len, "memopt") if mem_fp16_csv else None
    )
    vram_baseline_fp16: Optional[float] = (
        _parse_memory_csv(mem_fp16_csv, seq_len, "baseline") if mem_fp16_csv else None
    )
    vram_memopt_fp32: Optional[float] = (
        _parse_memory_csv(mem_fp32_csv, seq_len, "memopt") if mem_fp32_csv else None
    )
    tpot_memopt: Optional[float] = (
        _parse_latency_csv(lat_csv, 1, "memopt") if lat_csv else None
    )
    tpot_baseline: Optional[float] = (
        _parse_latency_csv(lat_csv, 1, "baseline") if lat_csv else None
    )
    ppl_baseline: Optional[float] = (
        _parse_accuracy_json(acc_json, "baseline", seq_len) if acc_json else None
    )
    ppl_streaming: Optional[float] = (
        _parse_accuracy_json(acc_json, "streaming_llm", seq_len) if acc_json else None
    )
    ppl_memopt: Optional[float] = (
        _parse_accuracy_json(acc_json, "memopt_full", seq_len) if acc_json else None
    )

    # ---- Per-config metric assignments ------------------------------------
    # A: VRAM=memopt_fp16,   TPOT=memopt,    PPL=memopt_full
    # B: VRAM=baseline_fp16, TPOT=baseline,  PPL=baseline
    # C: VRAM=memopt_fp32,   TPOT=memopt,    PPL=memopt_full
    # D: VRAM=memopt_fp16,   TPOT=baseline,  PPL=streaming_llm
    raw: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {
        "config_a_full_memopt": (vram_memopt_fp16,   tpot_memopt,    ppl_memopt),
        "config_b_no_pruning":  (vram_baseline_fp16, tpot_baseline,  ppl_baseline),
        "config_c_no_quant":    (vram_memopt_fp32,   tpot_memopt,    ppl_memopt),
        "config_d_no_chunking": (vram_memopt_fp16,   tpot_baseline,  ppl_streaming),
    }

    # ---- Assemble per-config output entries --------------------------------
    configs_out: Dict[str, Any] = {}
    for cfg_key, descriptor in _ABLATION_CONFIGS.items():
        vram, tpot, ppl = raw[cfg_key]
        entry: Dict[str, Any] = {
            "description": descriptor["description"],
            "flags": descriptor["flags"],
            "peak_vram_mib": vram,
            "tpot_ms": tpot,
            "ppl_wikitext2": ppl,
            "source_model_memory": descriptor["source_model_memory"],
            "source_model_latency": descriptor["source_model_latency"],
            "source_model_accuracy": descriptor["source_model_accuracy"],
        }
        if "approximation_note" in descriptor:
            entry["approximation_note"] = descriptor["approximation_note"]
        configs_out[cfg_key] = entry

    # ---- Metadata ----------------------------------------------------------
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "metadata": {
            "seq_len": seq_len,
            "date": now_utc,
            "model_preset": model_preset,
            "dry_run": dry_run,
            "scripts_run": [
                "bench_memory_v2.py",
                "bench_latency_v2.py",
                "bench_accuracy.py",
            ],
        },
        "ablation_configs": configs_out,
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], fmt: str = ".1f", unit: str = "") -> str:
    """Format a nullable metric value for console display.

    Args:
        value: Numeric value or ``None``.
        fmt: Python format spec for the number (default ``".1f"``).
        unit: Suffix appended after the number (e.g. ``" MiB"``).

    Returns:
        Formatted string, or ``"N/A"`` when ``value`` is ``None``.
    """
    if value is None:
        return "N/A"
    return f"{value:{fmt}}{unit}"


def _print_summary(output: Dict[str, Any]) -> None:
    """Print the aggregated results table to stdout.

    Args:
        output: Full ablation output dict as returned by :func:`_build_output`.
    """
    print("\n=== Aggregation Results ===")
    labels = {
        "config_a_full_memopt": "Config A (Full MemOpt)  ",
        "config_b_no_pruning":  "Config B (No Pruning)   ",
        "config_c_no_quant":    "Config C (No Quant)     ",
        "config_d_no_chunking": "Config D (No Chunking)  ",
    }
    for cfg_key, label in labels.items():
        cfg = output["ablation_configs"][cfg_key]
        vram = _fmt(cfg["peak_vram_mib"], ".1f", " MiB")
        tpot = _fmt(cfg["tpot_ms"], ".2f", "ms")
        ppl = _fmt(cfg["ppl_wikitext2"], ".1f")
        print(f"{label}: VRAM={vram}  TPOT={tpot}  PPL={ppl}")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation harness.

    Returns:
        Parsed namespace containing all benchmark parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "MemOpt ablation harness: runs bench_memory_v2, bench_latency_v2, "
            "and bench_accuracy as subprocesses and aggregates results into a "
            "unified ablation JSON over four configurations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        metavar="N",
        help="Fixed sequence length for all subprocesses.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=_RESULTS_DIR / "ablation_data.json",
        metavar="PATH",
        help="Destination path for the aggregated ablation JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Pass --dry-run to bench_accuracy.py and use seq_len=128 "
            "(overrides --seq-len)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Torch device string forwarded to all subprocesses "
            "(e.g. 'cuda', 'cpu').  Defaults to auto-detect."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        metavar="PRESET",
        help="Model preset forwarded to bench_memory_v2.py ('tiny', 'medium').",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help=(
            "Keep results/ablation_tmp/ after completion. "
            "By default the directory is deleted when all subprocesses succeed."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="DATASET",
        help=(
            "Dataset name forwarded to bench_accuracy.py "
            "(e.g. 'wikitext2', 'ptb').  Defaults to the sub-script default."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate the ablation benchmark and write the output JSON.

    Runs four subprocesses sequentially: memory fp16, memory fp32, latency,
    and accuracy.  Each subprocess failure is handled gracefully — affected
    metrics are recorded as ``null`` in the output JSON, and the harness
    continues with remaining subprocesses.

    On completion, prints a summary table to stdout and persists the aggregated
    JSON to ``--output-file``.  The temp directory is deleted when all
    subprocesses succeed and ``--keep-tmp`` is not set.
    """
    args = _parse_args()

    seq_len: int = 128 if args.dry_run else args.seq_len
    output_file: Path = Path(args.output_file).resolve()

    print("=== MemOpt Ablation Study ===")
    print(f"Sequence length : {seq_len}")
    print(f"Output          : {output_file}")

    # Ensure required directories exist.
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    interp: str = sys.executable
    total_steps: int = 4
    all_succeeded: bool = True
    device_flags: List[str] = (["--device", args.device] if args.device else [])

    # -------------------------------------------------------------------
    # Step 1: VRAM benchmark (fp16)
    # -------------------------------------------------------------------
    cmd_mem_fp16: List[str] = [
        interp,
        str(_SCRIPTS_DIR / "bench_memory_v2.py"),
        "--seq-lengths", str(seq_len),
        "--model", args.model,
        "--no-plot",
        "--output-dir", str(_TMP_DIR),
    ] + device_flags

    ok1, _ = _run_subprocess(
        cmd_mem_fp16, "bench_memory_v2.py (fp16)", 1, total_steps
    )
    all_succeeded = all_succeeded and ok1

    # Snapshot all memory CSVs present after step 1 to distinguish fp16 vs fp32.
    mem_csvs_after_step1 = set(
        glob.glob(str(_TMP_DIR / "memory_v2_*.csv"))
    )
    fp16_csv_path = _find_latest_csv(str(_TMP_DIR / "memory_v2_*.csv"))
    status1 = "OK" if ok1 else "FAILED"
    print(f"      → {fp16_csv_path or '(no CSV found)'}  ({status1})")

    # -------------------------------------------------------------------
    # Step 2: VRAM benchmark (fp32)
    # -------------------------------------------------------------------
    cmd_mem_fp32: List[str] = [
        interp,
        str(_SCRIPTS_DIR / "bench_memory_v2.py"),
        "--seq-lengths", str(seq_len),
        "--model", args.model,
        "--fp32",
        "--no-plot",
        "--output-dir", str(_TMP_DIR),
    ] + device_flags

    ok2, _ = _run_subprocess(
        cmd_mem_fp32, "bench_memory_v2.py (fp32)", 2, total_steps
    )
    all_succeeded = all_succeeded and ok2

    # The fp32 CSV is whichever memory_v2_*.csv is new since step 1.
    mem_csvs_after_step2 = set(
        glob.glob(str(_TMP_DIR / "memory_v2_*.csv"))
    )
    new_csvs = mem_csvs_after_step2 - mem_csvs_after_step1
    fp32_csv_path: Optional[Path] = (
        Path(max(new_csvs, key=os.path.getmtime)) if new_csvs else None
    )
    status2 = "OK" if ok2 else "FAILED"
    print(f"      → {fp32_csv_path or '(no fp32 CSV found)'}  ({status2})")

    # -------------------------------------------------------------------
    # Step 3: Latency benchmark
    # -------------------------------------------------------------------
    cmd_lat: List[str] = [
        interp,
        str(_SCRIPTS_DIR / "bench_latency_v2.py"),
        "--batch-sizes", "1",
        "--no-plot",
        "--output-dir", str(_TMP_DIR),
    ] + device_flags

    ok3, _ = _run_subprocess(
        cmd_lat, "bench_latency_v2.py", 3, total_steps
    )
    all_succeeded = all_succeeded and ok3

    lat_csv = _find_latest_csv(str(_TMP_DIR / "latency_v2_*.csv"))
    status3 = "OK" if ok3 else "FAILED"
    print(f"      → {lat_csv or '(no CSV found)'}  ({status3})")

    # -------------------------------------------------------------------
    # Step 4: Accuracy benchmark
    # -------------------------------------------------------------------
    acc_json_path = _TMP_DIR / "accuracy.json"
    cmd_acc: List[str] = [
        interp,
        str(_SCRIPTS_DIR / "bench_accuracy.py"),
        "--context-lengths", str(seq_len),
        "--evaluate-dataset", args.dataset if args.dataset else "wikitext2",
        "--output-file", str(acc_json_path),
        "--no-plot",
    ] + device_flags
    if args.dry_run:
        cmd_acc.append("--dry-run")

    ok4, _ = _run_subprocess(
        cmd_acc, "bench_accuracy.py", 4, total_steps
    )
    all_succeeded = all_succeeded and ok4

    acc_path: Optional[Path] = (
        acc_json_path if acc_json_path.exists() else None
    )
    status4 = "OK" if ok4 else "FAILED"
    print(f"      → {acc_json_path}  ({status4})")

    # -------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------
    output = _build_output(
        seq_len=seq_len,
        dry_run=args.dry_run,
        model_preset=args.model,
        mem_fp16_csv=fp16_csv_path,
        mem_fp32_csv=fp32_csv_path,
        lat_csv=lat_csv,
        acc_json=acc_path,
    )

    _print_summary(output)

    # -------------------------------------------------------------------
    # Persist JSON
    # -------------------------------------------------------------------
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSaved → {output_file}")

    # -------------------------------------------------------------------
    # Cleanup temp directory (only when all subprocesses succeeded)
    # -------------------------------------------------------------------
    if not args.keep_tmp and all_succeeded:
        try:
            shutil.rmtree(_TMP_DIR)
        except OSError as exc:
            print(f"  WARNING: could not remove {_TMP_DIR}: {exc}")


if __name__ == "__main__":
    main()
