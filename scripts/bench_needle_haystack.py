"""Needle-In-A-Haystack (NIAH) benchmark: Baseline vs StreamingLLM vs MemOpt.

Location: scripts/bench_needle_haystack.py
Summary: Evaluates the retrieval capability of three Transformer configurations by
    embedding a "needle" fact at a specific fractional depth inside a large "haystack"
    context, then prompting the model to recall that fact.  Results are written as a
    JSON file for downstream analysis and heatmap plotting via plot_results.py.

Configurations evaluated
------------------------
* ``baseline``      — Standard causal SDPA; no KV eviction or masking.
* ``streaming_llm`` — StreamingLLM (arXiv 2309.17453): sink + sliding-window masking.
* ``memopt_full``   — DynamicAttention (Ada-KV pruning + W4A16 quant) from src/memopt/.

Used with / by
--------------
- scripts/bench_accuracy.py  — provides wrapper classes and patch utility (imported)
- scripts/plot_results.py    — consumes the JSON output to render NIAH heatmaps
- src/memopt/attention.py    — DynamicAttention module (memopt_full config)
- src/memopt/scheduler.py    — MemoryScheduler (memopt_full config)

Usage
-----
    # Quick smoke test (GPT-2, short contexts, all configs)
    python scripts/bench_needle_haystack.py --context-lens 128,256 --depths 0.0,0.5,1.0

    # Default run (GPT-2, context <= 1024 only due to model position limit)
    python scripts/bench_needle_haystack.py

    # Select specific configs
    python scripts/bench_needle_haystack.py --configs baseline,streaming_llm
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project path setup — allows running from repo root without pip install.
# ---------------------------------------------------------------------------
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch

# ---------------------------------------------------------------------------
# Optional tqdm for progress bars.
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm  # type: ignore[import]

    _TQDM_AVAILABLE: bool = True
except ImportError:
    _TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import wrapper classes and patch utility from bench_accuracy.
# These are imported rather than duplicated to keep DRY conventions.
# ---------------------------------------------------------------------------
try:
    _scripts_dir = str(Path(__file__).resolve().parent)
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    from bench_accuracy import (  # type: ignore[import]
        _HFBaselineWrapper,
        _HFMemOptWrapper,
        _HFStreamingLLMWrapper,
        patch_gpt2_with_dynamic_attention,
    )

    _BENCH_ACCURACY_AVAILABLE: bool = True
except Exception as _exc:  # noqa: BLE001
    warnings.warn(
        f"Could not import wrappers from bench_accuracy ({_exc}). "
        "Will use inline model.generate() without attention patching — "
        "streaming_llm and memopt_full configs will behave like baseline.",
        ImportWarning,
        stacklevel=1,
    )
    _BENCH_ACCURACY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional memopt imports — degrade gracefully when the C++ extension is absent.
# ---------------------------------------------------------------------------
try:
    from memopt.attention import DynamicAttention  # noqa: F401
    from memopt.scheduler import MemoryScheduler, SchedulerConfig  # noqa: F401

    _MEMOPT_AVAILABLE: bool = True
except Exception:  # noqa: BLE001
    _MEMOPT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional HuggingFace transformers
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    _TRANSFORMERS_AVAILABLE: bool = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("bench_needle_haystack")

# ===========================================================================
# Module-level constants — tune here without touching the logic below.
# ===========================================================================

NEEDLE: str = "The secret password is 'memopt2026'."
NEEDLE_QUESTION: str = "What is the secret password mentioned in the text?"
NEEDLE_ANSWER: str = "memopt2026"  # substring used for retrieval check

# Context lengths to probe.  GPT-2 max_position_embeddings=1024, so entries
# above 1024 are automatically skipped for GPT-2 (see evaluate_configuration).
CONTEXT_LENGTHS: List[int] = [1024, 2048, 4096, 8192, 16384, 32768]

# Fractional positions (0.0 = very start, 1.0 = very end) at which to insert
# the needle inside the haystack.
NEEDLE_DEPTHS: List[float] = [0.0, 0.25, 0.50, 0.75, 1.00]

# Haystack filler text.  ~500 words of generic technical prose that can be
# repeated to fill any context length.  Chosen to be factually bland so that
# the needle stands out clearly as the only "secret" piece of information.
HAYSTACK_TEXT: str = """
Software engineering is fundamentally about managing complexity. As systems grow
larger, the challenges multiply: codebases sprawl across thousands of files,
dependencies proliferate, and the invisible connections between components become
harder to reason about. Good engineers learn to impose structure on this chaos
through abstraction, modularization, and disciplined documentation.

Startups face a particular version of this challenge. Early on, a small team
can hold the entire system in their heads. The architecture is often informal,
decisions are made quickly, and technical debt accumulates at a rate that feels
acceptable when the goal is shipping fast. But as the company grows, that
informality becomes a liability. Engineers who join later cannot easily understand
the system. Velocity slows. Regressions appear in unexpected places.

The best organizations recognize this inflection point and invest in paying down
technical debt before it becomes crippling. This typically means writing tests,
refactoring modules that have grown too large, and establishing conventions that
new engineers can follow. The payoff is compounding: a well-structured codebase
makes every future feature cheaper to build.

Machine learning systems add another dimension to this complexity. A traditional
software bug is deterministic: given the same inputs, the program produces the
same wrong output. An ML bug might be stochastic, only manifesting under certain
data distributions or hardware configurations. Debugging requires a different
mindset—statistical rather than logical—and tooling that most organizations are
still building.

Memory management is one of the core challenges in deep learning. Modern
Transformer models are enormous, and the activations they produce during a
forward pass can easily exceed available GPU VRAM, especially for long sequences.
Researchers have developed a variety of techniques to address this: gradient
checkpointing, quantization, tensor parallelism, and more recently, attention
mechanisms that avoid materializing the full attention matrix. Each approach
involves trade-offs between speed, memory, and numerical precision.

The key-value cache is another major memory consumer. During autoregressive
generation, a language model produces one token at a time, and to avoid
recomputing attention over all previous tokens at each step, the intermediate
key and value tensors are cached. For long sequences, this cache can grow to
be larger than the model weights themselves. Efficient KV cache management is
therefore critical for serving large language models in production.

Programming languages have evolved significantly to support concurrent and
distributed computation. Languages like Rust provide strong safety guarantees
without sacrificing performance, while Python's ecosystem of scientific
computing libraries makes it the lingua franca of machine learning research.
The tension between research velocity and production reliability often plays
out at this language boundary—prototype in Python, deploy in C++ or Rust.

Open-source communities have been central to the progress of machine learning.
Frameworks like PyTorch and TensorFlow lowered the barrier to entry dramatically,
allowing researchers without systems expertise to experiment with novel
architectures. The result has been an explosion of innovation: transformers,
diffusion models, reinforcement learning from human feedback, and many other
paradigms that would have been inaccessible to most researchers a decade ago.

Infrastructure as code has become standard practice in software engineering.
Tools like Terraform and Kubernetes allow teams to define their deployment
environments declaratively, making it possible to reproduce environments exactly
and to version-control infrastructure alongside application code. This shift
has reduced the gap between development and production, enabling faster and
more reliable deployments.

Data quality is often the limiting factor in machine learning projects. A model
is only as good as its training data, and curating high-quality datasets requires
significant human effort. Annotation pipelines, quality filters, and deduplication
routines all contribute to the final dataset, and mistakes at any stage can
propagate silently into model behavior. Careful data provenance tracking is
therefore an essential practice for teams that care about reproducibility.
"""

# StreamingLLM / DynamicAttention parameters — must match bench_accuracy defaults.
WINDOW_SIZE: int = 256
NUM_SINK_TOKENS: int = 4

# Default HuggingFace model name.
MODEL_NAME: str = "gpt2"

# Device selection.
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Config name constants — keep consistent with bench_accuracy.py.
_CFG_BASELINE: str = "baseline"
_CFG_STREAMING: str = "streaming_llm"
_CFG_MEMOPT: str = "memopt_full"

_ALL_CONFIGS: List[str] = [_CFG_BASELINE, _CFG_STREAMING, _CFG_MEMOPT]


# ===========================================================================
# Context construction helpers
# ===========================================================================


def build_context(
    context_length_tokens: int,
    depth: float,
    tokenizer,
) -> Tuple[str, int]:
    """Assemble a haystack context with the needle embedded at a given depth.

    The haystack is built by repeating ``HAYSTACK_TEXT`` until it fills
    ``context_length_tokens - len(needle_tokens) - len(question_tokens) - 50``
    tokens.  The needle is then inserted at position
    ``int(haystack_token_count * depth)`` within the haystack token sequence.
    Finally, the question prompt is appended.

    Args:
        context_length_tokens: Target total token count.
        depth: Fractional insertion position in [0.0, 1.0].
            0.0 places the needle at the very beginning of the haystack.
            1.0 places it at the very end.
        tokenizer: A HuggingFace tokenizer instance.

    Returns:
        Tuple of:
        - assembled_text: The full prompt string (haystack + needle + question).
        - total_tokens: The actual token count of assembled_text.
    """
    # Tokenize the fixed components to know their sizes.
    needle_tokens: List[int] = tokenizer.encode(NEEDLE, add_special_tokens=False)
    question_prompt: str = f"\n\nQuestion: {NEEDLE_QUESTION}\nAnswer:"
    question_tokens: List[int] = tokenizer.encode(question_prompt, add_special_tokens=False)

    # Budget for haystack tokens — leave a 50-token safety margin.
    haystack_budget: int = max(
        0,
        context_length_tokens - len(needle_tokens) - len(question_tokens) - 50,
    )

    # Repeat haystack text to build enough tokens.
    base_tokens: List[int] = tokenizer.encode(HAYSTACK_TEXT, add_special_tokens=False)
    if len(base_tokens) == 0:
        base_tokens = [tokenizer.eos_token_id or 0]

    # Tile and truncate to exact budget length.
    reps: int = max(1, (haystack_budget // len(base_tokens)) + 1)
    haystack_tokens: List[int] = (base_tokens * reps)[:haystack_budget]

    # Compute insertion index from fractional depth.
    # depth=0.0 → index 0 (needle at start); depth=1.0 → needle at end.
    insert_idx: int = int(len(haystack_tokens) * depth)
    insert_idx = max(0, min(insert_idx, len(haystack_tokens)))

    # Splice needle into haystack token list.
    full_tokens: List[int] = (
        haystack_tokens[:insert_idx]
        + needle_tokens
        + haystack_tokens[insert_idx:]
        + question_tokens
    )

    # Decode back to text so we can pass a string to the tokenizer later.
    assembled_text: str = tokenizer.decode(full_tokens, skip_special_tokens=True)
    total_tokens: int = len(full_tokens)

    return assembled_text, total_tokens


def check_retrieval(generated_text: str) -> int:
    """Check whether the model output contains the needle answer.

    Args:
        generated_text: The decoded text produced by the model.

    Returns:
        1 if ``NEEDLE_ANSWER`` appears (case-insensitive) in ``generated_text``,
        otherwise 0.
    """
    return 1 if NEEDLE_ANSWER.lower() in generated_text.lower() else 0


# ===========================================================================
# Model loading
# ===========================================================================


def load_hf_model(model_name: str, device: str):
    """Load a HuggingFace causal LM and tokenizer.

    Uses fp16 precision on CUDA, fp32 on CPU.

    Args:
        model_name: HuggingFace model identifier (e.g. "gpt2").
        device: Target device string ("cuda" or "cpu").

    Returns:
        Tuple of (model, tokenizer).  Both are None on load failure.

    Raises:
        RuntimeError: If transformers is not installed.
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "The 'transformers' package is required.  "
            "Install with: pip install transformers"
        )

    logger.info("Loading tokenizer '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad_token is set (GPT-2 has no pad token by default).
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model '%s' onto device=%s...", model_name, device)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()

    return model, tokenizer


# ===========================================================================
# Evaluation
# ===========================================================================


def evaluate_configuration(
    config_name: str,
    wrapper,
    tokenizer,
    context_lengths: List[int],
    depths: List[float],
    device: str,
    max_new_tokens: int = 50,
) -> List[Dict]:
    """Run NIAH trials for one model configuration.

    For each (context_length, depth) pair:
    1. Builds the context via ``build_context()``.
    2. Skips if total_tokens exceeds ``model.config.max_position_embeddings``.
    3. Runs greedy generation via ``wrapper.model.generate()``.
    4. Decodes only the newly generated tokens.
    5. Calls ``check_retrieval()`` on the generated text.

    Args:
        config_name: Human-readable name for this configuration (e.g. "baseline").
        wrapper: One of ``_HFBaselineWrapper``, ``_HFStreamingLLMWrapper``,
            ``_HFMemOptWrapper``.  Must expose ``.model`` (the HF CausalLM) and
            optionally override ``forward()``.
        tokenizer: HuggingFace tokenizer.
        context_lengths: Sorted list of context window sizes to probe.
        depths: Fractional insertion positions, each in [0.0, 1.0].
        device: Device string ("cuda" or "cpu").
        max_new_tokens: Maximum new tokens to generate per trial.

    Returns:
        List of result dicts with keys:
        ``model``, ``context_len``, ``depth``, ``accuracy``, ``generated``,
        ``total_tokens``.
    """
    records: List[Dict] = []

    # Determine the maximum position embedding supported by the underlying model.
    hf_model = wrapper.model
    max_pos: Optional[int] = getattr(
        getattr(hf_model, "config", None), "max_position_embeddings", None
    )

    # Determine whether this config needs use_cache=False during generation.
    # _HFMemOptWrapper uses DynamicAttention, which does not return KV-cache
    # present tuples, so use_cache must be False.
    is_memopt: bool = config_name == _CFG_MEMOPT

    # Build the cartesian product of trials.
    trials = [(cl, d) for cl in context_lengths for d in depths]

    iterable = (
        tqdm(trials, desc=f"[{config_name}]", unit="trial")
        if _TQDM_AVAILABLE
        else trials
    )

    for context_len, depth in iterable:
        logger.info(
            "[%s] Starting trial: context_len=%d, depth=%.2f",
            config_name,
            context_len,
            depth,
        )

        # Build the prompt and measure its token count.
        assembled_text, total_tokens = build_context(context_len, depth, tokenizer)

        # Skip trials where the prompt would exceed the model's positional limit.
        if max_pos is not None and total_tokens > max_pos:
            logger.warning(
                "[%s] Skipping context_len=%d, depth=%.2f — "
                "total_tokens=%d exceeds max_position_embeddings=%d.",
                config_name,
                context_len,
                depth,
                total_tokens,
                max_pos,
            )
            records.append(
                {
                    "model": config_name,
                    "context_len": context_len,
                    "depth": depth,
                    "accuracy": None,
                    "generated": None,
                    "total_tokens": total_tokens,
                }
            )
            continue

        # Tokenize the assembled prompt.
        input_ids = tokenizer.encode(assembled_text, return_tensors="pt").to(device)

        # Defensive guard: also check actual tokenized length.
        actual_len: int = input_ids.shape[-1]
        if max_pos is not None and actual_len > max_pos:
            logger.warning(
                "[%s] Skipping context_len=%d, depth=%.2f — "
                "actual input_ids length=%d exceeds max_position_embeddings=%d.",
                config_name,
                context_len,
                depth,
                actual_len,
                max_pos,
            )
            records.append(
                {
                    "model": config_name,
                    "context_len": context_len,
                    "depth": depth,
                    "accuracy": None,
                    "generated": None,
                    "total_tokens": actual_len,
                }
            )
            continue

        # Run greedy generation.
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        if is_memopt:
            generate_kwargs["use_cache"] = False

        try:
            with torch.no_grad():
                output_ids = hf_model.generate(input_ids, **generate_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[%s] Generation failed at context_len=%d, depth=%.2f: %s",
                config_name,
                context_len,
                depth,
                exc,
            )
            records.append(
                {
                    "model": config_name,
                    "context_len": context_len,
                    "depth": depth,
                    "accuracy": None,
                    "generated": f"ERROR: {exc}",
                    "total_tokens": actual_len,
                }
            )
            continue

        # Decode only the newly generated tokens (slice off the input).
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        generated_text: str = tokenizer.decode(new_tokens, skip_special_tokens=True)

        hit: int = check_retrieval(generated_text)

        logger.info(
            "[%s] context_len=%d, depth=%.2f → accuracy=%d, generated=%r",
            config_name,
            context_len,
            depth,
            hit,
            generated_text[:80],
        )

        records.append(
            {
                "model": config_name,
                "context_len": context_len,
                "depth": depth,
                "accuracy": hit,
                "generated": generated_text[:100],
                "total_tokens": actual_len,
            }
        )

    return records


# ===========================================================================
# CLI argument parsing
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the NIAH benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Needle-In-A-Haystack (NIAH) benchmark for MemOpt Transformer configurations."
        )
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"HuggingFace model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--context-lens",
        default="1024,2048,4096,8192",
        help=(
            "Comma-separated context lengths to probe "
            "(default: '1024,2048,4096,8192'). "
            "Lengths exceeding the model's max_position_embeddings are skipped."
        ),
    )
    parser.add_argument(
        "--depths",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated needle depths in [0.0, 1.0] (default: '0.0,0.25,0.5,0.75,1.0')",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for JSON output (default: 'results')",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate per trial (default: 50)",
    )
    parser.add_argument(
        "--configs",
        default=",".join(_ALL_CONFIGS),
        help=(
            f"Comma-separated subset of configs to run "
            f"(default: '{','.join(_ALL_CONFIGS)}')"
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"StreamingLLM / DynamicAttention recency window (default: {WINDOW_SIZE})",
    )
    parser.add_argument(
        "--num-sink-tokens",
        type=int,
        default=NUM_SINK_TOKENS,
        help=f"Number of attention sink tokens (default: {NUM_SINK_TOKENS})",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help=f"Compute device (default: '{DEVICE}')",
    )
    return parser.parse_args()


# ===========================================================================
# Main entry point
# ===========================================================================


def main() -> None:
    """Load models, run NIAH trials, and save results to JSON."""
    args = _parse_args()

    # Parse CLI lists.
    try:
        context_lengths: List[int] = [int(x.strip()) for x in args.context_lens.split(",")]
    except ValueError as exc:
        logger.error("Invalid --context-lens value: %s", exc)
        sys.exit(1)

    try:
        depths: List[float] = [float(x.strip()) for x in args.depths.split(",")]
    except ValueError as exc:
        logger.error("Invalid --depths value: %s", exc)
        sys.exit(1)

    requested_configs: List[str] = [c.strip() for c in args.configs.split(",")]
    invalid = [c for c in requested_configs if c not in _ALL_CONFIGS]
    if invalid:
        logger.error(
            "Unknown config(s): %s. Valid options: %s",
            invalid,
            _ALL_CONFIGS,
        )
        sys.exit(1)

    device: str = args.device

    # Validate device availability.
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU.")
        device = "cpu"

    # Ensure output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Load base HF model and tokenizer once; wrap per config.
    # ---------------------------------------------------------------------------
    if not _TRANSFORMERS_AVAILABLE:
        logger.error(
            "HuggingFace transformers is required for the NIAH benchmark.  "
            "Install with: pip install transformers"
        )
        sys.exit(1)

    try:
        hf_model, tokenizer = load_hf_model(args.model, device)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model '%s': %s", args.model, exc)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Build per-config wrappers.
    # ---------------------------------------------------------------------------
    wrappers: Dict[str, object] = {}

    if _CFG_BASELINE in requested_configs:
        if _BENCH_ACCURACY_AVAILABLE:
            wrappers[_CFG_BASELINE] = _HFBaselineWrapper(hf_model)
        else:
            # Minimal fallback wrapper with the same .model interface.
            class _FallbackWrapper:
                def __init__(self, m):
                    self.model = m
            wrappers[_CFG_BASELINE] = _FallbackWrapper(hf_model)

    if _CFG_STREAMING in requested_configs:
        if _BENCH_ACCURACY_AVAILABLE:
            wrappers[_CFG_STREAMING] = _HFStreamingLLMWrapper(
                hf_model=copy.deepcopy(hf_model),
                num_sink_tokens=args.num_sink_tokens,
                window_size=args.window_size,
            )
        else:
            class _FallbackWrapper:  # type: ignore[no-redef]
                def __init__(self, m):
                    self.model = m
            wrappers[_CFG_STREAMING] = _FallbackWrapper(copy.deepcopy(hf_model))

    if _CFG_MEMOPT in requested_configs:
        if _BENCH_ACCURACY_AVAILABLE and _MEMOPT_AVAILABLE:
            try:
                hf_for_memopt = copy.deepcopy(hf_model)
                hf_for_memopt = patch_gpt2_with_dynamic_attention(
                    model=hf_for_memopt,
                    window_size=args.window_size,
                    num_sink_tokens=args.num_sink_tokens,
                )
                wrappers[_CFG_MEMOPT] = _HFMemOptWrapper(hf_for_memopt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "patch_gpt2_with_dynamic_attention failed (%s) — "
                    "memopt_full will fall back to baseline wrapper.",
                    exc,
                )
                wrappers[_CFG_MEMOPT] = (
                    _HFBaselineWrapper(hf_model)
                    if _BENCH_ACCURACY_AVAILABLE
                    else type("_FB", (), {"model": hf_model})()
                )
        else:
            logger.warning(
                "MemOpt not available — memopt_full will use an unpatched baseline wrapper."
            )
            wrappers[_CFG_MEMOPT] = (
                _HFBaselineWrapper(hf_model)
                if _BENCH_ACCURACY_AVAILABLE
                else type("_FB", (), {"model": hf_model})()
            )

    # ---------------------------------------------------------------------------
    # Run evaluation for each requested config.
    # ---------------------------------------------------------------------------
    all_records: List[Dict] = []

    for config_name in requested_configs:
        wrapper = wrappers.get(config_name)
        if wrapper is None:
            logger.error("No wrapper found for config '%s'; skipping.", config_name)
            continue

        logger.info(
            "=== Evaluating config: %s | %d context_lengths x %d depths = %d trials ===",
            config_name,
            len(context_lengths),
            len(depths),
            len(context_lengths) * len(depths),
        )

        records = evaluate_configuration(
            config_name=config_name,
            wrapper=wrapper,
            tokenizer=tokenizer,
            context_lengths=context_lengths,
            depths=depths,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        all_records.extend(records)

    # ---------------------------------------------------------------------------
    # Assemble and save the results JSON.
    # ---------------------------------------------------------------------------
    timestamp: str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_filename: str = f"needle_haystack_{args.model}_{timestamp}.json"
    output_path: str = os.path.join(args.output_dir, output_filename)

    output: Dict = {
        "metadata": {
            "model": args.model,
            "needle": NEEDLE,
            "needle_answer": NEEDLE_ANSWER,
            "needle_question": NEEDLE_QUESTION,
            "timestamp": timestamp,
            "context_lengths": context_lengths,
            "depths": depths,
            "configs": requested_configs,
            "window_size": args.window_size,
            "num_sink_tokens": args.num_sink_tokens,
            "device": device,
            "max_new_tokens": args.max_new_tokens,
        },
        "results": all_records,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        logger.info("Results saved to: %s", os.path.abspath(output_path))
    except OSError as exc:
        logger.error("Failed to write results to '%s': %s", output_path, exc)
        sys.exit(1)

    # Print a brief summary.
    total_trials = len(all_records)
    retrievals = sum(r["accuracy"] for r in all_records if r["accuracy"] is not None)
    skipped = sum(1 for r in all_records if r["accuracy"] is None)
    print(
        f"\nNIAH benchmark complete.\n"
        f"  Total trials: {total_trials}\n"
        f"  Retrieved:    {retrievals}\n"
        f"  Skipped:      {skipped} (exceeded max_position_embeddings or error)\n"
        f"  Output:       {os.path.abspath(output_path)}"
    )


if __name__ == "__main__":
    main()
