# MemOpt: Memory-Optimized Transformer Inference Runtime

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](https://github.com/TheRadDani/memopt-transformer-runtime)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-informational?style=flat-square)](pyproject.toml)

> Runtime for memory-efficient Transformer inference — paged KV cache, Ada-KV token pruning, W4A16 quantization, and Triton FlashAttention-2.

## Overview

Modern large language model inference faces two interlocking memory bottlenecks. The **KV cache** — the stored Key and Value tensors for every generated token — grows linearly with sequence length and batch size. At 100K+ context lengths it dwarfs model weights, directly capping the maximum batch size and usable context window on a fixed GPU. The **activation memory** from the attention forward pass scales quadratically with sequence length, causing out-of-memory failures on long documents even when the model itself fits.

MemOpt breaks both bottlenecks through a layered strategy. At the cache level, it allocates KV entries in non-contiguous 16-token pages (paged attention), eliminating fragmentation; prunes low-scoring tokens using Ada-KV attention-magnitude heuristics while protecting structural attention sinks; and compresses surviving cache entries to INT4 via group-wise symmetric W4A16 quantization with just-in-time dequantization before each query matmul. At the kernel level, it replaces the standard O(S²) attention with a paged sparse FlashAttention forward pass that skips pruned blocks entirely, keeping peak activation memory proportional to the window size rather than the full context length.

Coordinating these mechanisms is a runtime **MemoryScheduler** — a daemon thread that polls `torch.cuda.memory_allocated()` at 100 ms intervals and dynamically tightens or relaxes the eviction and quantization policy in response to live VRAM pressure. All of this is exposed through a `DynamicAttention` module that is drop-in compatible with `torch.nn.MultiheadAttention` and degrades gracefully to `F.scaled_dot_product_attention` when the compiled CUDA extension is unavailable. A new **Triton FlashAttention-2 backend** provides an alternative to the C++ kernel, implementing the same Ada-KV block-skipping semantics in pure Triton with hardware-efficient tiling and tensor-core acceleration.

## Architecture

```
Control Plane (Python — src/memopt/)          Data Plane (CUDA C++ — src/csrc/)
──────────────────────────────────────        ──────────────────────────────────────
MemoryScheduler                               kv_cache.cu / kv_cache.h
  ├─ background thread: polls VRAM              ├─ paged pool: init / alloc / free
  ├─ threshold 85%: retain_ratio→0.75,          ├─ write_cache kernel
  │  use_int4→True                              ├─ evict_tokens_kernel (sink-aware)
  └─ threshold 50%: restore FP16 mode           └─ evict_cache_kernel

DynamicAttention (nn.Module)                  attention.cu / attention.h
  ├─ Q/K/V projections                          └─ dynamic_sparse_attention
  ├─ _custom_kernel_forward                         (paged block tables, window mask)
  │    ├─ memopt_C path: write_cache
  │    │  → dynamic_sparse_attention           quantization.cu / quantization.h
  │    └─ triton path: triton_paged_             ├─ quantize_kv_cache_int4
  │         sparse_attention                    └─ dequantize_kv_cache_int4
  └─ _sdpa_forward (fallback, CPU/no ext)

triton_attention.py                           extension.cpp
  └─ _flash_attn_paged_sparse_kernel            └─ Pybind11 bindings (memopt_C)
       (FA-2 online softmax, Ada-KV gates,
        sink-token protection, tensor cores)
```

**Two-plane design.** The Python control plane enforces policy (which tokens to retain, when to quantize); the CUDA data plane executes the low-level memory operations. Policy variables (`retain_ratio`, `use_int4`) flow from the scheduler to the kernel at each forward call — the data plane never makes policy decisions.

## Features

- **Paged KV Cache** — KV entries are stored in 16-token non-contiguous physical blocks managed by a C++ allocator. Eliminates memory fragmentation and enables dynamic sizing without pre-allocating a contiguous buffer proportional to max sequence length.

- **Ada-KV Token Pruning** — At each attention step, tokens with low aggregate attention scores are candidates for eviction from the paged pool. A configurable `retain_ratio` controls the fraction kept. The first `num_sink_tokens` (default 4) are structurally protected, preventing the perplexity collapse that would otherwise occur from evicting initial tokens (attention sinks, per StreamingLLM arXiv 2309.17453).

- **W4A16 KV Cache Quantization** — KV cache entries are compressed from FP16 to INT4 using group-wise symmetric quantization with a group size of 64. The CUDA kernel stores 4-bit codes and per-group FP16 scales; dequantization back to FP16 happens just-in-time before the query matmul, keeping arithmetic in FP16 while cutting cache VRAM by up to 75%.

- **Sparse FlashAttention (C++ kernel)** — A custom `dynamic_sparse_attention` CUDA kernel reads from paged block tables and skips slots for pruned tokens. The attention matrix grows as O(S × W) where W is the window size, not O(S²), eliminating the quadratic activation spike for long sequences.

- **Triton FlashAttention-2 Backend (new)** — A pure-Triton implementation of paged sparse attention (`triton_paged_sparse_attention`) that shares the same physical KV pool with the C++ path. The kernel uses FA-2 online softmax (running max + denominator rescaling), skips KV blocks outside both the sliding window and the sink-token region, and uses `tl.dot` for tensor-core acceleration on Ampere+. Enable with `use_triton=True` on `DynamicAttention`; falls back gracefully if Triton is not installed.

- **AutoChunk Activation Chunking** — For extreme context lengths, activation dependencies are chunked temporally so the peak VRAM spike never exceeds the physical GPU memory limit.

- **Dynamic Memory Scheduler** — A daemon thread polls `torch.cuda.memory_allocated()` every 100 ms. Above 85% VRAM utilization it sets `retain_ratio = 0.75` and `use_int4 = True`; below 50% it restores `retain_ratio = 1.0` and `use_int4 = False`. A 35-point hysteresis dead band prevents oscillation. The scheduler can also attach as a `register_forward_pre_hook` for synchronous per-call adjustment.

- **Sliding-Window Attention with Sink Token Protection** — `DynamicAttention` accepts a `window_size` parameter that restricts visible KV positions to the most recent W tokens while unconditionally retaining the first `num_sink_tokens`. Both the C++ kernel path and the Triton kernel path implement this consistently. The SDPA fallback path achieves the same semantics by truncating the KV tensors rather than building an S×S mask.

- **Graceful Degradation** — Every kernel path has a fallback. C++ extension missing: falls back to SDPA. Triton not installed: falls back to SDPA-based paged reconstruction. No CUDA: CPU-compatible SDPA with sliding-window truncation. The same model code runs in unit-test environments without GPU or compiled extension.

## Installation

**Prerequisites:** CUDA Toolkit 11.8+, Python 3.12+, PyTorch 2.0+ with CUDA support.

```bash
git clone https://github.com/TheRadDani/memopt-transformer-runtime.git
cd memopt-transformer-runtime

# Install the package and compile the C++/CUDA extension
pip install -e .

# Verify the extension loaded correctly
python -c "import memopt; print(memopt.get_hello())"
```

**Optional: Triton backend**

```bash
pip install triton
```

The Triton backend (`use_triton=True`) provides an alternative to the compiled C++ kernel. It requires a CUDA GPU and Triton >= 2.0.

**Development dependencies** (formatters, linters):

```bash
pip install -e ".[dev]"   # installs black, isort, pytest
```

## Quick Start

### DynamicAttention — C++ backend (default)

```python
import torch
from memopt import DynamicAttention

# Drop-in replacement for torch.nn.MultiheadAttention
attn = DynamicAttention(
    embed_dim=512,
    num_heads=8,
    window_size=2048,       # sliding window; -1 = full context
    num_sink_tokens=4,      # protect first 4 tokens from eviction
    use_triton=False,       # use compiled C++ kernel (default)
)

x = torch.randn(2, 1024, 512)  # (batch=2, seq=1024, dim=512)
out = attn(x)                   # (2, 1024, 512)
```

### DynamicAttention — Triton FlashAttention-2 backend (new)

```python
from memopt import DynamicAttention

attn = DynamicAttention(
    embed_dim=512,
    num_heads=8,
    window_size=2048,
    num_sink_tokens=4,
    use_triton=True,        # Triton FA-2 kernel; falls back to SDPA if unavailable
)
```

When paged-cache tensors (`block_tables`, `slot_mapping`, `context_lens`) are not provided, both backends fall back to standard SDPA automatically.

### MemoryScheduler

```python
from memopt import DynamicAttention, MemoryScheduler, SchedulerConfig
import torch.nn as nn

config = SchedulerConfig(
    poll_interval_sec=0.1,      # poll VRAM every 100 ms
    vram_high_threshold=0.85,   # trigger eviction above 85% VRAM
    vram_low_threshold=0.50,    # restore FP16 below 50% VRAM
    drop_retain_ratio=0.75,     # retain top 75% of KV tokens under pressure
    num_sink_tokens=4,
)
scheduler = MemoryScheduler(config)
scheduler.start()

# Optionally attach as a forward hook for synchronous per-call adjustment
model = nn.Module()  # your transformer model
scheduler.attach_hook(model)

# Read live policy state in your forward wrapper
retain_ratio = scheduler.retain_ratio   # float in [0.0, 1.0]
use_int4     = scheduler.use_int4       # bool

scheduler.stop()
```

## Benchmarks

All scripts write CSV output to `results/` by default and accept `--dry-run` for a quick sanity check that requires no dataset downloads and completes in under 60 seconds on CPU.

### Latency — `bench_latency_v2.py`

Measures Time-To-First-Token (TTFT, full prefill) and Time-Per-Output-Token (TPOT, decode loop) separately, across configurable batch sizes. Compares `nn.MultiheadAttention` baseline against `DynamicAttention + MemoryScheduler`.

```bash
# Quick smoke test — batch=[1,2], gen_len=10
python scripts/bench_latency_v2.py --dry-run

# GPU run, default batch sizes [1, 4, 16, 64]
python scripts/bench_latency_v2.py --device cuda

# Custom batch sizes and prompt length
python scripts/bench_latency_v2.py \
    --batch-sizes 1 4 16 64 \
    --prompt-len 128 \
    --gen-len 200 \
    --model medium \
    --no-plot
```

| Argument | Default | Description |
|---|---|---|
| `--batch-sizes` | `1 4 16 64` | Space- or comma-separated batch sizes |
| `--prompt-len` | `128` | Prompt length in tokens |
| `--gen-len` | `200` | Tokens to generate per decode phase |
| `--model` | `tiny` | Architecture preset: `tiny` (d=256, h=4, L=4) or `medium` (d=512, h=8, L=6) |
| `--warmup` | `1` | Warmup passes per batch size before timing |
| `--no-plot` | off | Skip matplotlib figure |

Output: timestamped CSV (`results/latency_v2_YYYYMMDD_HHMMSS.csv`) + optional PNG.

### Memory — `bench_memory_v2.py`

Profiles VRAM broken down into four categories (weights, KV cache analytical estimate, activations, reserved) across sequence lengths, for both baseline and MemOpt models.

```bash
# Default: tiny model, seq_lengths=[512, 1024, 2048, 4096, 8192]
python scripts/bench_memory_v2.py

# llama-7b scale (auto fp16 on CUDA)
python scripts/bench_memory_v2.py --model llama-7b --seq-lengths 4096 8192

# Custom sequence lengths
python scripts/bench_memory_v2.py --model medium --seq-lengths 512 1024 2048

# Force float32 (2x weight VRAM)
python scripts/bench_memory_v2.py --fp32

# Smoke test, CPU only
python scripts/bench_memory_v2.py --dry-run --no-plot --device cpu
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `tiny` | `tiny`, `medium`, or `llama-7b` |
| `--seq-lengths` | `512 1024 2048 4096 8192` | Sequence lengths to profile |
| `--fp32` | off | Disable auto fp16 cast on CUDA |
| `--dry-run` | off | Fast smoke test with seq_len=[64, 128] |

Output: timestamped CSV + optional PNG (two-subplot: active VRAM and VRAM breakdown).

### Accuracy / Perplexity — `bench_accuracy.py`

Sliding-window perplexity evaluation on WikiText-2 and/or C4, comparing three configurations across context lengths. Supports HuggingFace pretrained models (GPT-2, etc.) and a self-contained tiny decoder for offline smoke-testing.

```bash
# Smoke test — no network, tiny model
python scripts/bench_accuracy.py --dry-run --no-plot --device cpu

# GPT-2 on WikiText-2
python scripts/bench_accuracy.py --model-name gpt2 --evaluate-dataset wikitext2

# Full run: both datasets, multiple context lengths
python scripts/bench_accuracy.py \
    --model-name gpt2 \
    --evaluate-dataset both \
    --context-lengths 1024 4096 8192

# Tune sink token protection
python scripts/bench_accuracy.py --num-sink-tokens 4
```

Output: `scripts/bench_accuracy.json` + PNG figures.

### Needle-In-A-Haystack — `bench_needle_haystack.py` (new)

Evaluates long-context retrieval by embedding a "needle" fact at a configurable fractional depth inside a haystack context, then prompting the model to recall it. Tests all three configurations across a grid of context lengths and needle depths.

```bash
# Quick smoke test (GPT-2, short contexts)
python scripts/bench_needle_haystack.py \
    --context-lens 128,256 \
    --depths 0.0,0.5,1.0

# Default run (GPT-2, context <= 1024)
python scripts/bench_needle_haystack.py

# Select specific configurations
python scripts/bench_needle_haystack.py --configs baseline,streaming_llm
```

Output: JSON results file consumed by `plot_results.py` for heatmap rendering.

### Ablation Suite — `run_ablations.py`

Orchestrates memory, latency, and accuracy benchmarks across four ablation configurations and aggregates results into a single JSON report.

| Config | Ada-KV | W4A16 | AutoChunk |
|--------|--------|-------|-----------|
| A — Full MemOpt | ON | ON | ON |
| B — No Pruning | OFF | ON | ON |
| C — No Quantization | ON | OFF | ON |
| D — No Chunking | ON | ON | OFF |

```bash
# Smoke test (seq_len=128)
python scripts/run_ablations.py --dry-run

# Standard run at seq_len=8192
python scripts/run_ablations.py --seq-len 8192 --output results/ablation_data.json
```

Output: `results/ablation_data.json` + per-script CSVs.

### Publication Figures — `plot_results.py`

Generates four NeurIPS/MLSys-quality PDF figures from benchmark outputs in `results/`.

| Figure | Output file | Description |
|--------|-------------|-------------|
| Pareto Curve | `figures/pareto_curve.pdf` | Peak VRAM vs. perplexity Pareto frontier |
| Memory Breakdown | `figures/memory_breakdown.pdf` | Stacked bars: weights / KV cache / activations |
| Ablation Impact | `figures/ablation_impact.pdf` | 3-panel grouped bars: VRAM, PPL, TPOT per config |
| Attention Sparsity | `figures/attention_sparsity.pdf` | Ada-KV causal attention heatmap |

```bash
# Generate all figures
python scripts/plot_results.py \
    --accuracy-json scripts/bench_accuracy.json \
    --ablation-json results/ablation_data.json \
    --results-dir results/ \
    --output-dir figures/

# Generate specific figures only
python scripts/plot_results.py --no-pareto --no-heatmap --output-dir figures/
```

| Skip Flag | Omits |
|-----------|-------|
| `--no-pareto` | Pareto curve |
| `--no-memory` | VRAM breakdown stacked bar |
| `--no-ablation` | Ablation impact panels |
| `--no-heatmap` | Attention sparsity heatmap |

### End-to-End Workflow

```bash
# 1. Build C++ extension
pip install -e .

# 2. Run unit tests
pytest tests/

# 3. Accuracy benchmark
python scripts/bench_accuracy.py \
    --model-name gpt2 \
    --evaluate-dataset wikitext2 \
    --context-lengths 1024 4096 8192

# 4. Latency benchmark
python scripts/bench_latency_v2.py --batch-sizes 1 4 16 64 --prompt-len 128

# 5. Memory benchmark
python scripts/bench_memory_v2.py --model medium --seq-lengths 512 1024 2048 4096 8192

# 6. NIAH benchmark
python scripts/bench_needle_haystack.py

# 7. Full ablation study
python scripts/run_ablations.py --seq-len 8192 --output results/ablation_data.json

# 8. Generate publication figures
python scripts/plot_results.py \
    --accuracy-json scripts/bench_accuracy.json \
    --ablation-json results/ablation_data.json \
    --results-dir results/ \
    --output-dir figures/
```

## Model Configurations

Three configurations are defined across `bench_accuracy.py`, `bench_needle_haystack.py`, and the ablation suite:

| Configuration | Attention | KV Eviction | Quantization | Scheduler |
|---------------|-----------|-------------|--------------|-----------|
| `baseline` | Standard causal SDPA | None | None (FP16) | None |
| `streaming_llm` | Sink + sliding-window mask | Implicit (mask only) | None (FP16) | None |
| `memopt_full` | DynamicAttention (paged + sparse) | Ada-KV (retain_ratio adaptive) | W4A16 INT4 | MemoryScheduler active |

`streaming_llm` implements the attention-sink masking from arXiv 2309.17453 as a reference baseline for perplexity comparison.

## Testing

```bash
pytest tests/
```

The test suite requires a compiled `memopt_C` extension and a CUDA-capable GPU. Tests are skipped automatically when either dependency is absent.

| Test file | Coverage |
|-----------|----------|
| `test_import.py` | Package import, C++ extension load, version assertion |
| `test_attention.py` | `memopt_C.dynamic_sparse_attention` vs `F.scaled_dot_product_attention` (full-context, no eviction) at `atol=1e-3` |
| `test_quant.py` | `quantize_kv_cache_int4` / `dequantize_kv_cache_int4` round-trip: shape/dtype contracts, scale correctness (`atol=1e-4`), MSE threshold, edge cases (zeros, saturated values, single group), determinism, q-code equivalence (±1 tolerance), and error guards |

Mathematical equivalence for the Triton kernel is verified by the built-in smoke test in `src/memopt/triton_attention.py`:

```bash
python src/memopt/triton_attention.py
# Smoke test passed.
```

## Project Structure

```
memopt-transformer-runtime/
├── src/
│   ├── memopt/                     # Python control plane
│   │   ├── __init__.py             # Package init; exports DynamicAttention,
│   │   │                           #   MemoryScheduler, SchedulerConfig
│   │   ├── attention.py            # DynamicAttention nn.Module (C++ + Triton paths)
│   │   ├── triton_attention.py     # Triton FA-2 kernel (new)
│   │   ├── scheduler.py            # MemoryScheduler + SchedulerConfig
│   │   └── models.py               # BaselineLlama transformer wrappers
│   └── csrc/                       # CUDA data plane
│       ├── extension.cpp           # Pybind11 bindings → memopt_C module
│       ├── kv_cache.cu / .h        # Paged pool, write_cache, evict kernels
│       ├── attention.cu / .h       # dynamic_sparse_attention (paged + sparse)
│       └── quantization.cu / .h   # quantize/dequantize_kv_cache_int4
├── scripts/
│   ├── bench_latency_v2.py         # TTFT + TPOT latency benchmark
│   ├── bench_memory_v2.py          # VRAM breakdown by category
│   ├── bench_accuracy.py           # Perplexity on WikiText-2 / C4
│   ├── bench_needle_haystack.py    # NIAH long-context retrieval benchmark
│   ├── run_ablations.py            # Ablation harness (configs A–D)
│   └── plot_results.py             # NeurIPS/MLSys figure generator
├── tests/
│   ├── test_import.py              # Import and extension load
│   ├── test_attention.py           # Kernel correctness vs SDPA reference
│   └── test_quant.py               # INT4 quantization round-trip
├── results/                        # Benchmark CSV and JSON outputs
├── figures/                        # Generated PDF figures
├── pyproject.toml                  # Build config, dependencies, black settings
└── setup.py                        # torch.utils.cpp_extension build
```

## Research Context

MemOpt is an MLSys/NeurIPS-grade research artifact implementing 2024 literature techniques for memory-efficient Transformer inference:

- **Ada-KV / PyramidKV** — attention-score-based adaptive KV cache compression with per-layer budget allocation
- **Paged Attention** (vLLM, 2023) — non-contiguous block-based KV cache management
- **FlashAttention-2** (Dao, 2023) — online softmax with rescaled accumulators, hardware-efficient tiling
- **StreamingLLM** (arXiv 2309.17453) — attention-sink protection for unbounded generation
- **W4A16 quantization** — group-wise INT4 KV cache compression with FP16 activations
- **AutoChunk** — temporal chunking of activation dependencies to bound peak VRAM

The Triton backend in particular demonstrates that all of these mechanisms — paged addressing, Ada-KV block skipping, sink-token protection, and FA-2 online softmax — can be composed in a single hardware-portable kernel without a compiled C++ extension.

## Citation

```bibtex
@software{memopt_2026,
  author = {Luis Daniel Ferreto Chavarria},
  title  = {MemOpt: Memory-Optimized Transformer Inference Runtime},
  year   = {2026},
  url    = {https://github.com/TheRadDani/memopt-transformer-runtime}
}
```

## Future Work

- **RL-based Scheduler** — Replace the heuristic `MemoryScheduler` with a PPO agent trained to preemptively optimize the context window distribution based on observed attention patterns.
- **Speculative Decoding Integration** — Analyze draft-model memory overhead and mitigate via dynamic memory pooling shared between draft and target model KV caches.
- **Cross-Layer KV Offloading** — Asynchronous NVMe/CPU offloading for historically cold KV cache blocks, with prefetch-on-demand triggered by block access patterns.
- **Multi-GPU Paged Attention** — Extend the block allocator to span multiple devices, enabling context lengths that exceed single-GPU memory.
