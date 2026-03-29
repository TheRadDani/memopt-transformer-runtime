# MemOpt: Dynamic Memory Optimization for Transformer Inference

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)

## Overview

**MemOpt** is a research-grade runtime system that reduces VRAM footprint during Transformer inference by dynamically managing the KV cache and activation memory. It targets long-context inference under tight multi-tenant GPU memory budgets without severe perplexity degradation.

## Implemented Optimizations

1. **Token Pruning with Attention Sink Protection (Ada-KV / PyramidKV):** Attention-score-based eviction of low-importance KV tokens. The first `N` tokens (attention sinks) are structurally protected from eviction, aligning with the StreamingLLM mechanism to prevent perplexity degradation at long context.
2. **W4A16 KV Cache Quantization:** Group-wise symmetric INT4 compression of the KV cache, reducing its VRAM footprint by up to 75% with JIT dequantization before the query matmul.
3. **Paged KV Cache (Paged Attention):** Non-contiguous block-based memory allocation for the KV cache to eliminate fragmentation and support dynamic sizing.
4. **Sparse FlashAttention:** Custom attention forward pass that reads from paged block tables and skips pruned token slots.
5. **Runtime Memory Scheduler:** Background daemon thread polling `torch.cuda.memory_allocated()`. Above 85% VRAM utilization it increases eviction aggressiveness (`retain_ratio → 0.75`) and activates INT4 quantization; below 50% it restores full-retention FP16 mode.

## System Architecture

```
Control Plane (Python)                  Data Plane (CUDA C++)
─────────────────────────────           ──────────────────────────────
MemoryScheduler                         kv_cache.cu
  └─ polls VRAM every 50ms               ├─ Paged memory pool (init/alloc/free)
  └─ adjusts retain_ratio                ├─ write_cache kernel
  └─ toggles use_int4                    ├─ evict_tokens_kernel  ← sink-aware
                                         └─ evict_cache_kernel
DynamicAttention (nn.Module)
  └─ write_cache → paged pool           attention.cu
  └─ dynamic_sparse_attention            └─ Sparse FlashAttention (paged)
  └─ fallback: F.scaled_dot_product_attention
                                        quantization.cu
                                         ├─ quantize_kv_cache_int4
                                         └─ dequantize_kv_cache_int4
```

## Project Structure

```
memopt-transformer-runtime/
├── src/
│   ├── memopt/
│   │   ├── __init__.py       # Auto-loads C++ extension (memopt_C)
│   │   ├── attention.py      # DynamicAttention nn.Module
│   │   ├── scheduler.py      # MemoryScheduler + SchedulerConfig
│   │   └── models.py         # BaselineLlama transformer wrappers
│   └── csrc/
│       ├── extension.cpp     # Pybind11 bindings
│       ├── kv_cache.cu/.h    # Paged KV cache + Ada-KV eviction kernels
│       ├── attention.cu/.h   # Sparse FlashAttention w/ paged block tables
│       └── quantization.cu/.h# INT4 group-wise fused quant/dequant
├── scripts/
│   ├── bench_accuracy.py     # Perplexity on WikiText-2 / C4
│   ├── bench_latency_v2.py   # TTFT + TPOT latency benchmark
│   ├── bench_memory_v2.py    # VRAM breakdown benchmark
│   ├── run_ablations.py      # Full ablation study harness
│   └── plot_results.py       # NeurIPS/MLSys figure generator
├── tests/
│   ├── test_attention.py     # DynamicAttention correctness (torch.allclose)
│   ├── test_quant.py         # INT4 quant/dequant accuracy
│   └── test_import.py        # Extension load smoke test
├── setup.py                  # torch.utils.cpp_extension build
└── pyproject.toml
```

## Installation

Requires CUDA toolkit and PyTorch with CUDA support.

```bash
# Clone and install (compiles the C++/CUDA extension)
pip install -e .

# Verify the extension loaded correctly
python -c "import memopt_C; print(memopt_C.get_hello())"
```

## Unit Tests

```bash
pytest tests/
```

Tests verify mathematical equivalence of all kernels against PyTorch reference implementations to `1e-3` tolerance.

## Benchmarking Suite

### A. Perplexity / Accuracy

Evaluates sliding-window perplexity on WikiText-2 and/or C4 across context lengths, comparing Baseline vs StreamingLLM vs MemOpt:

```bash
# Full run (downloads dataset on first use)
python scripts/bench_accuracy.py --evaluate-dataset wikitext-2 --context-lengths 1024 4096 8192

# Comma-separated context lengths also accepted
python scripts/bench_accuracy.py --evaluate-dataset both --context-lengths 1024,4096,8192

# Quick smoke test (no network required)
python scripts/bench_accuracy.py --dry-run --no-plot

# Tune attention sink protection
python scripts/bench_accuracy.py --num-sink-tokens 4
```

Output: `scripts/bench_accuracy.json` + PNG figure.

---

### B. Latency (TTFT / TPOT)

Measures Time-To-First-Token (prefill) and Time-Per-Output-Token (decode) across batch sizes:

```bash
# Comma-separated or space-separated batch sizes both work
python scripts/bench_latency_v2.py --batch-size 1,4,16 --seq-len 8192

python scripts/bench_latency_v2.py --batch-sizes 1 4 16 --prompt-len 8192 --gen-len 64

# CPU smoke test
python scripts/bench_latency_v2.py --device cpu --batch-size 1 --seq-len 128 --gen-len 8 --warmup 1 --no-plot

# Model presets: tiny (default), medium
python scripts/bench_latency_v2.py --model medium --batch-size 1,4,16 --seq-len 4096
```

Output: `scripts/bench_latency_v2.csv` + PNG figure.

| Argument | Default | Description |
|---|---|---|
| `--batch-size` / `--batch-sizes` | `1 4 16 64` | Comma- or space-separated batch sizes |
| `--seq-len` / `--prompt-len` | `128` | Prompt length in tokens |
| `--gen-len` | `128` | Tokens to generate per decode phase |
| `--model` | `tiny` | Architecture preset: `tiny`, `medium` |

---

### C. VRAM Memory Breakdown

Profiles weight footprint, KV cache (analytical), and activation memory per sequence length:

```bash
# Default: tiny model, seq-lengths 512–8192
python scripts/bench_memory_v2.py

# llama-7b scale (auto fp16 on CUDA to fit in 16 GiB)
python scripts/bench_memory_v2.py --model llama-7b --seq-len 8192

# Specific sequence lengths
python scripts/bench_memory_v2.py --model medium --seq-len 512 1024 2048 4096

# Force float32 (2× more VRAM)
python scripts/bench_memory_v2.py --model medium --fp32

# Quick smoke test
python scripts/bench_memory_v2.py --dry-run --no-plot
```

Output: `scripts/bench_memory_v2.csv` + PNG figure.

| Argument | Default | Description |
|---|---|---|
| `--model` | `tiny` | `tiny`, `medium`, `llama-7b` |
| `--seq-len` / `--seq-lengths` | `512 1024 2048 4096 8192` | Sequence length(s) |
| `--fp32` | off | Disable auto fp16 cast on CUDA |
| `--dry-run` | off | Fast smoke test with seq-len 64, 128 |

> **Note:** Models are automatically cast to **float16** on CUDA to halve weight memory. Pass `--fp32` to disable. If estimated weight size exceeds 98% of free VRAM the benchmark skips that config with a warning rather than freezing the GPU.

---

### D. Ablation Study

Runs the full benchmark suite (memory + accuracy + latency) for four configurations — Full MemOpt, No Pruning, No Quantization, No Chunking — and aggregates results into a single JSON report:

```bash
python scripts/run_ablations.py --seq-len 8192 --output results/ablation_data.json

# Quick dry run (no dataset download, short sequences)
python scripts/run_ablations.py --dry-run --no-plot

# Single dataset
python scripts/run_ablations.py --dataset wikitext2 --seq-len 4096
```

Output: `results/ablation_data.json` + per-script CSVs in `results/`.

---

### E. Research Figures

Generates NeurIPS/MLSys-quality figures from benchmark outputs:

```bash
# Generate all figures (Pareto, memory breakdown, ablation, attention heatmap)
python scripts/plot_results.py --output-dir figures/

# Point to specific result files
python scripts/plot_results.py \
    --accuracy-json scripts/bench_accuracy.json \
    --memory-csv scripts/bench_memory_v2.csv \
    --ablation-json results/ablation_data.json \
    --output-dir figures/

# Generate only specific plots
python scripts/plot_results.py --no-pareto --no-heatmap --output-dir figures/
```

| Flag | Skips |
|---|---|
| `--no-pareto` | PPL vs VRAM Pareto curve |
| `--no-memory` | VRAM breakdown bar chart |
| `--no-ablation` | Ablation impact chart |
| `--no-heatmap` | Attention sparsity heatmap |

Output: PDF figures in `figures/`.

---

### Typical End-to-End Workflow

```bash
# 1. Build C++ extension
pip install -e .

# 2. Run unit tests
pytest tests/

# 3. Accuracy benchmark
python scripts/bench_accuracy.py --evaluate-dataset wikitext-2 --context-lengths 1024,4096,8192

# 4. Latency benchmark
python scripts/bench_latency_v2.py --batch-size 1,4,16 --seq-len 8192

# 5. Memory benchmark
python scripts/bench_memory_v2.py --model llama-7b --seq-len 8192

# 6. Full ablation study
python scripts/run_ablations.py --seq-len 8192 --output results/ablation_data.json

# 7. Generate figures
python scripts/plot_results.py \
    --accuracy-json scripts/bench_accuracy.json \
    --memory-csv scripts/bench_memory_v2.csv \
    --ablation-json results/ablation_data.json \
    --output-dir figures/
```

## Python API

```python
from memopt import DynamicAttention, MemoryScheduler, SchedulerConfig

# Drop-in attention module with paged KV cache
attn = DynamicAttention(embed_dim=512, num_heads=8, num_sink_tokens=4)

# VRAM-adaptive scheduler
scheduler = MemoryScheduler(SchedulerConfig(
    vram_high_threshold=0.85,   # activate eviction above 85% VRAM
    vram_low_threshold=0.50,    # restore full retention below 50%
    drop_retain_ratio=0.75,     # keep top 75% of KV tokens under pressure
    num_sink_tokens=4,          # always keep first 4 tokens
))
scheduler.start()

# Access live policy state
retain_ratio = scheduler.retain_ratio   # float in [0, 1]
use_int4    = scheduler.use_int4        # bool

scheduler.stop()
```

## Future Work

- **RL-based Scheduler:** Replace the heuristic `MemoryScheduler` with a PPO agent trained to preemptively optimize the context window distribution.
- **Speculative Decoding Integration:** Analyze memory overhead of draft models and mitigate via dynamic memory pooling.
- **Cross-Layer KV Offloading:** Asynchronous NVMe/CPU offloading for historically cold KV cache blocks.

## Citation

```bibtex
@software{memopt_2026,
  author = {Luis Daniel Ferreto Chavarria},
  title  = {MemOpt: Dynamic Memory Optimization for Transformer Inference},
  year   = {2026},
  url    = {https://github.com/TheRadDani/memopt-transformer-runtime}
}
```
