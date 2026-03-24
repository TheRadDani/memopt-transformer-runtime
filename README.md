# memopt-transformer-runtime

MemOpt is a research-oriented system for memory-efficient deep learning inference, targeting one of the most critical bottlenecks in modern AI systems: RAM usage during inference.

While most optimization efforts focus on compute (e.g., faster attention kernels), real-world deployments of large models—especially transformers—are often memory-bound, with KV cache and intermediate activations dominating runtime memory consumption.

This project introduces a dynamic, runtime-aware memory optimization framework that reduces memory footprint without significantly degrading accuracy or latency.

🔹 KV Cache Optimization
Adaptive KV cache compression (INT8 / INT4)
Token importance-based pruning
Sliding window + long-term memory hybrid strategies
--
🔹 Activation Memory Optimization
Selective activation storage
Recomputation-aware execution
Memory vs compute tradeoff modeling
--
🔹 Dynamic Memory Scheduler
Runtime-aware policy engine
Adjusts:
compression level
pruning strategy
recomputation decisions
Adapts to:
sequence length
available memory
latency constraints
--
🔹 Custom CUDA Kernels (Planned / Implemented)
Memory-efficient KV packing/unpacking
Quantized tensor operations
Fused attention with compressed KV cache


Research Contributions
Dynamic KV cache compression strategies
Runtime-aware activation memory management
Hybrid memory policies combining:
quantization
pruning
recomputation
Exploration of memory–accuracy–latency tradeoffs
