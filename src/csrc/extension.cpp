#include <torch/extension.h>
#include "kv_cache.h"
#include "attention.h"
#include "quantization.h"

std::string get_hello() {
    return "MemOpt C++/CUDA Backend Initialized.";
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_hello", &get_hello, "Initial handshake function for MemOpt C++ extension");

    // -----------------------------------------------------------------------
    // KV Cache — Paged Memory Pool
    // -----------------------------------------------------------------------

    // init_pool(num_blocks, block_size, num_heads, head_dim)
    //   Allocates two GPU tensors of shape [num_blocks, block_size, num_heads,
    //   head_dim] (one for K, one for V) in fp16.  Seeds the CPU-side free-list.
    m.def("init_pool",
          &memopt::kv_cache::init_pool,
          "Allocate the paged KV cache pool tensors on GPU and seed the free-list",
          pybind11::arg("num_blocks"),
          pybind11::arg("block_size"),
          pybind11::arg("num_heads"),
          pybind11::arg("head_dim"));

    // allocate_block() -> int
    //   Pops one free block index from the pool free-list.
    //   Returns -1 if the pool is exhausted.
    m.def("allocate_block",
          &memopt::kv_cache::allocate_block,
          "Pop one free block index from the pool free-list; returns -1 if exhausted");

    // free_block(block_idx: int)
    //   Returns a block index to the free-list (marks it available for reuse).
    m.def("free_block",
          &memopt::kv_cache::free_block,
          "Return a block index to the pool free-list",
          pybind11::arg("block_idx"));

    // get_pool_tensors() -> (Tensor, Tensor)
    //   Returns (key_pool, val_pool) tensors so Python can pass them directly
    //   to attention kernels without going through the C++ pool state.
    m.def("get_pool_tensors",
          &memopt::kv_cache::get_pool_tensors,
          "Return (key_pool, val_pool) GPU tensors of shape [num_blocks, block_size, num_heads, head_dim]");

    // free_blocks_count() -> int
    //   Returns the number of currently available (free) blocks.
    m.def("free_blocks_count",
          &memopt::kv_cache::free_blocks_count,
          "Return the number of free blocks remaining in the pool");

    // write_cache(keys, values, block_tables, slot_mapping)
    //   Scatters (num_tokens, num_heads, head_dim) key/value tensors into the
    //   paged pool according to the flat physical slot_mapping [num_tokens].
    //   slot = block_idx * block_size + position_within_block.
    m.def("write_cache",
          &memopt::kv_cache::write_cache,
          "Scatter key/value tokens into the paged KV cache pool via slot_mapping",
          pybind11::arg("keys"),
          pybind11::arg("values"),
          pybind11::arg("block_tables"),
          pybind11::arg("slot_mapping"));

    // evict_cache(block_tables, attention_scores, retain_ratio)
    //   Ada-KV eviction: zeros low-importance token slots in the pool based on
    //   per-token attention scores and a configurable retain budget.
    m.def("evict_cache",
          &memopt::kv_cache::evict_cache,
          "Zero evicted token slots in the pool based on attention score ranking",
          pybind11::arg("block_tables"),
          pybind11::arg("attention_scores"),
          pybind11::arg("retain_ratio"),
          pybind11::arg("num_sink_tokens") = 4);

    // evict_tokens(attention_scores, block_tables, is_block_free, retain_ratio)
    //   Sequence-centric Ada-KV eviction.  Operates on a batch of sequences
    //   described by block_tables[num_seqs, max_blocks_per_seq].  Uses warp-level
    //   __shfl_xor_sync binary-search threshold selection.  Fully-evicted blocks
    //   are flagged in is_block_free[total_pool_blocks] and unlinked from
    //   block_tables (set to -1).  Caller must call free_block() for each block
    //   where is_block_free[b] == num_heads.
    m.def("evict_tokens",
          &memopt::kv_cache::evict_tokens,
          "Sequence-centric Ada-KV token eviction with block_table unlinking",
          pybind11::arg("attention_scores"),
          pybind11::arg("block_tables"),
          pybind11::arg("is_block_free"),
          pybind11::arg("retain_ratio"),
          pybind11::arg("num_sink_tokens") = 4);

    // -----------------------------------------------------------------------
    // Attention
    // -----------------------------------------------------------------------
    m.def("dynamic_sparse_attention",
          &memopt::attention::dynamic_sparse_attention,
          "Dynamic sparse/chunked attention over paged KV cache");

    // -----------------------------------------------------------------------
    // Quantization
    // -----------------------------------------------------------------------
    // quantize_kv_cache_int4(cache, group_size) -> List[Tensor]
    //   Quantizes an FP16 cache tensor to packed INT4 (group-wise symmetric).
    //   Returns a Python list of two tensors:
    //     [0] packed : UInt8  tensor, last dim halved (2 INT4 per byte)
    //     [1] scales : float32 tensor, shape [total_elements / group_size]
    m.def("quantize_kv_cache_int4",
          &memopt::quantization::quantize_kv_cache_int4,
          "Quantize KV cache to packed INT4 (group-wise symmetric); "
          "returns [packed_uint8, scales_float32]",
          pybind11::arg("cache"),
          pybind11::arg("group_size"));

    // dequantize_kv_cache_int4(q_cache, scales, group_size) -> Tensor (FP16)
    //   Decompresses a packed INT4 cache back to FP16.
    m.def("dequantize_kv_cache_int4",
          &memopt::quantization::dequantize_kv_cache_int4,
          "Dequantize packed INT4 KV cache to FP16",
          pybind11::arg("q_cache"),
          pybind11::arg("scales"),
          pybind11::arg("group_size"));
}
