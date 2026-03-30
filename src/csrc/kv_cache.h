// kv_cache.h
#pragma once
#include <torch/extension.h>
#include <utility>  // std::pair

namespace memopt {
namespace kv_cache {

/**
 * Initializes a paged KV cache memory pool.
 *
 * Allocates two GPU tensors (key_pool, val_pool) of shape
 * [num_blocks, block_size, num_heads, head_dim] in fp16 on the current CUDA
 * device, and seeds a CPU-side free-list with all block indices.
 *
 * Calling this function a second time with identical parameters is a no-op.
 * Calling with different parameters resets the pool.
 *
 * @param num_blocks  Total number of pages in the pool.
 * @param block_size  Number of token slots per page.
 * @param num_heads   Number of attention heads.
 * @param head_dim    Dimension of each attention head (must be even).
 */
void init_pool(int num_blocks, int block_size, int num_heads, int head_dim);

/**
 * Pops one free block index from the pool free-list.
 *
 * @return A block index in [0, num_blocks) on success, or -1 if the pool is
 *         exhausted (caller must evict or fail gracefully).
 */
int allocate_block();

/**
 * Returns a block index to the pool free-list, making it available for reuse.
 *
 * @param block_idx  Index previously obtained from allocate_block().
 */
void free_block(int block_idx);

/**
 * Returns the (key_pool, val_pool) GPU tensors.
 *
 * Tensors have shape [num_blocks, block_size, num_heads, head_dim] in fp16.
 * Intended for passing directly to attention kernels from Python.
 *
 * @return Pair (key_pool, val_pool).
 */
std::pair<torch::Tensor, torch::Tensor> get_pool_tensors();

/**
 * Returns the number of free blocks currently available in the pool.
 */
int free_blocks_count();

/**
 * Writes keys and values into the KV cache, mapping them to the correct blocks.
 *
 * Dispatches write_cache_kernel with grid=(num_tokens,), block=(256,).
 * Each CUDA block handles one token; threads stride over num_heads * head_dim.
 *
 * @param keys         Key tensor   (num_tokens, num_heads, head_dim) fp16/bf16.
 * @param values       Value tensor (num_tokens, num_heads, head_dim) fp16/bf16.
 * @param block_tables Block mapping per sequence (batch_size, max_blocks) int32.
 *                     Accepted for API symmetry; physical routing uses slot_mapping.
 * @param slot_mapping Flat physical slot per token (num_tokens,) int32.
 *                     slot = block_idx * block_size + position_within_block.
 */
void write_cache(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor block_tables,
    torch::Tensor slot_mapping);

/**
 * Evicts tokens dynamically based on Ada-KV / PyramidKV principles.
 *
 * For each (block, head) the kernel retains the top retain_ratio fraction of
 * tokens by attention score and zeroes the rest in the pool.
 *
 * @param block_tables      Block mappings (batch_size, max_blocks) int32.
 *                          Included for API symmetry; kernel operates pool-wide.
 * @param attention_scores  Per-token importance scores
 *                          (num_blocks, block_size, num_heads) float32.
 *                          Higher score = more important = retained.
 * @param retain_ratio      Fraction of tokens to keep per block/head, in (0, 1].
 */
void evict_cache(
    torch::Tensor block_tables,
    torch::Tensor attention_scores,
    float retain_ratio,
    int num_sink_tokens);

/**
 * Sequence-centric Ada-KV token eviction with explicit block_table management.
 *
 * Processes a batch of sequences described by their block_tables.  For each
 * (sequence, head) pair it finds the eviction threshold via warp-level
 * __shfl_xor_sync reductions, zeroes evicted token slots in the pool, and
 * atomically tracks per-block eviction votes.  When all token positions within
 * a physical block are evicted across all heads, the block is flagged in
 * is_block_free and unlinked from block_tables (set to -1).
 *
 * After this call the Python scheduler should scan is_block_free for values
 * equal to num_heads (fully evicted), call free_block() for each such block,
 * and reset is_block_free[block_idx] to 0.
 *
 * @param attention_scores  Importance scores (num_seqs, num_heads, seq_len) float32.
 *                          Higher score = more important = retained.
 * @param block_tables      Physical block indices (num_seqs, max_blocks_per_seq) int32.
 *                          Mutable: fully-evicted entries are set to -1.
 * @param is_block_free     Eviction vote accumulator (total_pool_blocks,) int32.
 *                          Must be initialised to 0 by the caller.
 *                          Entry == num_heads signals a fully-evicted block.
 * @param retain_ratio      Fraction of tokens per (seq, head) to retain, in (0, 1].
 */
void evict_tokens(
    torch::Tensor attention_scores,
    torch::Tensor block_tables,
    torch::Tensor is_block_free,
    float retain_ratio,
    int num_sink_tokens);

} // namespace kv_cache
} // namespace memopt
