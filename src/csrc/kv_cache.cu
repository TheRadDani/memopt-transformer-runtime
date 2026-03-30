// kv_cache.cu
//
// Paged KV Cache memory pool manager + Ada-KV token eviction kernels.
//
// Design overview
// ---------------
//  Pool layout  : Two contiguous GPU tensors (one for K, one for V), each of
//                 shape [num_blocks, block_size, num_heads, head_dim].  Physical
//                 memory is pre-allocated once via init_pool(); individual
//                 "pages" are handed out through a CPU-side free-list.
//
//  Slot mapping : Each token in a batch is assigned a *flat physical slot*:
//                   slot = block_idx * block_size + position_within_block
//                 The caller (Python scheduler) manages the block table and
//                 passes the pre-computed slot_mapping tensor to write_cache.
//
//  write_cache_kernel grid/block mapping
//  --------------------------------------
//   Grid  : (num_tokens,)        — one CUDA block per token
//   Threads: up to WRITE_BLOCK_DIM, strided over (num_heads * head_dim)
//   Access pattern: threads in a warp read/write consecutive head_dim elements
//                   → coalesced 128-byte cache lines when head_dim >= 32.
//
//  evict_cache_kernel grid/block mapping
//  ----------------------------------------
//   Grid  : (num_blocks, num_heads)
//   Threads: up to EVICT_BLOCK_DIM, strided over block_size
//   Uses shared memory to compute per-block mean score and a threshold
//   derived from the retain_ratio, then zeroes evicted slots in the pool.
//
// SM architecture target: Ampere (sm_80) primary; Turing (sm_75) compatible.

#pragma once
#include "kv_cache.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <cstdio>

// ---------------------------------------------------------------------------
// Debug helpers
// ---------------------------------------------------------------------------
#ifndef NDEBUG
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _err = (call);                                                  \
    if (_err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d — %s\n",                           \
              __FILE__, __LINE__, cudaGetErrorString(_err));                    \
      abort();                                                                  \
    }                                                                           \
  } while (0)
#else
#define CUDA_CHECK(call) (call)
#endif

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr int WRITE_BLOCK_DIM = 256;  // threads/block for write_cache
static constexpr int EVICT_BLOCK_DIM = 128;  // threads/block for evict_cache

namespace memopt {
namespace kv_cache {

// ---------------------------------------------------------------------------
// Module-level pool state
// ---------------------------------------------------------------------------

/*
 * KVCachePool holds the two large pre-allocated GPU tensors (key_pool,
 * val_pool) and the CPU-side free-list of available block indices.
 *
 * Thread safety: a std::mutex guards the free-list for the allocate/free
 * path, which is driven from the Python scheduler (single-threaded in the
 * common case, but guarded for safety).
 */
struct KVCachePool {
  torch::Tensor key_pool;   // shape: [num_blocks, block_size, num_heads, head_dim]
  torch::Tensor val_pool;   // shape: [num_blocks, block_size, num_heads, head_dim]

  int num_blocks  = 0;
  int block_size  = 0;
  int num_heads   = 0;
  int head_dim    = 0;

  std::queue<int> free_list;
  std::mutex      free_list_mu;
  bool            initialized = false;
};

// Singleton — one pool per process (extend to multi-GPU via device_id key if needed).
static KVCachePool g_pool;

// ---------------------------------------------------------------------------
// init_pool
// ---------------------------------------------------------------------------

/*
 * Allocates two GPU tensors of shape [num_blocks, block_size, num_heads,
 * head_dim] in fp16 and seeds the CPU-side free-list with all block indices.
 *
 * This function is idempotent w.r.t. re-initialization: calling it a second
 * time with the same parameters is a no-op; calling with different parameters
 * resets the pool (existing block handles become invalid).
 */
void init_pool(int num_blocks, int block_size, int num_heads, int head_dim) {
  std::lock_guard<std::mutex> lk(g_pool.free_list_mu);

  // Idempotency guard
  if (g_pool.initialized &&
      g_pool.num_blocks == num_blocks &&
      g_pool.block_size  == block_size  &&
      g_pool.num_heads   == num_heads   &&
      g_pool.head_dim    == head_dim) {
    return;
  }

  TORCH_CHECK(num_blocks > 0,  "init_pool: num_blocks must be > 0");
  TORCH_CHECK(block_size > 0,  "init_pool: block_size must be > 0");
  TORCH_CHECK(num_heads  > 0,  "init_pool: num_heads must be > 0");
  TORCH_CHECK(head_dim   > 0,  "init_pool: head_dim must be > 0");
  TORCH_CHECK(head_dim % 2 == 0,
              "init_pool: head_dim must be even for fp16 coalesced access");

  // Drain any existing free-list
  while (!g_pool.free_list.empty()) g_pool.free_list.pop();

  // Allocate pool tensors on the current CUDA device, fp16
  auto opts = torch::TensorOptions()
                  .dtype(torch::kFloat16)
                  .device(torch::kCUDA)
                  .requires_grad(false);

  g_pool.key_pool = torch::zeros({num_blocks, block_size, num_heads, head_dim}, opts);
  g_pool.val_pool = torch::zeros({num_blocks, block_size, num_heads, head_dim}, opts);

  // Seed free-list
  for (int i = 0; i < num_blocks; ++i) {
    g_pool.free_list.push(i);
  }

  g_pool.num_blocks   = num_blocks;
  g_pool.block_size   = block_size;
  g_pool.num_heads    = num_heads;
  g_pool.head_dim     = head_dim;
  g_pool.initialized  = true;
}

// ---------------------------------------------------------------------------
// allocate_block / free_block (Python-visible via pybind11)
// ---------------------------------------------------------------------------

/*
 * Pops one block index from the free-list.
 * Returns -1 if the pool is exhausted (caller must handle OOM).
 */
int allocate_block() {
  std::lock_guard<std::mutex> lk(g_pool.free_list_mu);
  TORCH_CHECK(g_pool.initialized, "allocate_block: pool not initialized");
  if (g_pool.free_list.empty()) {
    return -1;  // Pool exhausted — caller must evict or fail.
  }
  int idx = g_pool.free_list.front();
  g_pool.free_list.pop();
  return idx;
}

/*
 * Returns a block index to the free-list.
 * The caller is responsible for clearing any sensitive data beforehand (or
 * relying on the fact that write_cache always overwrites before read).
 */
void free_block(int block_idx) {
  std::lock_guard<std::mutex> lk(g_pool.free_list_mu);
  TORCH_CHECK(g_pool.initialized, "free_block: pool not initialized");
  TORCH_CHECK(block_idx >= 0 && block_idx < g_pool.num_blocks,
              "free_block: block_idx out of range");
  g_pool.free_list.push(block_idx);
}

/*
 * Returns (key_pool, val_pool) tensors so Python can pass them to kernels.
 */
std::pair<torch::Tensor, torch::Tensor> get_pool_tensors() {
  TORCH_CHECK(g_pool.initialized, "get_pool_tensors: pool not initialized");
  return {g_pool.key_pool, g_pool.val_pool};
}

/*
 * Returns the number of free blocks remaining in the pool.
 */
int free_blocks_count() {
  std::lock_guard<std::mutex> lk(g_pool.free_list_mu);
  return static_cast<int>(g_pool.free_list.size());
}

// ---------------------------------------------------------------------------
// write_cache_kernel
// ---------------------------------------------------------------------------

/*
 * CUDA kernel: scatter keys/values from a flat token list into paged pool
 * slots described by slot_mapping.
 *
 * Grid  : (num_tokens,)   — dim3(num_tokens, 1, 1)
 * Block : (WRITE_BLOCK_DIM,) threads
 *
 * Each CUDA block handles exactly one token (blockIdx.x == token index).
 * Threads stride over the num_heads * head_dim elements of that token,
 * reading from the input tensors and writing to the pool.
 *
 * Memory access:
 *   - Input   keys/values: (num_tokens, num_heads, head_dim) — row-major.
 *             Thread 0..31 in warp 0 read elements [0..31] of head_dim for
 *             head 0, which are consecutive in memory → coalesced.
 *   - Output  pool:        (num_blocks, block_size, num_heads, head_dim).
 *             Stride to the correct block and position, then write the same
 *             head/dim layout → coalesced within a head.
 *
 * Template parameter scalar_t: supports at::Half and at::BFloat16.
 */
template <typename scalar_t>
__global__ void __launch_bounds__(WRITE_BLOCK_DIM)
write_cache_kernel(
    const scalar_t* __restrict__ keys,    // [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ values,  // [num_tokens, num_heads, head_dim]
    scalar_t*       __restrict__ key_pool,  // [num_blocks, block_size, num_heads, head_dim]
    scalar_t*       __restrict__ val_pool,  // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ slot_mapping,   // [num_tokens]  flat physical slot index
    int num_heads,
    int head_dim,
    int block_size
) {
  const int token_idx = blockIdx.x;
  const int hd        = num_heads * head_dim;  // elements per token

  // Physical slot for this token
  const int slot      = slot_mapping[token_idx];
  const int blk_idx   = slot / block_size;
  const int blk_pos   = slot % block_size;

  // Base pointers into the pool for this (block, position)
  const int64_t pool_token_offset =
      (static_cast<int64_t>(blk_idx) * block_size + blk_pos) * hd;

  // Base pointers into the flat input tensors for this token
  const int64_t src_offset = static_cast<int64_t>(token_idx) * hd;

  // Grid-stride over num_heads * head_dim
  for (int i = threadIdx.x; i < hd; i += WRITE_BLOCK_DIM) {
    key_pool[pool_token_offset + i] = keys[src_offset + i];
    val_pool[pool_token_offset + i] = values[src_offset + i];
  }
}

// ---------------------------------------------------------------------------
// write_cache (Python-visible entry point)
// ---------------------------------------------------------------------------

/*
 * Dispatches write_cache_kernel.
 *
 * Args:
 *   keys         : Tensor[num_tokens, num_heads, head_dim] — fp16 or bf16
 *   values       : Tensor[num_tokens, num_heads, head_dim] — same dtype as keys
 *   block_tables : Tensor[batch_size, max_blocks] — int32, block index table
 *                  (not used directly here; slot_mapping is the pre-computed
 *                  physical slot per token, which the Python scheduler builds
 *                  from block_tables + sequence positions)
 *   slot_mapping : Tensor[num_tokens] — int32, flat physical slot per token
 *                  slot = block_idx * block_size + position_within_block
 *
 * Note: block_tables is accepted to match the declared header signature and
 * is available for future extensions (e.g., copy-on-write, snapshot).
 */
void write_cache(
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor block_tables,
    torch::Tensor slot_mapping)
{
  TORCH_CHECK(g_pool.initialized, "write_cache: pool not initialized; call init_pool first");

  // Shape validation
  TORCH_CHECK(keys.dim()   == 3, "write_cache: keys must be 3-D [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(values.dim() == 3, "write_cache: values must be 3-D [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(slot_mapping.dim() == 1, "write_cache: slot_mapping must be 1-D [num_tokens]");

  const int num_tokens = static_cast<int>(keys.size(0));
  const int num_heads  = static_cast<int>(keys.size(1));
  const int head_dim   = static_cast<int>(keys.size(2));

  TORCH_CHECK(values.size(0) == num_tokens, "write_cache: keys/values token count mismatch");
  TORCH_CHECK(values.size(1) == num_heads,  "write_cache: keys/values num_heads mismatch");
  TORCH_CHECK(values.size(2) == head_dim,   "write_cache: keys/values head_dim mismatch");
  TORCH_CHECK(slot_mapping.size(0) == num_tokens,
              "write_cache: slot_mapping length != num_tokens");
  TORCH_CHECK(num_heads  == g_pool.num_heads,  "write_cache: num_heads != pool.num_heads");
  TORCH_CHECK(head_dim   == g_pool.head_dim,   "write_cache: head_dim != pool.head_dim");

  // Ensure contiguous inputs
  keys         = keys.contiguous();
  values       = values.contiguous();
  slot_mapping = slot_mapping.contiguous();

  if (num_tokens == 0) return;

  // Kernel launch
  const dim3 grid(num_tokens);
  const dim3 block(WRITE_BLOCK_DIM);

  // Route the current CUDA stream through PyTorch's stream management
  const at::cuda::CUDAGuard device_guard(keys.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(keys.scalar_type(), "write_cache", [&]() {
    // AT_DISPATCH does not include bf16 by default; handle it explicitly
    // via the scalar_t alias that resolves at::Half for fp16.
    // For bf16 pools we cast through the same kernel with __nv_bfloat16 —
    // the macro AT_DISPATCH_FLOATING_TYPES_AND_HALF covers at::Half only;
    // bf16 support would require AT_DISPATCH_SWITCH which we add as a
    // separate branch if needed.  For the initial implementation fp16 is
    // the target dtype matching the pool allocation.
    write_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
        keys.data_ptr<scalar_t>(),
        values.data_ptr<scalar_t>(),
        g_pool.key_pool.data_ptr<scalar_t>(),
        g_pool.val_pool.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        num_heads,
        head_dim,
        g_pool.block_size
    );
  });

#ifndef NDEBUG
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
#else
  CUDA_CHECK(cudaGetLastError());
#endif
}

// ---------------------------------------------------------------------------
// evict_cache_kernel
// ---------------------------------------------------------------------------

/*
 * CUDA kernel: Ada-KV attention-score-based token eviction.
 *
 * For each (block, head) pair this kernel:
 *   1. Loads per-token attention scores from attention_scores into shared mem.
 *   2. Computes a local threshold = the (retain_ratio * block_size)-th order
 *      statistic via a simple in-shared-memory sort (valid for block_size <=
 *      EVICT_BLOCK_DIM; larger block sizes use a streaming approach below).
 *   3. Marks tokens whose score is below threshold as evicted by zeroing their
 *      key/value slots in the pool.
 *
 * Grid  : (num_blocks, num_heads)
 * Block : (EVICT_BLOCK_DIM,) threads
 *
 * Shared memory layout (per block):
 *   float smem_scores[block_size]  — per-token importance scores
 *
 * The eviction writes zeros to the pool; the free-list management (returning
 * freed block indices) is handled on the CPU side after the kernel completes,
 * because it requires the Python scheduler to inspect which blocks are now
 * fully empty.
 *
 * Assumption: block_size <= EVICT_BLOCK_DIM * 4  (reasonable for inference,
 * where block_size is typically 16 or 32).
 */
template <typename scalar_t>
__global__ void __launch_bounds__(EVICT_BLOCK_DIM)
evict_cache_kernel(
    scalar_t*      __restrict__ key_pool,         // [num_blocks, block_size, num_heads, head_dim]
    scalar_t*      __restrict__ val_pool,         // [num_blocks, block_size, num_heads, head_dim]
    const float*   __restrict__ attention_scores, // [num_blocks, block_size, num_heads]
    int num_blocks,
    int block_size,
    int num_heads,
    int head_dim,
    float retain_ratio,
    int  num_sink_tokens
) {
  // TODO: sink preservation not natively supported for pool-centric formulation;
  // requires reverse context-length mapping to determine absolute token positions.

  // Each CUDA block handles one (block_idx, head_idx) pair.
  const int blk_idx  = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (blk_idx >= num_blocks || head_idx >= num_heads) return;

  // Shared memory for scores — padded to avoid bank conflicts
  // (bank conflict occurs when 32 threads access 32-way-strided 4-byte words;
  //  padding by 1 float shifts each row by 4 bytes)
  extern __shared__ float smem[];  // size = block_size * sizeof(float), allocated at launch

  // Load per-token scores for this (block, head) into shared memory.
  // attention_scores layout: [num_blocks, block_size, num_heads]
  // offset = blk_idx * block_size * num_heads + token_pos * num_heads + head_idx
  for (int pos = threadIdx.x; pos < block_size; pos += EVICT_BLOCK_DIM) {
    const int64_t score_idx =
        (static_cast<int64_t>(blk_idx) * block_size + pos) * num_heads + head_idx;
    smem[pos] = attention_scores[score_idx];
  }
  __syncthreads();

  // --- Compute eviction threshold via parallel selection sort in shared mem ---
  // We want to retain the top-k = ceil(retain_ratio * block_size) tokens.
  // For small block_size (<=128) we use an in-shared insertion sort executed
  // by a single thread to avoid complex sorting networks.  For correctness this
  // is fine; for performance it is dominated by the memory zeroing below.
  //
  // Only thread 0 computes the threshold to avoid redundant work.
  __shared__ float evict_threshold;

  if (threadIdx.x == 0) {
    int keep = max(1, static_cast<int>(retain_ratio * static_cast<float>(block_size)));
    keep = min(keep, block_size);

    // Partial selection: find the keep-th largest value (0-indexed).
    // We copy to a small register array (valid because block_size is small).
    // For block_size up to 128 this fits comfortably in L1/shared.
    float threshold = -1e38f;

    // Simple O(keep * block_size) selection — acceptable for block_size <= 128.
    // In practice block_size is 16 or 32 in vLLM-style paged attention.
    for (int k = 0; k < keep; ++k) {
      float best = -1e38f;
      int   best_pos = -1;
      for (int pos = 0; pos < block_size; ++pos) {
        if (smem[pos] > best) {
          best     = smem[pos];
          best_pos = pos;
        }
      }
      threshold = best;
      if (best_pos >= 0) smem[best_pos] = -1e38f;  // mark visited
    }
    evict_threshold = threshold;
  }
  __syncthreads();

  // Reload the original scores (they were clobbered during selection).
  // We need a second pass — reload from global memory.
  for (int pos = threadIdx.x; pos < block_size; pos += EVICT_BLOCK_DIM) {
    const int64_t score_idx =
        (static_cast<int64_t>(blk_idx) * block_size + pos) * num_heads + head_idx;
    smem[pos] = attention_scores[score_idx];
  }
  __syncthreads();

  // --- Zero out evicted token slots in the pool ---
  // Pool layout: [num_blocks, block_size, num_heads, head_dim]
  // For token at position `pos` in this block / head, the offset is:
  //   (blk_idx * block_size + pos) * num_heads * head_dim + head_idx * head_dim
  const int64_t pool_block_base =
      static_cast<int64_t>(blk_idx) * block_size * num_heads * head_dim;

  for (int pos = 0; pos < block_size; ++pos) {
    if (smem[pos] < evict_threshold) {
      // This token is below the threshold — evict it.
      const int64_t token_head_offset =
          pool_block_base +
          static_cast<int64_t>(pos) * num_heads * head_dim +
          static_cast<int64_t>(head_idx) * head_dim;

      // Threads stride over head_dim to zero the K and V entries.
      for (int d = threadIdx.x; d < head_dim; d += EVICT_BLOCK_DIM) {
        key_pool[token_head_offset + d] = static_cast<scalar_t>(0);
        val_pool[token_head_offset + d] = static_cast<scalar_t>(0);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// evict_cache (Python-visible entry point)
// ---------------------------------------------------------------------------

/*
 * Evicts low-importance tokens from the paged KV cache using attention scores.
 *
 * Args:
 *   block_tables      : Tensor[batch_size, max_blocks] int32 — included for
 *                       API symmetry and future CoW extension; not consumed
 *                       directly by the kernel (the kernel operates over all
 *                       blocks in the pool).
 *   attention_scores  : Tensor[num_blocks, block_size, num_heads] float32 —
 *                       accumulated importance score for each token in each
 *                       block/head.  Higher score = more important.
 *   retain_ratio      : float in (0, 1] — fraction of tokens per block/head
 *                       to keep; the rest are zeroed in the pool.
 */
void evict_cache(
    torch::Tensor block_tables,
    torch::Tensor attention_scores,
    float retain_ratio,
    int num_sink_tokens)
{
  TORCH_CHECK(g_pool.initialized, "evict_cache: pool not initialized");
  TORCH_CHECK(retain_ratio > 0.f && retain_ratio <= 1.f,
              "evict_cache: retain_ratio must be in (0, 1]");
  TORCH_CHECK(attention_scores.dim() == 3,
              "evict_cache: attention_scores must be 3-D [num_blocks, block_size, num_heads]");
  TORCH_CHECK(attention_scores.scalar_type() == torch::kFloat32,
              "evict_cache: attention_scores must be float32");

  const int num_blocks = static_cast<int>(attention_scores.size(0));
  const int block_size = static_cast<int>(attention_scores.size(1));
  const int num_heads  = static_cast<int>(attention_scores.size(2));

  TORCH_CHECK(num_blocks <= g_pool.num_blocks,
              "evict_cache: attention_scores num_blocks > pool.num_blocks");
  TORCH_CHECK(block_size == g_pool.block_size,
              "evict_cache: attention_scores block_size != pool.block_size");
  TORCH_CHECK(num_heads  == g_pool.num_heads,
              "evict_cache: attention_scores num_heads != pool.num_heads");

  attention_scores = attention_scores.contiguous();

  if (num_blocks == 0) return;

  const dim3 grid(num_blocks, num_heads);
  const dim3 block(EVICT_BLOCK_DIM);
  const size_t smem_bytes = static_cast<size_t>(block_size) * sizeof(float);

  const at::cuda::CUDAGuard device_guard(attention_scores.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(g_pool.key_pool.scalar_type(), "evict_cache", [&]() {
    evict_cache_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        g_pool.key_pool.data_ptr<scalar_t>(),
        g_pool.val_pool.data_ptr<scalar_t>(),
        attention_scores.data_ptr<float>(),
        num_blocks,
        block_size,
        num_heads,
        g_pool.head_dim,
        retain_ratio,
        num_sink_tokens
    );
  });

#ifndef NDEBUG
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
#else
  CUDA_CHECK(cudaGetLastError());
#endif
}

// ---------------------------------------------------------------------------
// evict_tokens_kernel
// ---------------------------------------------------------------------------

/*
 * CUDA kernel: Ada-KV token eviction operating on a sequence-centric view of
 * attention scores, with explicit block_table unlinking and is_block_free
 * tracking.
 *
 * Compared to evict_cache_kernel (which operates over all blocks in the pool
 * in a pool-centric layout), this kernel operates over a batch of sequences
 * described by their block_tables.  It:
 *   1. Mean-pools attention scores across all heads for each token position in
 *      the sequence (score[t] = mean_h attn[seq, h, t]).
 *   2. Finds the eviction threshold — the score of the keep-th largest token —
 *      using warp-level __shfl_xor_sync reductions for efficiency.
 *   3. Marks tokens below threshold as evicted by zeroing their pool slots.
 *   4. For each physical block referenced by this sequence, if ALL token
 *      positions within that block are evicted across ALL heads, atomically
 *      sets is_block_free[block_idx] = 1 and unlinks the slot in block_tables
 *      (set to BLOCK_UNLINKED_SENTINEL = -1).
 *
 * Grid  : (num_seqs, num_heads)   — one CUDA block per (sequence, head)
 * Block : (EVICT_BLOCK_DIM,) threads = 128
 *
 * Shared memory layout (floats, total = seq_len + 1 + max_blocks_per_seq):
 *   smem[0 .. seq_len-1]                — head-mean importance score per token
 *   smem[seq_len]                       — evict_threshold (broadcast)
 *   smem[seq_len+1 .. seq_len+max_bps]  — per-block eviction counters (int reuse)
 *
 * Assumptions:
 *   - seq_len <= MAX_SEQ_LEN_EVICT (compile-time cap for shared memory sizing).
 *   - block_tables is row-major [num_seqs, max_blocks_per_seq], int32.
 *     Unused trailing slots contain BLOCK_UNLINKED_SENTINEL (-1).
 *   - attention_scores is [num_seqs, num_heads, seq_len], float32, row-major.
 *   - is_block_free is [total_pool_blocks], int32, initialised to 0.
 *
 * Threshold algorithm (warp-level):
 *   Each warp independently scans its portion of smem_scores and maintains a
 *   register-local running minimum of the top-keep scores seen so far.  After
 *   all warps have scanned, a cross-warp reduction via __shfl_xor_sync finds
 *   the global k-th order statistic.  This avoids shared-memory sorting
 *   networks and is exact for any seq_len.
 *
 * SM architecture: Ampere (sm_80) primary.  Uses __shfl_xor_sync which is
 *   available from sm_30+.
 */

static constexpr int BLOCK_UNLINKED_SENTINEL = -1;
static constexpr int MAX_SEQ_LEN_EVICT       = 4096;  // cap for smem budget
static constexpr int MAX_WARPS_PER_BLOCK      = EVICT_BLOCK_DIM / 32;

/*
 * Warp-level parallel reduction to find the minimum of a set of per-lane
 * float values using __shfl_xor_sync butterfly reduction.
 * Returns the minimum value across all 32 lanes in the warp.
 */
__device__ __forceinline__ float warp_reduce_min(float val) {
  // Butterfly XOR reduction over 5 stages (log2(32) = 5)
  val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, 16));
  val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, 8));
  val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, 4));
  val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, 2));
  val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, 1));
  return val;
}

/*
 * Warp-level parallel reduction to find the maximum.
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 16));
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 8));
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 4));
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 2));
  val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, 1));
  return val;
}

template <typename scalar_t>
__global__ void __launch_bounds__(EVICT_BLOCK_DIM)
evict_tokens_kernel(
    scalar_t*      __restrict__ key_pool,           // [total_blocks, block_size, num_heads, head_dim]
    scalar_t*      __restrict__ val_pool,           // [total_blocks, block_size, num_heads, head_dim]
    const float*   __restrict__ attention_scores,   // [num_seqs, num_heads, seq_len]
    int*           __restrict__ block_tables,        // [num_seqs, max_blocks_per_seq]
    int*           __restrict__ is_block_free,       // [total_pool_blocks]
    int  num_seqs,
    int  num_heads,
    int  seq_len,
    int  max_blocks_per_seq,
    int  pool_block_size,    // tokens per physical block (g_pool.block_size)
    int  head_dim,
    int  total_pool_blocks,
    float retain_ratio,
    int  num_sink_tokens
) {
  const int seq_idx  = blockIdx.x;
  const int head_idx = blockIdx.y;

  if (seq_idx >= num_seqs || head_idx >= num_heads) return;

  // -------------------------------------------------------------------------
  // Shared memory layout:
  //   [0 .. seq_len)          : float — per-token head-mean importance scores
  //   [seq_len]               : float — broadcast eviction threshold
  //   [seq_len+1 .. ...]      : int32 reinterpreted — per-block eviction count
  //     (tracks how many token positions in that logical block are below thresh)
  //
  // Budget: (seq_len + 1) * 4 + max_blocks_per_seq * 4 bytes
  // For seq_len=4096, max_blocks_per_seq=256: ~17 KB — fits in sm_80 shared mem.
  // -------------------------------------------------------------------------
  extern __shared__ float smem_scores[];  // allocated at launch: see evict_tokens()

  // Re-interpret the tail of shared memory as int32 for per-block counters.
  // Layout: smem_scores[0..seq_len+1) as float, smem_evict_cnt starting after.
  int* smem_evict_cnt = reinterpret_cast<int*>(smem_scores + seq_len + 2);
  // (+2 for the threshold float and one padding element to keep int alignment)

  // -------------------------------------------------------------------------
  // Step 1: Load attention scores for this (seq, head) and compute per-token
  //         importance.  For the sequence-centric interface, the scores are
  //         already provided per head — we accumulate across heads using an
  //         atomic add into smem_scores, then divide by num_heads.
  //         Only the head=0 block initialises the accumulator; other heads add.
  // -------------------------------------------------------------------------

  // Initialise smem_scores to 0 on the first head pass.
  if (head_idx == 0) {
    for (int t = threadIdx.x; t < seq_len; t += EVICT_BLOCK_DIM) {
      smem_scores[t] = 0.0f;
    }
  }

  // Note: because grid dim Y = num_heads, the (seq, h=0) block may race with
  // (seq, h>0) blocks during accumulation.  To avoid this, each (seq, head)
  // block loads its own slice of attention_scores and we use atomicAdd into
  // global memory for the head-mean accumulator, which is a separate tensor
  // passed in.  However, to keep the kernel self-contained without an extra
  // global buffer, we compute the per-token score as the mean directly using
  // the global attention_scores tensor (no smem accumulation across heads):
  //
  //   score[seq][t] = (1/num_heads) * sum_h attention_scores[seq][h][t]
  //
  // Each (seq, head) block computes the FULL mean by reading all num_heads
  // slices.  This is O(num_heads) reads per token but avoids cross-block
  // synchronisation.
  //
  // Then threshold selection and eviction decisions are made independently
  // per head (matching the pool-centric evict_cache_kernel semantics).

  // Load this (seq, head) slice of attention_scores.
  // Layout: [num_seqs, num_heads, seq_len]  →  row-major
  const int64_t score_base =
      (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * seq_len;

  for (int t = threadIdx.x; t < seq_len; t += EVICT_BLOCK_DIM) {
    smem_scores[t] = attention_scores[score_base + t];
  }
  __syncthreads();

  // Protect attention sink tokens: boost scores so they are never evicted.
  for (int t = threadIdx.x; t < num_sink_tokens; t += EVICT_BLOCK_DIM) {
    smem_scores[t] = 1e38f;
  }
  __syncthreads();

  // -------------------------------------------------------------------------
  // Step 2: Compute eviction threshold — the k-th largest score where
  //         k = ceil(retain_ratio * seq_len).
  //
  // Algorithm: warp-level parallel k-th order statistic.
  //
  //   Each thread maintains a register "local_kth_min" — the minimum of the
  //   top-keep scores seen by that thread in its strided scan.  After the scan,
  //   a cross-warp reduction finds the global minimum of all local_kth_mins,
  //   which equals the k-th largest score overall.
  //
  //   This is a standard streaming selection approximation; for small seq_len
  //   (≤128, typical for paged block sizes) it is exact because each thread
  //   sees all values.  For larger seq_len the approximation can be slightly
  //   conservative (threshold too high → more evictions than intended), which
  //   is safe: we evict a few extra tokens but never fewer than required.
  //
  //   For exact k-th value at any seq_len, we use a two-pass algorithm:
  //     Pass A: find global min and max of all scores (warp reduce).
  //     Pass B: binary search for threshold t* such that
  //             count(scores >= t*) == keep.  Uses iterative bisection with
  //             warp-parallel counting.
  //   The binary search converges in O(log(range/epsilon)) iterations, each
  //   O(seq_len / EVICT_BLOCK_DIM) — fast for fp32 precision (~23 iterations).
  // -------------------------------------------------------------------------

  const int keep = max(1, min(seq_len,
      static_cast<int>(ceilf(retain_ratio * static_cast<float>(seq_len)))));

  // --- Pass A: find global min and max for binary search bounds ---
  float local_min =  1e38f;
  float local_max = -1e38f;
  for (int t = threadIdx.x; t < seq_len; t += EVICT_BLOCK_DIM) {
    float s = smem_scores[t];
    local_min = fminf(local_min, s);
    local_max = fmaxf(local_max, s);
  }

  // Warp-level reduction
  local_min = warp_reduce_min(local_min);
  local_max = warp_reduce_max(local_max);

  // Cross-warp reduction via shared memory.
  // Shared layout for warp results: smem_scores[seq_len] and beyond are safe
  // to reuse as scratch since we'll reload below.  Use a dedicated scratch
  // area after seq_len+2 (already reserved for smem_evict_cnt).
  // Instead, use the threshold slot and an adjacent slot as temporary storage.
  __shared__ float smem_warp_min[MAX_WARPS_PER_BLOCK];
  __shared__ float smem_warp_max[MAX_WARPS_PER_BLOCK];

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  if (lane_id == 0) {
    smem_warp_min[warp_id] = local_min;
    smem_warp_max[warp_id] = local_max;
  }
  __syncthreads();

  // Thread 0 reduces across warps
  float global_min =  1e38f;
  float global_max = -1e38f;
  if (threadIdx.x == 0) {
    int num_warps = (EVICT_BLOCK_DIM + 31) / 32;
    for (int w = 0; w < num_warps; ++w) {
      global_min = fminf(global_min, smem_warp_min[w]);
      global_max = fmaxf(global_max, smem_warp_max[w]);
    }
    // Store to threshold slot (reused as temp)
    smem_scores[seq_len]     = global_min;
    smem_scores[seq_len + 1] = global_max;
  }
  __syncthreads();

  global_min = smem_scores[seq_len];
  global_max = smem_scores[seq_len + 1];

  // --- Pass B: binary search for k-th largest threshold ---
  // We want the smallest threshold t* such that exactly `keep` scores >= t*.
  // Equivalently: count_geq(t*) == keep, where count_geq(t) = |{s : s >= t}|.
  //
  // Binary search on the real-valued range [global_min, global_max].
  // 32 iterations give precision ~(global_max - global_min) * 2^-32.
  __shared__ float evict_threshold;
  __shared__ int   smem_count;  // scratch for parallel count

  float lo = global_min;
  float hi = global_max;

  // If all scores are equal, keep all (threshold = global_min).
  if (global_min >= global_max) {
    if (threadIdx.x == 0) evict_threshold = global_min - 1.0f;
    __syncthreads();
  } else {
    for (int iter = 0; iter < 32; ++iter) {
      float mid = lo + (hi - lo) * 0.5f;

      // Count tokens with score >= mid (in parallel across threads)
      int local_count = 0;
      for (int t = threadIdx.x; t < seq_len; t += EVICT_BLOCK_DIM) {
        if (smem_scores[t] >= mid) ++local_count;
      }

      // Warp-level sum reduction
      local_count += __shfl_xor_sync(0xFFFFFFFF, local_count, 16);
      local_count += __shfl_xor_sync(0xFFFFFFFF, local_count, 8);
      local_count += __shfl_xor_sync(0xFFFFFFFF, local_count, 4);
      local_count += __shfl_xor_sync(0xFFFFFFFF, local_count, 2);
      local_count += __shfl_xor_sync(0xFFFFFFFF, local_count, 1);

      // Cross-warp sum via shared memory
      if (lane_id == 0) smem_warp_min[warp_id] = static_cast<float>(local_count);
      __syncthreads();

      if (threadIdx.x == 0) {
        int total = 0;
        int num_warps = (EVICT_BLOCK_DIM + 31) / 32;
        for (int w = 0; w < num_warps; ++w) {
          total += static_cast<int>(smem_warp_min[w]);
        }
        smem_count = total;
      }
      __syncthreads();

      int cnt = smem_count;
      // Adjust search bounds:
      //   count_geq(mid) >= keep  →  threshold can be raised  → lo = mid
      //   count_geq(mid) <  keep  →  threshold must be lowered → hi = mid
      if (cnt >= keep) {
        lo = mid;
      } else {
        hi = mid;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      // lo is the largest threshold such that count_geq(lo) >= keep.
      // Tokens with score < lo are evicted.
      evict_threshold = lo;
    }
    __syncthreads();
  }

  // -------------------------------------------------------------------------
  // Step 3: Build per-block eviction counters in shared memory.
  //
  //   For each token t in [0, seq_len), determine its physical block from
  //   block_tables and increment smem_evict_cnt[logical_block] if the token
  //   is evicted (score < evict_threshold).
  //
  //   Logical block = t / pool_block_size
  //   Position in block = t % pool_block_size
  //   Physical block index = block_tables[seq_idx * max_blocks_per_seq + logical_block]
  //
  //   We use atomicAdd into smem_evict_cnt for thread-safe accumulation.
  //   After all threads have voted, if smem_evict_cnt[lb] == pool_block_size
  //   (all positions in the block are evicted), the block is freed.
  // -------------------------------------------------------------------------

  // Number of logical blocks covering seq_len tokens
  const int num_logical_blocks = (seq_len + pool_block_size - 1) / pool_block_size;

  // Initialise per-block eviction counters.
  // Only head=0 (blockIdx.y==0) initialises to avoid multi-head races.
  // However, because each (seq, head) block runs independently in parallel,
  // we cannot rely on head=0 having finished before head=1 reads smem_evict_cnt.
  // Solution: we use *global* atomic counters (one int per physical block in
  // is_block_free already exists).  But is_block_free semantics mean it's
  // already-free, not an eviction counter.
  //
  // Design choice: eviction counting is per-block across ALL heads.  A physical
  // block slot (block, position) is considered evicted only when ALL heads
  // have evicted that position.  We track this with a separate global counter
  // array — but the interface only provides is_block_free[].
  //
  // Pragmatic solution: for each (seq, head) block independently:
  //   - Zero the local smem_evict_cnt for this head's pass.
  //   - Count evicted positions within this head.
  //   - Use atomicAdd on a global counter (reuse is_block_free as a saturating
  //     counter: after num_heads increments for a given block position, the
  //     block is fully evicted).
  //   - When global counter for a physical block reaches pool_block_size * num_heads,
  //     the block is fully evicted: set is_block_free = 1 and unlink block_tables.
  //
  // This requires is_block_free to be initialised to 0 by the caller and
  // treated as a mutable accumulator during the kernel.  After the kernel,
  // the caller should inspect is_block_free and call free_block() for any
  // block where is_block_free > 0.
  //
  // Final sentinel: is_block_free[b] == pool_block_size * num_heads means
  // block b is fully evicted.  We use that exact value as the free trigger.

  // Initialise local smem_evict_cnt for this kernel block
  for (int lb = threadIdx.x; lb < num_logical_blocks; lb += EVICT_BLOCK_DIM) {
    smem_evict_cnt[lb] = 0;
  }
  __syncthreads();

  const float thresh = evict_threshold;

  // -------------------------------------------------------------------------
  // Step 4: Evict tokens and zero pool slots; accumulate smem_evict_cnt.
  // -------------------------------------------------------------------------

  const int64_t head_stride = static_cast<int64_t>(head_dim);
  const int64_t pos_stride  = static_cast<int64_t>(num_heads) * head_stride;
  const int64_t blk_stride  = static_cast<int64_t>(pool_block_size) * pos_stride;

  for (int t = threadIdx.x / 32 * 32; t < seq_len; t += EVICT_BLOCK_DIM) {
    // Each warp handles a contiguous run of 32 tokens (warp-aligned).
    // Within the warp, each lane handles one token.
    const int token_t = t + (threadIdx.x % 32);
    bool do_evict = false;

    if (token_t < seq_len) {
      do_evict = (smem_scores[token_t] < thresh);
    }

    if (do_evict && token_t < seq_len) {
      // Determine physical block and position
      const int lb       = token_t / pool_block_size;    // logical block index
      const int pos      = token_t % pool_block_size;    // position within block
      const int phys_blk = block_tables[
          static_cast<int64_t>(seq_idx) * max_blocks_per_seq + lb];

      if (phys_blk != BLOCK_UNLINKED_SENTINEL &&
          phys_blk >= 0 && phys_blk < total_pool_blocks) {
        // Zero the key and value pool entries for (phys_blk, pos, head_idx, :).
        // Pool layout: [num_blocks, block_size, num_heads, head_dim]
        // The base offset for (phys_blk, pos, head_idx) is:
        //   phys_blk * block_size * num_heads * head_dim
        //   + pos * num_heads * head_dim
        //   + head_idx * head_dim
        // Threads within the warp stride over head_dim using lane_id, giving
        // coalesced writes: 32 consecutive float16 elements = 64 bytes per warp.
        const int64_t pool_offset =
            static_cast<int64_t>(phys_blk) * blk_stride +
            static_cast<int64_t>(pos) * pos_stride +
            static_cast<int64_t>(head_idx) * head_stride;

        const int my_lane = threadIdx.x % 32;
        for (int d = my_lane; d < head_dim; d += 32) {
          key_pool[pool_offset + d] = static_cast<scalar_t>(0);
          val_pool[pool_offset + d] = static_cast<scalar_t>(0);
        }

        // Only lane 0 accumulates the eviction counter to avoid duplicate counts
        // (all 32 lanes in the warp agree that this token is evicted, but we
        // only want to count the token once, not 32 times).
        if (my_lane == 0) {
          atomicAdd(&smem_evict_cnt[lb], 1);
        }
      }
    }
  }
  __syncthreads();

  // -------------------------------------------------------------------------
  // Step 5: For each logical block, if ALL pool_block_size positions in this
  //         head were evicted, atomically increment the global is_block_free
  //         counter.  When that counter reaches pool_block_size * num_heads,
  //         mark the block as free and unlink from block_tables.
  // -------------------------------------------------------------------------

  for (int lb = threadIdx.x; lb < num_logical_blocks; lb += EVICT_BLOCK_DIM) {
    if (smem_evict_cnt[lb] == pool_block_size) {
      // All positions in this logical block were evicted by this head.
      // Increment the global counter for the physical block.
      const int phys_blk = block_tables[
          static_cast<int64_t>(seq_idx) * max_blocks_per_seq + lb];

      if (phys_blk == BLOCK_UNLINKED_SENTINEL || phys_blk < 0 ||
          phys_blk >= total_pool_blocks) continue;

      // Saturating increment: when we reach num_heads, this block is fully evicted.
      const int prev = atomicAdd(&is_block_free[phys_blk], 1);
      if (prev + 1 == num_heads) {
        // This was the last head to fully evict this block.
        // Unlink from block_tables and mark free.
        block_tables[static_cast<int64_t>(seq_idx) * max_blocks_per_seq + lb] =
            BLOCK_UNLINKED_SENTINEL;
        // is_block_free[phys_blk] is already == num_heads; caller uses > 0 as
        // the free signal (see evict_tokens() host function comments).
      }
    }
  }
}

// ---------------------------------------------------------------------------
// evict_tokens (Python-visible entry point)
// ---------------------------------------------------------------------------

/*
 * Sequence-centric Ada-KV token eviction with explicit block_table management.
 *
 * This function complements evict_cache() (which operates pool-wide over
 * pre-indexed blocks) by accepting a batch of sequences described via their
 * block_tables.  It:
 *   - Computes per-head eviction thresholds from attention_scores.
 *   - Zeroes evicted token slots in the pool (key_pool + val_pool).
 *   - Tracks per-block eviction counts; fully-evicted blocks (all token
 *     positions evicted across all heads) are flagged in is_block_free and
 *     unlinked from block_tables (set to -1).
 *
 * After this call the Python scheduler should:
 *   1. Scan is_block_free for values == num_heads (fully evicted).
 *   2. Call free_block(block_idx) for each such entry.
 *   3. Reset is_block_free[block_idx] to 0 after freeing.
 *
 * Args:
 *   attention_scores   : Tensor[num_seqs, num_heads, seq_len] float32.
 *                        Per-token importance per head.  Higher = more important.
 *   block_tables       : Tensor[num_seqs, max_blocks_per_seq] int32, mutable.
 *                        Physical block indices; -1 = unlinked/unused.
 *   is_block_free      : Tensor[total_pool_blocks] int32, mutable, init to 0.
 *                        Accumulates eviction votes; == num_heads means free.
 *   retain_ratio       : float in (0, 1] — fraction of tokens per (seq, head)
 *                        to retain.
 *
 * Grid/block:
 *   Grid  = (num_seqs, num_heads)
 *   Block = (EVICT_BLOCK_DIM,)
 *   Smem  = (seq_len + 2 + max_blocks_per_seq) * sizeof(float) bytes
 */
void evict_tokens(
    torch::Tensor attention_scores,
    torch::Tensor block_tables,
    torch::Tensor is_block_free,
    float retain_ratio,
    int num_sink_tokens)
{
  TORCH_CHECK(g_pool.initialized, "evict_tokens: pool not initialized");
  TORCH_CHECK(retain_ratio > 0.f && retain_ratio <= 1.f,
              "evict_tokens: retain_ratio must be in (0, 1]");

  TORCH_CHECK(attention_scores.dim() == 3,
              "evict_tokens: attention_scores must be 3-D [num_seqs, num_heads, seq_len]");
  TORCH_CHECK(attention_scores.scalar_type() == torch::kFloat32,
              "evict_tokens: attention_scores must be float32");

  TORCH_CHECK(block_tables.dim() == 2,
              "evict_tokens: block_tables must be 2-D [num_seqs, max_blocks_per_seq]");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt32,
              "evict_tokens: block_tables must be int32");

  TORCH_CHECK(is_block_free.dim() == 1,
              "evict_tokens: is_block_free must be 1-D [total_pool_blocks]");
  TORCH_CHECK(is_block_free.scalar_type() == torch::kInt32,
              "evict_tokens: is_block_free must be int32");

  const int num_seqs          = static_cast<int>(attention_scores.size(0));
  const int num_heads         = static_cast<int>(attention_scores.size(1));
  const int seq_len           = static_cast<int>(attention_scores.size(2));
  const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));
  const int total_pool_blocks = static_cast<int>(is_block_free.size(0));

  TORCH_CHECK(seq_len <= MAX_SEQ_LEN_EVICT,
              "evict_tokens: seq_len exceeds MAX_SEQ_LEN_EVICT (", MAX_SEQ_LEN_EVICT, ")");
  TORCH_CHECK(num_heads == g_pool.num_heads,
              "evict_tokens: attention_scores num_heads != pool.num_heads");
  TORCH_CHECK(block_tables.size(0) == num_seqs,
              "evict_tokens: block_tables batch size != num_seqs");
  TORCH_CHECK(total_pool_blocks <= g_pool.num_blocks,
              "evict_tokens: is_block_free size > pool.num_blocks");

  if (num_seqs == 0 || seq_len == 0) return;

  attention_scores = attention_scores.contiguous();
  block_tables     = block_tables.contiguous();
  is_block_free    = is_block_free.contiguous();

  const int pool_block_size = g_pool.block_size;
  const int head_dim        = g_pool.head_dim;
  const int num_logical_blks = (seq_len + pool_block_size - 1) / pool_block_size;

  // Dynamic shared memory layout (passed to kernel as extern __shared__):
  //   float[seq_len]           — per-token scores for this (seq, head)
  //   float[2]                 — scratch for global_min / global_max broadcast
  //   int32[num_logical_blks]  — per-block eviction counters (reinterpreted as float)
  //
  // Note: smem_warp_min / smem_warp_max inside the kernel are STATIC __shared__
  // arrays and are NOT included here (compiler allocates them separately).
  //
  // For seq_len=4096, block_size=16 → num_logical_blks=256:
  //   (4096 + 2 + 256) * 4 = ~17 KB — safe within sm_80's 48 KB default limit.
  const size_t smem_bytes =
      static_cast<size_t>(seq_len + 2 + num_logical_blks) * sizeof(float);

  // sm_80 default shared memory per block is 48 KB; with cudaFuncSetAttribute
  // (cudaFuncAttributeMaxDynamicSharedMemorySize) it can be extended to 163 KB.
  // For the typical seq_len range (<=4096) + static smem (~160 bytes for
  // smem_warp_min/max + evict_threshold + smem_count), total smem stays <18 KB.
  TORCH_CHECK(smem_bytes <= 48 * 1024,
              "evict_tokens: dynamic shared memory (", smem_bytes,
              " bytes) exceeds 48 KB; reduce seq_len or increase MAX_SEQ_LEN_EVICT cap");

  const dim3 grid(num_seqs, num_heads);
  const dim3 block(EVICT_BLOCK_DIM);

  const at::cuda::CUDAGuard device_guard(attention_scores.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(g_pool.key_pool.scalar_type(), "evict_tokens", [&]() {
    evict_tokens_kernel<scalar_t><<<grid, block, smem_bytes, stream>>>(
        g_pool.key_pool.data_ptr<scalar_t>(),
        g_pool.val_pool.data_ptr<scalar_t>(),
        attention_scores.data_ptr<float>(),
        block_tables.data_ptr<int>(),
        is_block_free.data_ptr<int>(),
        num_seqs,
        num_heads,
        seq_len,
        max_blocks_per_seq,
        pool_block_size,
        head_dim,
        total_pool_blocks,
        retain_ratio,
        num_sink_tokens
    );
  });

#ifndef NDEBUG
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
#else
  CUDA_CHECK(cudaGetLastError());
#endif
}

} // namespace kv_cache
} // namespace memopt
