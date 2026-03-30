// attention.cu
//
// Sparse FlashAttention forward kernel for paged KV caches.
//
// Design overview
// ---------------
//  Inputs
//    query       : [batch, q_len, num_heads, head_dim]          fp16
//    key_cache   : [num_kv_blocks, block_size, num_heads, head_dim]  fp16
//    value_cache : [num_kv_blocks, block_size, num_heads, head_dim]  fp16
//    block_tables: [batch, max_blocks_per_seq]  int32
//                  Physical page index per logical KV block.
//                  Sentinel -1 means the block was evicted / never allocated.
//    context_lens: [batch]  int32
//                  Valid KV token count per sequence.
//    window_size : Local attention window (sliding window constraint).
//                  Only KV positions in [ctx_len - window_size, ctx_len) are
//                  fetched.  Pass max_seq_len or INT_MAX to attend to all.
//
//  Grid / block mapping
//  ---------------------
//   gridDim  = (batch_size, num_heads, q_len)
//   blockDim = (ATTN_BLOCK_DIM,)  — each thread owns one head_dim lane
//
//  Shared memory layout (static, template-sized)
//  -----------------------------------------------
//   smem_k      [BLOCK_SIZE * (HEAD_DIM + 1)]  __half   K tile, +1 pad avoids bank conflicts
//   smem_v      [BLOCK_SIZE * (HEAD_DIM + 1)]  __half   V tile
//   smem_qk     [BLOCK_SIZE]                   float    QK dot products
//   smem_scratch[ATTN_BLOCK_DIM]               float    warp-reduction scratch (separate)
//
//  Online softmax
//  ---------------
//   Each thread maintains (m, l, o) in registers.
//   For each KV block, after computing QK scores into smem_qk, every thread
//   reads the full smem_qk[0..BLOCK_SIZE) array and updates its own (m, l, o).
//   This is correct because smem_qk is broadcast-read (no race) and (m,l,o)
//   are per-thread (per head_dim lane) scalars.
//
//  SM target: Ampere (sm_80) primary; sm_75 compatible via __shfl_xor_sync.

#pragma once
#include "attention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <float.h>
#include <stdint.h>
#include <cmath>
#include <climits>

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
static constexpr int ATTN_BLOCK_DIM = 128;  // threads/block; must be >= head_dim

namespace memopt {
namespace attention {

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/*
 * sparse_flash_attention_kernel<BLOCK_SIZE, HEAD_DIM>
 *
 * Template parameters:
 *   BLOCK_SIZE : tokens per KV page (must match pool init block_size)
 *   HEAD_DIM   : head dimension (compile-time for loop unrolling)
 *
 * One CUDA block handles one (batch_idx, head_idx, q_pos) triple.
 * Thread `tid` owns head_dim lane `tid` (if tid < HEAD_DIM).
 *
 * Shared memory (declared as extern, size passed at launch):
 *   Offset 0                                : smem_k  [BLOCK_SIZE*(HEAD_DIM+1)]  __half
 *   Offset BLOCK_SIZE*(HEAD_DIM+1)*2 bytes  : smem_v  [BLOCK_SIZE*(HEAD_DIM+1)]  __half
 *   After smem_v                            : smem_qk [BLOCK_SIZE]               float
 *   After smem_qk                           : smem_scratch[ATTN_BLOCK_DIM]       float
 */
template <int BLOCK_SIZE, int HEAD_DIM>
__global__
__launch_bounds__(ATTN_BLOCK_DIM)
void sparse_flash_attention_kernel(
    const __half* __restrict__ query,        // [batch, q_len, num_heads, head_dim]
    const __half* __restrict__ key_cache,    // [num_kv_blocks, block_size, num_heads, head_dim]
    const __half* __restrict__ value_cache,  // [num_kv_blocks, block_size, num_heads, head_dim]
    const int*    __restrict__ block_tables, // [batch, max_blocks_per_seq]
    const int*    __restrict__ context_lens, // [batch]
    __half*       __restrict__ out,          // [batch, q_len, num_heads, head_dim]
    int num_heads,
    int q_len,
    int max_blocks_per_seq,
    int window_size,
    float scale                              // 1 / sqrt(head_dim)
)
{
    // -----------------------------------------------------------------------
    // Grid decoding
    // -----------------------------------------------------------------------
    const int batch_idx = static_cast<int>(blockIdx.x);
    const int head_idx  = static_cast<int>(blockIdx.y);
    const int q_pos     = static_cast<int>(blockIdx.z);
    const int tid       = static_cast<int>(threadIdx.x);

    // -----------------------------------------------------------------------
    // Shared memory layout
    // -----------------------------------------------------------------------
    extern __shared__ char _smem[];

    // K tile:      [BLOCK_SIZE][HEAD_DIM + 1]  (fp16)
    __half* smem_k = reinterpret_cast<__half*>(_smem);
    // V tile:      [BLOCK_SIZE][HEAD_DIM + 1]  (fp16)
    __half* smem_v = smem_k + BLOCK_SIZE * (HEAD_DIM + 1);
    // QK scores:   [BLOCK_SIZE]  (float)  — written by warp-0-lane-0, read by all threads
    float*  smem_qk      = reinterpret_cast<float*>(smem_v + BLOCK_SIZE * (HEAD_DIM + 1));
    // Warp-level reduction scratch: [ATTN_BLOCK_DIM]  (float)
    // Separate from smem_qk to avoid aliasing during the two-level reduction.
    float*  smem_scratch = smem_qk + BLOCK_SIZE;

    // -----------------------------------------------------------------------
    // Load this thread's query element (register)
    // query: [batch, q_len, num_heads, head_dim]
    // -----------------------------------------------------------------------
    float q_val = 0.f;
    if (tid < HEAD_DIM) {
        const int q_off = ((batch_idx * q_len + q_pos) * num_heads + head_idx) * HEAD_DIM + tid;
        q_val = __half2float(query[q_off]);
    }

    // -----------------------------------------------------------------------
    // Online softmax state (per-thread registers)
    // -----------------------------------------------------------------------
    float m_running = -FLT_MAX;  // running max of QK scores
    float l_running = 0.f;       // running sum of softmax numerator
    float o_running = 0.f;       // output accumulator for this head_dim lane

    // -----------------------------------------------------------------------
    // Context / window bounds
    // -----------------------------------------------------------------------
    const int ctx_len   = context_lens[batch_idx];
    const int win_start = (window_size < ctx_len) ? (ctx_len - window_size) : 0;

    // -----------------------------------------------------------------------
    // Row pointers
    // -----------------------------------------------------------------------
    const int* bt_row = block_tables + batch_idx * max_blocks_per_seq;

    // -----------------------------------------------------------------------
    // Main loop: iterate over logical KV block slots
    // -----------------------------------------------------------------------
    for (int blk_slot = 0; blk_slot < max_blocks_per_seq; ++blk_slot) {
        const int phys_page = bt_row[blk_slot];

        // Skip evicted / unallocated pages
        if (phys_page < 0) continue;

        // Token range for this logical block slot
        const int tok_start = blk_slot * BLOCK_SIZE;  // first absolute KV position
        const int tok_end   = tok_start + BLOCK_SIZE; // one past last

        // Skip block if entirely outside context
        if (tok_start >= ctx_len) continue;
        // Skip block if entirely before the attention window
        if (tok_end   <= win_start) continue;

        // Effective token range within this block that we care about
        const int eff_start = (tok_start > win_start) ? tok_start : win_start;
        const int eff_end   = (tok_end   < ctx_len  ) ? tok_end   : ctx_len;

        // -----------------------------------------------------------------------
        // Load K tile into shared memory
        // key_cache: [num_kv_blocks, block_size, num_heads, head_dim]
        // smem_k layout: [BLOCK_SIZE][HEAD_DIM + 1]
        //
        // Thread `tid` loads elements in a grid-stride loop over (kv_slot * HEAD_DIM + dim).
        // Invalid slots (outside [eff_start, eff_end)) are filled with 0.
        // -----------------------------------------------------------------------
        {
            constexpr int N = BLOCK_SIZE * HEAD_DIM;
            for (int idx = tid; idx < N; idx += ATTN_BLOCK_DIM) {
                const int kv_slot = idx / HEAD_DIM;  // position within block page [0, BLOCK_SIZE)
                const int dim     = idx % HEAD_DIM;
                const int abs_tok = tok_start + kv_slot; // absolute KV token position

                __half val = __float2half(0.f);
                if (abs_tok >= eff_start && abs_tok < eff_end) {
                    // page_off == kv_slot here since tok_start == phys_page base
                    const int k_off = ((phys_page * BLOCK_SIZE + kv_slot) * num_heads
                                       + head_idx) * HEAD_DIM + dim;
                    val = key_cache[k_off];
                }
                smem_k[kv_slot * (HEAD_DIM + 1) + dim] = val;
            }
        }

        // Load V tile
        {
            constexpr int N = BLOCK_SIZE * HEAD_DIM;
            for (int idx = tid; idx < N; idx += ATTN_BLOCK_DIM) {
                const int kv_slot = idx / HEAD_DIM;
                const int dim     = idx % HEAD_DIM;
                const int abs_tok = tok_start + kv_slot;

                __half val = __float2half(0.f);
                if (abs_tok >= eff_start && abs_tok < eff_end) {
                    const int v_off = ((phys_page * BLOCK_SIZE + kv_slot) * num_heads
                                       + head_idx) * HEAD_DIM + dim;
                    val = value_cache[v_off];
                }
                smem_v[kv_slot * (HEAD_DIM + 1) + dim] = val;
            }
        }

        __syncthreads();

        // -----------------------------------------------------------------------
        // Compute QK dot products for each KV slot in this block.
        //
        // For each kv_slot, every thread computes its partial product:
        //   partial = q_val[tid] * k[kv_slot][tid]
        // then we reduce partial over all HEAD_DIM threads to get the dot product.
        //
        // Two-level warp reduction (works for HEAD_DIM up to 128 = 4 warps):
        //   1. Warp-level __shfl_xor_sync reduces within each warp.
        //   2. Warp lane-0 writes partial sum to smem_scratch[warp_id].
        //   3. After __syncthreads, warp 0 reads smem_scratch and does a final
        //      warp-level reduction to produce the full dot product.
        //   4. Thread 0 writes the scaled result to smem_qk[kv_slot].
        //
        // We keep smem_scratch[0..num_warps-1] as the inter-warp buffer and
        // smem_qk[0..BLOCK_SIZE-1] as the final per-kv_slot QK scores.
        // These do NOT alias because smem_scratch follows smem_qk in layout.
        // -----------------------------------------------------------------------
        constexpr int NUM_WARPS = (ATTN_BLOCK_DIM + 31) / 32;

        for (int kv_slot = 0; kv_slot < BLOCK_SIZE; ++kv_slot) {
            // Each thread computes its partial dot product contribution
            float partial = 0.f;
            if (tid < HEAD_DIM) {
                partial = q_val * __half2float(smem_k[kv_slot * (HEAD_DIM + 1) + tid]);
            }

            // Step 1: warp-level reduction
            #pragma unroll
            for (int shfl_mask = 16; shfl_mask >= 1; shfl_mask >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, shfl_mask);
            }

            // Step 2: warp lane 0 writes to inter-warp scratch
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;
            if (lane_id == 0) {
                smem_scratch[warp_id] = partial;
            }
            __syncthreads();

            // Step 3: warp 0 reduces inter-warp partial sums
            float dot = 0.f;
            if (tid < NUM_WARPS) {
                dot = smem_scratch[tid];
            }
            if (tid < 32) {
                #pragma unroll
                for (int shfl_mask = 16; shfl_mask >= 1; shfl_mask >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, shfl_mask);
                }
            }

            // Step 4: thread 0 writes scaled QK score to smem_qk[kv_slot]
            if (tid == 0) {
                smem_qk[kv_slot] = dot * scale;
            }
            __syncthreads();
        } // end kv_slot QK loop

        // -----------------------------------------------------------------------
        // Online softmax update
        //
        // All threads read smem_qk[] (broadcast read — no race) and update
        // their own (m_running, l_running, o_running) state.
        // Tokens outside [eff_start, eff_end) get score -FLT_MAX (masked).
        // -----------------------------------------------------------------------
        #pragma unroll
        for (int kv_slot = 0; kv_slot < BLOCK_SIZE; ++kv_slot) {
            const int abs_tok = tok_start + kv_slot;
            const bool valid  = (abs_tok >= eff_start) && (abs_tok < eff_end);
            const float qk    = valid ? smem_qk[kv_slot] : -FLT_MAX;

            // Standard online softmax update:
            //   m_new = max(m, qk)
            //   alpha  = exp(m - m_new)          -- rescale factor for prior accumulator
            //   l_new  = alpha * l + exp(qk - m_new)
            //   o_new  = alpha * o + exp(qk - m_new) * v[kv_slot]
            const float m_new   = fmaxf(m_running, qk);
            const float exp_qk  = (qk <= -FLT_MAX * 0.5f) ? 0.f : __expf(qk - m_new);
            const float alpha   = __expf(m_running - m_new);

            l_running = alpha * l_running + exp_qk;

            if (tid < HEAD_DIM) {
                const float v_val = __half2float(smem_v[kv_slot * (HEAD_DIM + 1) + tid]);
                o_running = alpha * o_running + exp_qk * v_val;
            }

            m_running = m_new;
        }

        __syncthreads();
    } // end block slot loop

    // -----------------------------------------------------------------------
    // Normalise and write output
    // out: [batch, q_len, num_heads, head_dim]
    // -----------------------------------------------------------------------
    if (tid < HEAD_DIM) {
        const float out_val = (l_running > 0.f) ? (o_running / l_running) : 0.f;
        const int out_off   = ((batch_idx * q_len + q_pos) * num_heads + head_idx)
                              * HEAD_DIM + tid;
        out[out_off] = __float2half(out_val);
    }
}

// ---------------------------------------------------------------------------
// Shared memory size helper
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE, int HEAD_DIM>
static inline size_t smem_bytes_for() {
    // smem_k:       BLOCK_SIZE * (HEAD_DIM + 1) * sizeof(__half)
    // smem_v:       BLOCK_SIZE * (HEAD_DIM + 1) * sizeof(__half)
    // smem_qk:      BLOCK_SIZE * sizeof(float)
    // smem_scratch: ATTN_BLOCK_DIM * sizeof(float)
    return static_cast<size_t>(BLOCK_SIZE) * (HEAD_DIM + 1) * sizeof(__half) * 2
         + static_cast<size_t>(BLOCK_SIZE) * sizeof(float)
         + static_cast<size_t>(ATTN_BLOCK_DIM) * sizeof(float);
}

// ---------------------------------------------------------------------------
// Host-side dispatch function
// ---------------------------------------------------------------------------

/*
 * dynamic_sparse_attention
 *
 * Launches sparse_flash_attention_kernel for the given paged KV cache.
 * Dispatches on (block_size, head_dim) at runtime via compile-time templates.
 *
 * Supported (block_size, head_dim) pairs:
 *   (16, 64), (16, 128), (32, 64), (32, 128)
 *
 * All inputs must be CUDA tensors. query/key_cache/value_cache must be fp16.
 * block_tables and context_lens must be int32.
 */
torch::Tensor dynamic_sparse_attention(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int max_seq_len,
    int window_size
) {
    // -----------------------------------------------------------------------
    // Input validation
    // -----------------------------------------------------------------------
    TORCH_CHECK(query.is_cuda(),       "query must be a CUDA tensor");
    TORCH_CHECK(key_cache.is_cuda(),   "key_cache must be a CUDA tensor");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
    TORCH_CHECK(block_tables.is_cuda(),"block_tables must be a CUDA tensor");
    TORCH_CHECK(context_lens.is_cuda(),"context_lens must be a CUDA tensor");

    TORCH_CHECK(query.dim() == 4,
        "query must be 4-D [batch, q_len, num_heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 4,
        "key_cache must be 4-D [num_kv_blocks, block_size, num_heads, head_dim]");
    TORCH_CHECK(value_cache.dim() == 4,
        "value_cache must be 4-D [num_kv_blocks, block_size, num_heads, head_dim]");
    TORCH_CHECK(block_tables.dim() == 2,
        "block_tables must be 2-D [batch, max_blocks_per_seq]");
    TORCH_CHECK(context_lens.dim() == 1,
        "context_lens must be 1-D [batch]");

    TORCH_CHECK(query.scalar_type()       == torch::kFloat16, "query must be fp16");
    TORCH_CHECK(key_cache.scalar_type()   == torch::kFloat16, "key_cache must be fp16");
    TORCH_CHECK(value_cache.scalar_type() == torch::kFloat16, "value_cache must be fp16");
    TORCH_CHECK(block_tables.scalar_type()== torch::kInt32,   "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type()== torch::kInt32,   "context_lens must be int32");

    // Contiguity: kernels use raw strides derived from shapes, so we require
    // contiguous inputs to guarantee stride == size product.
    TORCH_CHECK(query.is_contiguous(),       "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(),   "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(),"block_tables must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(),"context_lens must be contiguous");

    // -----------------------------------------------------------------------
    // Dimension extraction
    // -----------------------------------------------------------------------
    const int batch_size         = static_cast<int>(query.size(0));
    const int q_len              = static_cast<int>(query.size(1));
    const int num_heads          = static_cast<int>(query.size(2));
    const int head_dim           = static_cast<int>(query.size(3));
    const int block_size         = static_cast<int>(key_cache.size(1));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    TORCH_CHECK(key_cache.size(2)   == num_heads,
        "key_cache num_heads ", key_cache.size(2), " != query num_heads ", num_heads);
    TORCH_CHECK(key_cache.size(3)   == head_dim,
        "key_cache head_dim ",  key_cache.size(3), " != query head_dim ",  head_dim);
    TORCH_CHECK(value_cache.size(1) == block_size,
        "value_cache block_size mismatch");
    TORCH_CHECK(value_cache.size(2) == num_heads,
        "value_cache num_heads mismatch");
    TORCH_CHECK(value_cache.size(3) == head_dim,
        "value_cache head_dim mismatch");
    TORCH_CHECK(block_tables.size(0) == batch_size,
        "block_tables batch dimension mismatch");
    TORCH_CHECK(context_lens.size(0) == batch_size,
        "context_lens batch dimension mismatch");
    TORCH_CHECK(head_dim <= ATTN_BLOCK_DIM,
        "head_dim ", head_dim, " must be <= ATTN_BLOCK_DIM (", ATTN_BLOCK_DIM, ")");

    // -----------------------------------------------------------------------
    // Output allocation
    // -----------------------------------------------------------------------
    auto out = torch::empty_like(query);

    if (batch_size == 0 || q_len == 0 || num_heads == 0 || head_dim == 0) {
        return out;
    }

    // -----------------------------------------------------------------------
    // Clamp window_size
    // -----------------------------------------------------------------------
    if (window_size <= 0 || window_size > max_seq_len) {
        window_size = max_seq_len;
    }

    // -----------------------------------------------------------------------
    // Scale factor: 1 / sqrt(head_dim)
    // -----------------------------------------------------------------------
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // -----------------------------------------------------------------------
    // Grid / block configuration
    // -----------------------------------------------------------------------
    dim3 grid(batch_size, num_heads, q_len);
    dim3 block(ATTN_BLOCK_DIM);

    // -----------------------------------------------------------------------
    // Device guard
    // -----------------------------------------------------------------------
    c10::cuda::CUDAGuard device_guard(query.device());

    // -----------------------------------------------------------------------
    // Raw pointers
    // -----------------------------------------------------------------------
    const __half* q_ptr  = reinterpret_cast<const __half*>(query.data_ptr<at::Half>());
    const __half* k_ptr  = reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>());
    const __half* v_ptr  = reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>());
    const int*    bt_ptr = block_tables.data_ptr<int>();
    const int*    cl_ptr = context_lens.data_ptr<int>();
          __half* o_ptr  = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    // -----------------------------------------------------------------------
    // Template dispatch on (block_size, head_dim)
    //
    // Supported: block_size in {16, 32}, head_dim in {64, 128}.
    // Add more pairs here as needed for different model configurations.
    // -----------------------------------------------------------------------
#define LAUNCH_KERNEL(BS, HD)                                                   \
    sparse_flash_attention_kernel<BS, HD><<<grid, block, smem_bytes_for<BS,HD>()>>>( \
        q_ptr, k_ptr, v_ptr, bt_ptr, cl_ptr, o_ptr,                            \
        num_heads, q_len, max_blocks_per_seq, window_size, scale)

    if (block_size == 16 && head_dim == 64) {
        LAUNCH_KERNEL(16, 64);
    } else if (block_size == 16 && head_dim == 128) {
        LAUNCH_KERNEL(16, 128);
    } else if (block_size == 32 && head_dim == 64) {
        LAUNCH_KERNEL(32, 64);
    } else if (block_size == 32 && head_dim == 128) {
        LAUNCH_KERNEL(32, 128);
    } else {
        TORCH_CHECK(false,
            "dynamic_sparse_attention: unsupported (block_size=", block_size,
            ", head_dim=", head_dim,
            "). Supported combinations: block_size in {16,32}, head_dim in {64,128}.");
    }
#undef LAUNCH_KERNEL

    // -----------------------------------------------------------------------
    // Error checking
    // -----------------------------------------------------------------------
#ifndef NDEBUG
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#else
    CUDA_CHECK(cudaGetLastError());
#endif

    return out;
}

} // namespace attention
} // namespace memopt
