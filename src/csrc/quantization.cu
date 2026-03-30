// quantization.cu
//
// INT4 group-wise symmetric quantization and dequantization kernels for the
// MemOpt paged KV cache.
//
// Design overview
// ---------------
//  Packing convention:
//    Two INT4 values are packed into one uint8_t byte:
//      byte = (q_lo & 0xF) | ((q_hi & 0xF) << 4)
//    where q_lo is the element at even index within the group (lower nibble)
//    and q_hi is the element at odd index (upper nibble).
//
//  Quantization (symmetric, zero_point = 8):
//    scale = max(abs(group)) / 7.0f
//    q     = clamp(round(val / scale) + 8, 0, 15)
//    Unsigned INT4 range [0, 15], represents signed [-7, 7] via offset-8.
//
//  quantize_int4_kernel grid/block mapping
//  ----------------------------------------
//    Grid  : (num_groups,)        — one CUDA block per quantization group
//    Block : (QUANT_BLOCK_DIM=64) — one thread per element in the group
//    Shared memory layout:
//      float smem_abs[64 + 1]   — abs-values with +1 padding to avoid bank
//                                  conflicts on the 32-thread-wide reduction
//    Phase 1: warp-level __shfl_xor_sync max-reduction within each warp (32t),
//             then cross-warp reduction via smem.
//    Phase 2: thread 0 writes scale; all threads clamp/round/pack.
//
//  dequantize_int4_kernel grid/block mapping
//  ------------------------------------------
//    Grid  : (num_groups,)        — one CUDA block per group
//    Block : (DEQUANT_BLOCK_DIM=32) — one thread per packed byte
//    Each thread handles exactly 1 byte → 2 FP16 outputs.
//    Uses __half2 stores for the two adjacent outputs → coalesced writes.
//
// SM architecture target: Ampere (sm_80) primary; Turing (sm_75) compatible.
// Requires CUDA 11+ for __half2 and cuda_fp16.h intrinsics.

#pragma once
#include "quantization.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <cstdio>
#include <cmath>

// ---------------------------------------------------------------------------
// Debug helpers  (mirrors kv_cache.cu convention)
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
static constexpr int QUANT_BLOCK_DIM   = 64;  // threads/block for quantize kernel
static constexpr int DEQUANT_BLOCK_DIM = 32;  // threads/block for dequantize kernel

namespace memopt {
namespace quantization {

// ---------------------------------------------------------------------------
// quantize_int4_kernel
// ---------------------------------------------------------------------------

/*
 * CUDA kernel: group-wise symmetric INT4 quantization of FP16 inputs.
 *
 * Grid  : (num_groups,)               — dim3(num_groups, 1, 1)
 * Block : (QUANT_BLOCK_DIM = 64)      — one thread per element in the group
 *
 * Arguments:
 *   input      — FP16 flat input array, length = num_groups * group_size
 *   packed_out — UInt8 output,          length = num_groups * group_size / 2
 *   scales_out — float32 scales output, length = num_groups
 *   group_size — elements per group (must equal QUANT_BLOCK_DIM = 64)
 *
 * Shared memory layout:
 *   float smem_abs[QUANT_BLOCK_DIM + 1]  — padded by 1 to prevent bank conflicts
 *   float smem_scale[1]                  — broadcast scale value
 *
 * Algorithm:
 *   1. Each thread loads one FP16 element, converts to float32, stores abs in smem.
 *   2. Warp-level __shfl_xor_sync max-reduction (warp 0 and warp 1 separately).
 *   3. Cross-warp reduction via smem[0] vs smem[32].
 *   4. Thread 0 computes scale = max_abs / 7.0f, writes to scales_out.
 *   5. All threads compute q = clamp(round(val / scale) + 8, 0, 15).
 *   6. Even threads (threadIdx.x % 2 == 0) pack two nibbles and write one byte.
 */
__global__ void __launch_bounds__(QUANT_BLOCK_DIM)
quantize_int4_kernel(
    const __half* __restrict__ input,       // [num_groups * group_size]
    uint8_t*      __restrict__ packed_out,  // [num_groups * group_size / 2]
    float*        __restrict__ scales_out,  // [num_groups]
    int group_size                          // elements per group (== QUANT_BLOCK_DIM)
) {
    const int group_idx = blockIdx.x;
    const int tid       = threadIdx.x;

    // Base index into the flat input/output arrays for this group
    const int64_t group_base_in  = static_cast<int64_t>(group_idx) * group_size;
    const int64_t group_base_out = static_cast<int64_t>(group_idx) * (group_size / 2);

    // -------------------------------------------------------------------
    // Phase 1: load element, compute abs-value
    // -------------------------------------------------------------------

    // Shared memory: padded by +1 per 32-element bank row to avoid conflicts.
    // Layout: [QUANT_BLOCK_DIM + 1] floats for abs values, then 1 float for scale.
    // Total: (QUANT_BLOCK_DIM + 1 + 1) floats — small enough to fit easily.
    __shared__ float smem_abs[QUANT_BLOCK_DIM + 1];   // +1 pad for bank-conflict avoidance
    __shared__ float smem_scale;                       // broadcast scale from thread 0
    __shared__ float smem_vals[QUANT_BLOCK_DIM];       // original fp32 values for quantization

    float val  = 0.f;
    float aval = 0.f;

    if (tid < group_size) {
        val  = __half2float(input[group_base_in + tid]);
        aval = fabsf(val);
    }

    smem_abs[tid]  = aval;
    smem_vals[tid] = val;
    __syncthreads();

    // -------------------------------------------------------------------
    // Phase 2: two-phase max-reduction to find max_abs across the group
    //   Phase 2a: warp-level reduction via __shfl_xor_sync (within warp)
    //   Phase 2b: cross-warp reduction via shared memory
    // -------------------------------------------------------------------

    // Warp-level max (each warp of 32 threads independently)
    float warp_max = aval;
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, offset));
    }

    // Warp leaders (tid 0 and tid 32) write their warp-max to smem_abs
    // Use positions [QUANT_BLOCK_DIM - 2] and [QUANT_BLOCK_DIM - 1] as
    // inter-warp communication slots (outside the per-element range).
    // Actually use a cleaner approach: separate accumulator slots.
    // We re-use smem_abs[0] and smem_abs[32] as warp-leader slots.
    if ((tid & 31) == 0) {
        smem_abs[tid] = warp_max;  // tid==0 → slot 0; tid==32 → slot 32
    }
    __syncthreads();

    // Thread 0 does the final cross-warp reduction and computes scale
    if (tid == 0) {
        float max_abs = fmaxf(smem_abs[0], smem_abs[32]);
        // Avoid division by zero: if all values are zero, scale = 1.0f
        float scale = (max_abs > 0.f) ? (max_abs / 7.0f) : 1.0f;
        smem_scale      = scale;
        scales_out[group_idx] = scale;
    }
    __syncthreads();

    // -------------------------------------------------------------------
    // Phase 3: quantize each element and pack pairs into bytes
    // -------------------------------------------------------------------

    const float scale = smem_scale;

    // Compute quantized value for this element
    int q = 0;
    if (tid < group_size) {
        float scaled = smem_vals[tid] / scale;
        // round-to-nearest, offset by 8, clamp to [0, 15]
        int iq = static_cast<int>(roundf(scaled)) + 8;
        q = min(max(iq, 0), 15);
    }

    // Pack: even thread tid packs (q[tid], q[tid+1]) into one byte.
    // Even thread reads its nibble from smem; needs the odd thread's nibble too.
    // Use smem_abs (now free since reduction is done) as nibble staging area.
    __shared__ uint8_t smem_nibbles[QUANT_BLOCK_DIM];
    smem_nibbles[tid] = static_cast<uint8_t>(q);
    __syncthreads();

    // Only even threads write packed bytes
    if (tid % 2 == 0) {
        uint8_t q_lo = smem_nibbles[tid];         // even index within group
        uint8_t q_hi = smem_nibbles[tid + 1];     // odd index within group
        uint8_t byte = (q_lo & 0xF) | ((q_hi & 0xF) << 4);
        packed_out[group_base_out + tid / 2] = byte;
    }
}

// ---------------------------------------------------------------------------
// dequantize_int4_kernel
// ---------------------------------------------------------------------------

/*
 * CUDA kernel: INT4 dequantization back to FP16.
 *
 * Grid  : (num_groups,)               — dim3(num_groups, 1, 1)
 * Block : (DEQUANT_BLOCK_DIM = 32)    — one thread per packed byte
 *
 * Arguments:
 *   packed_in  — UInt8 packed input,   length = num_groups * group_size / 2
 *   scales_in  — float32 scales,       length = num_groups
 *   output     — FP16 output,          length = num_groups * group_size
 *   group_size — elements per group (must equal 2 * DEQUANT_BLOCK_DIM = 64)
 *
 * Algorithm:
 *   Each thread reads 1 packed byte:
 *     lo = byte & 0xF           → FP16 output at index 2*tid
 *     hi = (byte >> 4) & 0xF   → FP16 output at index 2*tid+1
 *   Dequantize: fp16_val = (q - 8) * scale
 *   Writes a __half2 vector (lo, hi) at the aligned output position.
 *   This gives fully coalesced 64-bit stores across the warp.
 *
 * Note: group_size must be 64 so that DEQUANT_BLOCK_DIM=32 threads each
 *       handle exactly 1 byte → 2 outputs, covering the full group.
 */
__global__ void __launch_bounds__(DEQUANT_BLOCK_DIM)
dequantize_int4_kernel(
    const uint8_t* __restrict__ packed_in,   // [num_groups * group_size / 2]
    const float*   __restrict__ scales_in,   // [num_groups]
    __half*        __restrict__ output,      // [num_groups * group_size]
    int group_size                           // elements per group (== 64)
) {
    const int group_idx = blockIdx.x;
    const int tid       = threadIdx.x;

    const int bytes_per_group = group_size / 2;   // == 32

    // Base indices
    const int64_t byte_base   = static_cast<int64_t>(group_idx) * bytes_per_group;
    const int64_t out_base    = static_cast<int64_t>(group_idx) * group_size;

    // Load the scale for this group (broadcast; all threads read the same value)
    const float scale = scales_in[group_idx];

    // Each thread handles one packed byte
    #pragma unroll
    {
        const uint8_t byte = packed_in[byte_base + tid];

        // Unpack nibbles
        const int q_lo = static_cast<int>(byte & 0xF);
        const int q_hi = static_cast<int>((byte >> 4) & 0xF);

        // Dequantize to float32 then convert to FP16
        const __half h_lo = __float2half(static_cast<float>(q_lo - 8) * scale);
        const __half h_hi = __float2half(static_cast<float>(q_hi - 8) * scale);

        // __half2 vector store: two FP16 values in one 32-bit write
        // Output layout: even index = lo nibble, odd index = hi nibble
        // tid maps to output positions [2*tid, 2*tid+1]
        __half2 pair = __halves2half2(h_lo, h_hi);
        reinterpret_cast<__half2*>(output + out_base)[tid] = pair;
    }
}

// ---------------------------------------------------------------------------
// quantize_kv_cache_int4 — Python-visible host function
// ---------------------------------------------------------------------------

/*
 * Quantizes an FP16 KV cache tensor to packed INT4 using group-wise symmetric
 * quantization.
 *
 * Args:
 *   cache      : Tensor[..., last_dim] — FP16 tensor (any shape, flattened
 *                to 1-D for quantization).  last_dim must be divisible by 2.
 *   group_size : int — elements per quantization group (must be 64).
 *
 * Returns:
 *   [packed, scales] — Python list of two tensors:
 *     packed : UInt8  tensor, same shape as cache but last dim / 2.
 *     scales : float32 tensor, shape [total_elements / group_size].
 *
 * Constraints:
 *   total_elements % group_size == 0
 *   last_dim % 2 == 0
 */
std::vector<torch::Tensor> quantize_kv_cache_int4(torch::Tensor cache, int group_size) {
    TORCH_CHECK(cache.is_cuda(),  "quantize_kv_cache_int4: cache must be a CUDA tensor");
    TORCH_CHECK(cache.is_contiguous(), "quantize_kv_cache_int4: cache must be contiguous");
    TORCH_CHECK(cache.scalar_type() == torch::kFloat16,
                "quantize_kv_cache_int4: cache must be float16");
    TORCH_CHECK(group_size == QUANT_BLOCK_DIM,
                "quantize_kv_cache_int4: group_size must equal ", QUANT_BLOCK_DIM,
                " (got ", group_size, ")");

    const int64_t total_elements = cache.numel();
    TORCH_CHECK(total_elements % group_size == 0,
                "quantize_kv_cache_int4: total_elements (", total_elements,
                ") must be divisible by group_size (", group_size, ")");

    const int64_t last_dim = cache.size(cache.dim() - 1);
    TORCH_CHECK(last_dim % 2 == 0,
                "quantize_kv_cache_int4: last dimension (", last_dim,
                ") must be even for INT4 packing");

    const int64_t num_groups = total_elements / group_size;

    // Build output shapes: packed has same dims as cache but last dim halved
    auto packed_sizes = cache.sizes().vec();
    packed_sizes.back() /= 2;

    // Allocate output tensors
    auto packed = torch::empty(packed_sizes,
                               torch::TensorOptions()
                                   .dtype(torch::kUInt8)
                                   .device(cache.device()));
    auto scales = torch::empty({num_groups},
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(cache.device()));

    if (total_elements == 0) {
        return {packed, scales};
    }

    // Kernel launch parameters
    const dim3 grid(static_cast<unsigned>(num_groups));
    const dim3 block(QUANT_BLOCK_DIM);

    const at::cuda::CUDAGuard device_guard(cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    quantize_int4_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(cache.data_ptr<at::Half>()),
        packed.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        group_size
    );

    CUDA_CHECK(cudaGetLastError());
#ifndef NDEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    return {packed, scales};
}

// ---------------------------------------------------------------------------
// dequantize_kv_cache_int4 — Python-visible host function
// ---------------------------------------------------------------------------

/*
 * Dequantizes a packed INT4 KV cache back to FP16.
 *
 * Args:
 *   q_cache    : UInt8 Tensor[..., last_dim/2] — packed INT4 cache.
 *   scales     : float32 Tensor[num_groups]    — per-group quantization scales.
 *   group_size : int — elements per quantization group (must be 64).
 *
 * Returns:
 *   FP16 Tensor with the same shape as the original cache (last dim = q_cache
 *   last dim * 2).
 */
torch::Tensor dequantize_kv_cache_int4(
    torch::Tensor q_cache,
    torch::Tensor scales,
    int group_size)
{
    TORCH_CHECK(q_cache.is_cuda(),  "dequantize_kv_cache_int4: q_cache must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(),   "dequantize_kv_cache_int4: scales must be a CUDA tensor");
    TORCH_CHECK(q_cache.is_contiguous(), "dequantize_kv_cache_int4: q_cache must be contiguous");
    TORCH_CHECK(scales.is_contiguous(),  "dequantize_kv_cache_int4: scales must be contiguous");
    TORCH_CHECK(q_cache.scalar_type() == torch::kUInt8,
                "dequantize_kv_cache_int4: q_cache must be uint8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
                "dequantize_kv_cache_int4: scales must be float32");
    TORCH_CHECK(group_size == QUANT_BLOCK_DIM,
                "dequantize_kv_cache_int4: group_size must equal ", QUANT_BLOCK_DIM);

    // The output has last dim doubled relative to q_cache
    auto out_sizes = q_cache.sizes().vec();
    out_sizes.back() *= 2;

    const int64_t total_elements = q_cache.numel() * 2;  // FP16 output elements
    const int64_t num_groups     = total_elements / group_size;

    TORCH_CHECK(scales.numel() == num_groups,
                "dequantize_kv_cache_int4: scales length (", scales.numel(),
                ") != expected num_groups (", num_groups, ")");

    // Allocate FP16 output with restored shape
    auto output = torch::empty(out_sizes,
                               torch::TensorOptions()
                                   .dtype(torch::kFloat16)
                                   .device(q_cache.device()));

    if (total_elements == 0) {
        return output;
    }

    // Kernel launch parameters
    const dim3 grid(static_cast<unsigned>(num_groups));
    const dim3 block(DEQUANT_BLOCK_DIM);  // 32 threads, one per packed byte

    const at::cuda::CUDAGuard device_guard(q_cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dequantize_int4_kernel<<<grid, block, 0, stream>>>(
        q_cache.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        group_size
    );

    CUDA_CHECK(cudaGetLastError());
#ifndef NDEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    return output;
}

} // namespace quantization
} // namespace memopt
