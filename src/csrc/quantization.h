// quantization.h
#pragma once
#include <torch/extension.h>
#include <vector>

namespace memopt {
namespace quantization {

/**
 * Compresses an FP16 cache tensor into packed INT4 using group-wise symmetric
 * quantization (zero_point = 8, scale = max_abs / 7.0f per group).
 *
 * Args:
 *   cache      : FP16 CUDA tensor, any shape; last dim must be even.
 *   group_size : quantization group size (must be 64).
 *
 * Returns:
 *   [packed, scales]
 *     packed : UInt8 tensor, same shape as cache but last dim / 2.
 *     scales : float32 tensor, shape [total_elements / group_size].
 */
std::vector<torch::Tensor> quantize_kv_cache_int4(torch::Tensor cache, int group_size);

/**
 * Decompresses a packed INT4 cache tensor back to FP16.
 *
 * Args:
 *   q_cache    : UInt8 packed INT4 CUDA tensor.
 *   scales     : float32 per-group scales tensor.
 *   group_size : quantization group size (must be 64).
 *
 * Returns:
 *   FP16 tensor with last dim = q_cache last dim * 2.
 */
torch::Tensor dequantize_kv_cache_int4(torch::Tensor q_cache, torch::Tensor scales, int group_size);

} // namespace quantization
} // namespace memopt
