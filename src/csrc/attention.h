// attention.h
#pragma once
#include <torch/extension.h>

namespace memopt {
namespace attention {

/**
 * Computes sparse attention over paged/compressed KV cache.
 * Designed to reduce activation memory by restricting the sequence lengths.
 * 
 * @param query Query tensor (Batch_Size, Seq_Len, Num_Heads, Head_Dim)
 * @param key_cache Managed block memory for Keys
 * @param value_cache Managed block memory for Values
 * @param block_tables Mapping of sequence to blocks
 * @param context_lens The actual length of context per sequence
 * @param max_seq_len The maximum sequence length possible in this batch
 * @param window_size Sliding window constraint for local attention (hybrid attention)
 * @return Output tensor (Batch_Size, Seq_Len, Num_Heads, Head_Dim)
 */
torch::Tensor dynamic_sparse_attention(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int max_seq_len,
    int window_size
);

} // namespace attention
} // namespace memopt
