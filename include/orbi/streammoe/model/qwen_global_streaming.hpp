#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/model/qwen_global_binding.hpp"

namespace orbi::streammoe {

/// Read and materialize only a bounded contiguous row range from a checkpoint
/// matrix descriptor. Quantized affine rows are dequantized only for the
/// requested range.
[[nodiscard]] std::vector<float> read_qwen_matrix_rows(
    const QpackMlxCheckpoint& checkpoint,
    const QwenCheckpointMatrixDescriptor& matrix,
    std::size_t first_row,
    std::size_t row_count);

/// Read exactly one token embedding row.
[[nodiscard]] std::vector<float> read_qwen_embedding_row(
    const QpackMlxCheckpoint& checkpoint,
    const QwenGlobalCheckpointBinding& global,
    std::size_t token_id);

struct QwenLmHeadChunkResult {
  std::size_t first_token{};
  std::vector<float> logits;
};

/// Project one hidden vector against a bounded contiguous vocabulary chunk.
/// Memory is O(token_count * hidden_size), never O(vocab_size * hidden_size).
[[nodiscard]] QwenLmHeadChunkResult project_qwen_lm_head_chunk(
    const QpackMlxCheckpoint& checkpoint,
    const QwenGlobalCheckpointBinding& global,
    std::span<const float> hidden,
    std::size_t first_token,
    std::size_t token_count);

struct QwenGreedyTokenResult {
  std::size_t token_id{};
  float logit{};
};

/// Scan the LM head in bounded chunks and return the argmax without ever
/// allocating the complete vocabulary logits vector.
[[nodiscard]] QwenGreedyTokenResult greedy_qwen_lm_head_streaming(
    const QpackMlxCheckpoint& checkpoint,
    const QwenGlobalCheckpointBinding& global,
    std::span<const float> hidden,
    std::size_t chunk_rows);

}  // namespace orbi::streammoe
