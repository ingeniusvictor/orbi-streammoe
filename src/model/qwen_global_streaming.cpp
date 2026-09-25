#include "orbi/streammoe/model/qwen_global_streaming.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/cpu/reference_ops.hpp"

namespace orbi::streammoe {
namespace {

void validate_row_range(
    const QwenCheckpointMatrixDescriptor& matrix,
    std::size_t first_row,
    std::size_t row_count) {
  if (row_count == 0U) {
    throw std::invalid_argument(
        "qwen global streaming: row_count must be non-zero");
  }
  if (first_row > matrix.rows ||
      row_count > matrix.rows - first_row) {
    throw std::out_of_range(
        "qwen global streaming: matrix row range is out of bounds: " +
        matrix.path);
  }
  if (matrix.cols == 0U) {
    throw std::runtime_error(
        "qwen global streaming: matrix descriptor has zero width: " +
        matrix.path);
  }
}

std::size_t checked_mul(
    std::size_t a,
    std::size_t b,
    const char* label) {
  if (a != 0U &&
      b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::runtime_error(
        std::string("qwen global streaming: size overflow: ") + label);
  }
  return a * b;
}

}  // namespace

std::vector<float> read_qwen_matrix_rows(
    const QpackMlxCheckpoint& checkpoint,
    const QwenCheckpointMatrixDescriptor& matrix,
    std::size_t first_row,
    std::size_t row_count) {
  validate_row_range(matrix, first_row, row_count);

  const auto& dense = checkpoint.dense();
  const auto weight_name = matrix.path + ".weight";

  if (!matrix.quantized) {
    const auto offset =
        checked_mul(first_row, matrix.cols, "plain row offset");
    const auto count =
        checked_mul(row_count, matrix.cols, "plain row count");
    return dense.read_floats(weight_name, offset, count);
  }

  if (!matrix.quant.has_value()) {
    throw std::runtime_error(
        "qwen global streaming: quantized matrix missing quant spec: " +
        matrix.path);
  }

  const auto spec = *matrix.quant;
  if (spec.bits != 4U && spec.bits != 8U) {
    throw std::runtime_error(
        "qwen global streaming: only affine Q4/Q8 matrices are supported");
  }
  if (spec.group_size == 0U ||
      matrix.cols % spec.group_size != 0U) {
    throw std::runtime_error(
        "qwen global streaming: quantized row/group geometry mismatch");
  }

  const auto per_word = 32U / spec.bits;
  if (matrix.cols % per_word != 0U) {
    throw std::runtime_error(
        "qwen global streaming: quantized row is not word aligned");
  }

  const auto packed_cols = matrix.cols / per_word;
  const auto groups_per_row = matrix.cols / spec.group_size;

  const auto packed_offset =
      checked_mul(first_row, packed_cols, "packed row offset");
  const auto packed_count =
      checked_mul(row_count, packed_cols, "packed row count");
  const auto meta_offset =
      checked_mul(first_row, groups_per_row, "affine metadata row offset");
  const auto meta_count =
      checked_mul(row_count, groups_per_row, "affine metadata row count");

  const auto packed = dense.read_u32(
      weight_name,
      packed_offset,
      packed_count);
  const auto scales = dense.read_floats(
      matrix.path + ".scales",
      meta_offset,
      meta_count);
  const auto biases = dense.read_floats(
      matrix.path + ".biases",
      meta_offset,
      meta_count);

  return cpu::dequantize_affine_rows(
      packed,
      row_count,
      packed_cols,
      scales,
      biases,
      cpu::AffineQuantSpec{
          .bits = spec.bits,
          .group_size = spec.group_size,
      });
}

std::vector<float> read_qwen_embedding_row(
    const QpackMlxCheckpoint& checkpoint,
    const QwenGlobalCheckpointBinding& global,
    std::size_t token_id) {
  if (token_id >= global.vocab_size) {
    throw std::out_of_range(
        "qwen global streaming: token id exceeds vocabulary");
  }

  auto row = read_qwen_matrix_rows(
      checkpoint,
      global.embed_tokens,
      token_id,
      1U);
  if (row.size() != global.hidden_size) {
    throw std::runtime_error(
        "qwen global streaming: embedding row width mismatch");
  }
  return row;
}

QwenLmHeadChunkResult project_qwen_lm_head_chunk(
    const QpackMlxCheckpoint& checkpoint,
    const QwenGlobalCheckpointBinding& global,
    std::span<const float> hidden,
    std::size_t first_token,
    std::size_t token_count) {
  if (hidden.size() != global.hidden_size) {
    throw std::invalid_argument(
        "qwen global streaming: LM-head hidden width mismatch");
  }
  if (first_token > global.vocab_size ||
      token_count == 0U ||
      token_count > global.vocab_size - first_token) {
    throw std::out_of_range(
        "qwen global streaming: LM-head vocabulary chunk is out of bounds");
  }

  const auto rows = read_qwen_matrix_rows(
      checkpoint,
      global.lm_head,
      first_token,
      token_count);

  auto logits = cpu::matvec_row_major(
      rows,
      token_count,
      global.hidden_size,
      hidden);

  return {
      .first_token = first_token,
      .logits = std::move(logits),
  };
}

QwenGreedyTokenResult greedy_qwen_lm_head_streaming(
    const QpackMlxCheckpoint& checkpoint,
    const QwenGlobalCheckpointBinding& global,
    std::span<const float> hidden,
    std::size_t chunk_rows) {
  if (hidden.size() != global.hidden_size) {
    throw std::invalid_argument(
        "qwen global streaming: greedy hidden width mismatch");
  }
  if (global.vocab_size == 0U) {
    throw std::runtime_error(
        "qwen global streaming: vocabulary must be non-zero");
  }
  if (chunk_rows == 0U) {
    throw std::invalid_argument(
        "qwen global streaming: greedy chunk_rows must be non-zero");
  }

  QwenGreedyTokenResult best{
      .token_id = 0U,
      .logit = -std::numeric_limits<float>::infinity(),
  };

  for (std::size_t first = 0U;
       first < global.vocab_size;) {
    const auto count =
        std::min(chunk_rows, global.vocab_size - first);
    const auto chunk = project_qwen_lm_head_chunk(
        checkpoint,
        global,
        hidden,
        first,
        count);

    for (std::size_t i = 0U; i < chunk.logits.size(); ++i) {
      const auto value = chunk.logits[i];
      if (value > best.logit) {
        best.logit = value;
        best.token_id = first + i;
      }
    }

    first += count;
  }

  return best;
}

}  // namespace orbi::streammoe
