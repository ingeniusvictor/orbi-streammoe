#include "orbi/streammoe/model/qwen_global_binding.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace orbi::streammoe {
namespace {

std::size_t checked_rows(
    const std::vector<std::size_t>& shape,
    std::string_view path) {
  if (shape.size() != 2U ||
      shape[0] == 0U ||
      shape[1] == 0U) {
    throw std::runtime_error(
        "qwen global binding: expected a non-empty matrix: " +
        std::string(path));
  }
  return shape[0];
}

QwenCheckpointMatrixDescriptor describe_matrix(
    const QpackMlxCheckpoint& checkpoint,
    std::string path) {
  const auto& dense = checkpoint.dense();

  QwenCheckpointMatrixDescriptor descriptor;
  descriptor.path = path;
  descriptor.quantized = checkpoint.is_quantized(path);

  const auto weight_name = path + ".weight";
  if (!dense.contains(weight_name)) {
    throw std::runtime_error(
        "qwen global binding: missing matrix weight: " + weight_name);
  }

  const auto& info = dense.info(weight_name);
  descriptor.rows = checked_rows(info.shape, path);

  if (descriptor.quantized) {
    if (info.dtype != "U32") {
      throw std::runtime_error(
          "qwen global binding: packed affine weight must be U32: " + path);
    }

    descriptor.quant = checkpoint.quant_spec_for(path);
    if (!descriptor.quant.has_value()) {
      throw std::runtime_error(
          "qwen global binding: quantized matrix has no quantization spec: " +
          path);
    }

    const auto bits = descriptor.quant->bits;
    if (bits == 0U || (32U % bits) != 0U) {
      throw std::runtime_error(
          "qwen global binding: invalid quantization bits: " + path);
    }

    const auto per_word = 32U / bits;
    if (info.shape[1] >
        std::numeric_limits<std::size_t>::max() / per_word) {
      throw std::runtime_error(
          "qwen global binding: logical matrix width overflows size_t: " +
          path);
    }
    descriptor.cols = info.shape[1] * per_word;

    const auto scales_name = path + ".scales";
    const auto biases_name = path + ".biases";
    if (!dense.contains(scales_name) || !dense.contains(biases_name)) {
      throw std::runtime_error(
          "qwen global binding: incomplete affine matrix triplet: " + path);
    }

    const auto& scales = dense.info(scales_name);
    const auto& biases = dense.info(biases_name);
    if (scales.shape.size() != 2U ||
        biases.shape.size() != 2U ||
        scales.shape != biases.shape ||
        scales.shape[0] != descriptor.rows ||
        descriptor.cols % descriptor.quant->group_size != 0U ||
        scales.shape[1] !=
            descriptor.cols / descriptor.quant->group_size) {
      throw std::runtime_error(
          "qwen global binding: affine metadata geometry mismatch: " + path);
    }
  } else {
    if (info.dtype != "F32" &&
        info.dtype != "F16" &&
        info.dtype != "BF16") {
      throw std::runtime_error(
          "qwen global binding: unsupported plain matrix dtype: " + path);
    }
    descriptor.cols = info.shape[1];
  }

  return descriptor;
}

void require_geometry(
    const QwenCheckpointMatrixDescriptor& matrix,
    std::size_t rows,
    std::size_t cols,
    std::string_view label) {
  if (matrix.rows != rows || matrix.cols != cols) {
    throw std::runtime_error(
        "qwen global binding: " + std::string(label) +
        " matrix shape mismatch");
  }
}

}  // namespace

QwenGlobalCheckpointBinding bind_qwen3_next_global_checkpoint(
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config) {
  if (config.hidden_size == 0U) {
    throw std::runtime_error(
        "qwen global binding: hidden_size must be non-zero");
  }
  if (config.vocab_size == 0U) {
    throw std::runtime_error(
        "qwen global binding: vocab_size is required for model-global tensors");
  }

  QwenGlobalCheckpointBinding binding{
      .hidden_size = config.hidden_size,
      .vocab_size = config.vocab_size,
      .tie_word_embeddings = config.tie_word_embeddings,
      .embed_tokens =
          describe_matrix(checkpoint, "model.embed_tokens"),
      .final_norm =
          checkpoint.dense().read_floats("model.norm.weight"),
      .lm_head =
          describe_matrix(
              checkpoint,
              config.tie_word_embeddings
                  ? "model.embed_tokens"
                  : "lm_head"),
  };

  require_geometry(
      binding.embed_tokens,
      config.vocab_size,
      config.hidden_size,
      "model.embed_tokens");
  if (binding.final_norm.size() != config.hidden_size) {
    throw std::runtime_error(
        "qwen global binding: model.norm.weight size mismatch");
  }
  require_geometry(
      binding.lm_head,
      config.vocab_size,
      config.hidden_size,
      config.tie_word_embeddings
          ? "tied model.embed_tokens"
          : "lm_head");

  return binding;
}

}  // namespace orbi::streammoe
