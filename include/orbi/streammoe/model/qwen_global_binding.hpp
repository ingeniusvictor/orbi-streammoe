#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenCheckpointMatrixDescriptor {
  std::string path;
  std::size_t rows{};
  std::size_t cols{};
  bool quantized{};
  std::optional<MlxAffineQuantSpec> quant;
};

struct QwenGlobalCheckpointBinding {
  std::size_t hidden_size{};
  std::size_t vocab_size{};
  bool tie_word_embeddings{};

  QwenCheckpointMatrixDescriptor embed_tokens;
  std::vector<float> final_norm;
  QwenCheckpointMatrixDescriptor lm_head;
};

[[nodiscard]] QwenGlobalCheckpointBinding
bind_qwen3_next_global_checkpoint(
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config);

}  // namespace orbi::streammoe
