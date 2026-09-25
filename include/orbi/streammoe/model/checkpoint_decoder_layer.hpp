#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/model/checkpoint_full_attention_decoder_layer.hpp"
#include "orbi/streammoe/model/checkpoint_linear_decoder_layer.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

enum class QwenCheckpointDecoderLayerKind {
  gated_deltanet,
  gated_gqa,
};

[[nodiscard]] std::optional<QwenCheckpointDecoderLayerKind>
qwen_checkpoint_decoder_layer_kind(
    const Qwen3NextDenseConfig& config,
    std::size_t layer_index) noexcept;

struct QwenCheckpointDecoderLayerResult {
  bool executed{};
  QwenCheckpointDecoderLayerKind kind{
      QwenCheckpointDecoderLayerKind::gated_deltanet};
  std::vector<float> values;
  std::string diagnostic;
};

/// Unified Qwen3-Next decoder-layer wrapper. It selects the concrete
/// checkpoint-bound layer family from full_attention_interval.
class QwenCheckpointDecoderLayer {
 public:
  QwenCheckpointDecoderLayer();
  ~QwenCheckpointDecoderLayer();

  QwenCheckpointDecoderLayer(
      const QwenCheckpointDecoderLayer&) = delete;
  QwenCheckpointDecoderLayer& operator=(
      const QwenCheckpointDecoderLayer&) = delete;

  QwenCheckpointDecoderLayer(
      QwenCheckpointDecoderLayer&&) noexcept;
  QwenCheckpointDecoderLayer& operator=(
      QwenCheckpointDecoderLayer&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointDecoderLayer> create(
      VulkanComputeContext& context,
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] QwenCheckpointDecoderLayerKind kind() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;

  [[nodiscard]] const QwenGatedDeltaNetState*
  delta_state() const noexcept;
  [[nodiscard]] const QwenGqaState*
  gqa_state() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenCheckpointDecoderLayerResult run(
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::span<const float> hidden) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointDecoderLayer(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
