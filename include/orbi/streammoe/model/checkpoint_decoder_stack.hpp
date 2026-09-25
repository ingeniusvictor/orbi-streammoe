#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/model/checkpoint_decoder_layer.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenCheckpointDecoderStackResult {
  bool executed{};
  std::size_t layers_executed{};
  std::vector<float> values;
  std::string diagnostic;
};

/// Correctness-first multi-layer Qwen3-Next decoder stack.
///
/// Every layer is bound from the same checkpoint/config and selected through
/// OSM-33's topology-aware decoder-layer wrapper.
class QwenCheckpointDecoderStack {
 public:
  QwenCheckpointDecoderStack();
  ~QwenCheckpointDecoderStack();

  QwenCheckpointDecoderStack(
      const QwenCheckpointDecoderStack&) = delete;
  QwenCheckpointDecoderStack& operator=(
      const QwenCheckpointDecoderStack&) = delete;

  QwenCheckpointDecoderStack(
      QwenCheckpointDecoderStack&&) noexcept;
  QwenCheckpointDecoderStack& operator=(
      QwenCheckpointDecoderStack&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointDecoderStack> create(
      VulkanComputeContext& context,
      const QpackMlxCheckpoint& checkpoint,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_count() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t delta_layer_count() const noexcept;
  [[nodiscard]] std::size_t gqa_layer_count() const noexcept;

  [[nodiscard]] const QwenCheckpointDecoderLayer*
  layer(std::size_t index) const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenCheckpointDecoderStackResult run(
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::span<const float> hidden) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointDecoderStack(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
