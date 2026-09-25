#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/model/checkpoint_deltanet_sublayer.hpp"
#include "orbi/streammoe/model/checkpoint_moe_sublayer.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenCheckpointLinearDecoderLayerResult {
  bool executed{};
  QwenCheckpointDeltaNetSublayerResult attention_branch;
  QwenCheckpointMoeSublayerResult moe_branch;
  std::vector<float> values;
  std::string diagnostic;
};

/// Complete correctness-first single-token Qwen3-Next decoder layer for a
/// linear-attention (Gated DeltaNet) layer:
///
///   h   = x + DeltaNet(RMSNorm(x))
///   out = h + SparseMoE(RMSNorm(h))
class QwenCheckpointLinearDecoderLayer {
 public:
  QwenCheckpointLinearDecoderLayer();
  ~QwenCheckpointLinearDecoderLayer();

  QwenCheckpointLinearDecoderLayer(
      const QwenCheckpointLinearDecoderLayer&) = delete;
  QwenCheckpointLinearDecoderLayer& operator=(
      const QwenCheckpointLinearDecoderLayer&) = delete;

  QwenCheckpointLinearDecoderLayer(
      QwenCheckpointLinearDecoderLayer&&) noexcept;
  QwenCheckpointLinearDecoderLayer& operator=(
      QwenCheckpointLinearDecoderLayer&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointLinearDecoderLayer> create(
      VulkanComputeContext& context,
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      float rms_eps = 1e-6F,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] const QwenGatedDeltaNetState& delta_state() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenCheckpointLinearDecoderLayerResult run(
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::span<const float> hidden) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointLinearDecoderLayer(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
