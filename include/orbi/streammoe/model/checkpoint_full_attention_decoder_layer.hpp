#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/model/checkpoint_gqa_sublayer.hpp"
#include "orbi/streammoe/model/checkpoint_moe_sublayer.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenCheckpointFullAttentionDecoderLayerResult {
  bool executed{};
  QwenCheckpointGqaSublayerResult attention_branch;
  QwenCheckpointMoeSublayerResult moe_branch;
  std::vector<float> values;
  std::string diagnostic;
};

/// Complete correctness-first single-token Qwen3-Next decoder layer for a
/// full-attention (gated GQA) layer:
///
///   h   = x + GQA(RMSNorm(x))
///   out = h + SparseMoE(RMSNorm(h))
class QwenCheckpointFullAttentionDecoderLayer {
 public:
  QwenCheckpointFullAttentionDecoderLayer();
  ~QwenCheckpointFullAttentionDecoderLayer();

  QwenCheckpointFullAttentionDecoderLayer(
      const QwenCheckpointFullAttentionDecoderLayer&) = delete;
  QwenCheckpointFullAttentionDecoderLayer& operator=(
      const QwenCheckpointFullAttentionDecoderLayer&) = delete;

  QwenCheckpointFullAttentionDecoderLayer(
      QwenCheckpointFullAttentionDecoderLayer&&) noexcept;
  QwenCheckpointFullAttentionDecoderLayer& operator=(
      QwenCheckpointFullAttentionDecoderLayer&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointFullAttentionDecoderLayer>
  create(
      VulkanComputeContext& context,
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] const QwenGqaState& gqa_state() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenCheckpointFullAttentionDecoderLayerResult run(
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::span<const float> hidden) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointFullAttentionDecoderLayer(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
