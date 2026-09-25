#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/model/checkpoint_sparse_moe.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenCheckpointMoeSublayerResult {
  bool executed{};
  std::vector<float> normalized;
  QwenSparseMoeResult moe;
  std::vector<float> values;
  std::string diagnostic;
};

/// Correctness-first post-attention Qwen3-Next sublayer:
///
///   y = residual + MoE(RMSNorm(residual))
///
/// The RMSNorm weight and MoE dense weights come from the exact checkpoint
/// binding for the same decoder layer.
class QwenCheckpointMoeSublayer {
 public:
  QwenCheckpointMoeSublayer();
  ~QwenCheckpointMoeSublayer();

  QwenCheckpointMoeSublayer(const QwenCheckpointMoeSublayer&) = delete;
  QwenCheckpointMoeSublayer& operator=(const QwenCheckpointMoeSublayer&) = delete;

  QwenCheckpointMoeSublayer(QwenCheckpointMoeSublayer&&) noexcept;
  QwenCheckpointMoeSublayer& operator=(QwenCheckpointMoeSublayer&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointMoeSublayer> create(
      VulkanComputeContext& context,
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      float rms_eps = 1e-6F,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] float rms_eps() const noexcept;

  [[nodiscard]] QwenCheckpointMoeSublayerResult run(
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::span<const float> residual) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointMoeSublayer(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::vector<float> add_residual_cpu(
    std::span<const float> residual,
    std::span<const float> branch);

}  // namespace orbi::streammoe
