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
#include "orbi/streammoe/backend/vulkan_resident_shared_expert.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/model/routed_moe.hpp"

namespace orbi::streammoe {

/// Dequantize one MLX affine module into a row-major float matrix.
///
/// This is intentionally a CPU/reference bridge for dense tensors such as the
/// Qwen router and shared-expert scalar gate. Routed expert gate/up/down
/// matrices remain packed and streamed.
[[nodiscard]] std::vector<float> dequantize_mlx_affine_module_cpu(
    const MlxAffineModule& module);

/// Runtime sparse-MoE state created directly from one checkpoint-bound
/// Qwen3-Next dense layer.
///
/// The router is currently dequantized once to a resident CPU float matrix.
/// The shared expert gate/up/down projections are uploaded once to Vulkan.
/// Routed experts continue to be resolved lazily through
/// VulkanResidentExpertCache.
class QwenCheckpointSparseMoeLayer {
 public:
  QwenCheckpointSparseMoeLayer();
  ~QwenCheckpointSparseMoeLayer();

  QwenCheckpointSparseMoeLayer(
      const QwenCheckpointSparseMoeLayer&) = delete;
  QwenCheckpointSparseMoeLayer& operator=(
      const QwenCheckpointSparseMoeLayer&) = delete;

  QwenCheckpointSparseMoeLayer(
      QwenCheckpointSparseMoeLayer&&) noexcept;
  QwenCheckpointSparseMoeLayer& operator=(
      QwenCheckpointSparseMoeLayer&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointSparseMoeLayer> create(
      VulkanComputeContext& context,
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] const QwenRouterConfig& router_config() const noexcept;
  [[nodiscard]] std::span<const float> router_weight() const noexcept;
  [[nodiscard]] const VulkanResidentSharedExpert* shared_expert() const noexcept;

  [[nodiscard]] QwenSparseMoeResult run(
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::span<const float> hidden) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointSparseMoeLayer(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
