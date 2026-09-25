#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/model/checkpoint_gated_deltanet.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenCheckpointDeltaNetSublayerResult {
  bool executed{};
  std::vector<float> normalized;
  QwenGatedDeltaNetStepResult branch;
  std::vector<float> values;
  std::string diagnostic;
};

class QwenCheckpointDeltaNetSublayer {
 public:
  QwenCheckpointDeltaNetSublayer();
  ~QwenCheckpointDeltaNetSublayer();

  QwenCheckpointDeltaNetSublayer(
      const QwenCheckpointDeltaNetSublayer&) = delete;
  QwenCheckpointDeltaNetSublayer& operator=(
      const QwenCheckpointDeltaNetSublayer&) = delete;

  QwenCheckpointDeltaNetSublayer(
      QwenCheckpointDeltaNetSublayer&&) noexcept;
  QwenCheckpointDeltaNetSublayer& operator=(
      QwenCheckpointDeltaNetSublayer&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointDeltaNetSublayer> create(
      VulkanComputeContext& context,
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      float rms_eps = 1e-6F,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] const QwenGatedDeltaNetState& state() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenCheckpointDeltaNetSublayerResult run(
      VulkanComputeContext& context,
      std::span<const float> residual) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointDeltaNetSublayer(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
