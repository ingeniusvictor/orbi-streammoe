#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/model/checkpoint_gqa.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenCheckpointGqaSublayerResult {
  bool executed{};
  std::vector<float> normalized;
  QwenGqaStepResult branch;
  std::vector<float> values;
  std::string diagnostic;
};

/// Full-attention residual half of a Qwen3-Next decoder layer:
///
///   h = x + GQA(RMSNorm(x))
class QwenCheckpointGqaSublayer {
 public:
  QwenCheckpointGqaSublayer();
  ~QwenCheckpointGqaSublayer();

  QwenCheckpointGqaSublayer(
      const QwenCheckpointGqaSublayer&) = delete;
  QwenCheckpointGqaSublayer& operator=(
      const QwenCheckpointGqaSublayer&) = delete;

  QwenCheckpointGqaSublayer(
      QwenCheckpointGqaSublayer&&) noexcept;
  QwenCheckpointGqaSublayer& operator=(
      QwenCheckpointGqaSublayer&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointGqaSublayer> create(
      VulkanComputeContext& context,
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] const QwenGqaState& state() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenCheckpointGqaSublayerResult run(
      VulkanComputeContext& context,
      std::span<const float> residual) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointGqaSublayer(
      std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
