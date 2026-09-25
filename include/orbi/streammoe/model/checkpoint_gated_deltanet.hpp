#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "orbi/streammoe/model/gated_deltanet_cpu.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

class QwenCheckpointGatedDeltaNet {
 public:
  QwenCheckpointGatedDeltaNet();
  ~QwenCheckpointGatedDeltaNet();

  QwenCheckpointGatedDeltaNet(const QwenCheckpointGatedDeltaNet&) = delete;
  QwenCheckpointGatedDeltaNet& operator=(const QwenCheckpointGatedDeltaNet&) = delete;

  QwenCheckpointGatedDeltaNet(QwenCheckpointGatedDeltaNet&&) noexcept;
  QwenCheckpointGatedDeltaNet& operator=(QwenCheckpointGatedDeltaNet&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointGatedDeltaNet> create(
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      float rms_eps = 1e-6F,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] const QwenGatedDeltaNetConfig& config() const noexcept;
  [[nodiscard]] const QwenGatedDeltaNetState& state() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenGatedDeltaNetStepResult run(
      std::span<const float> normalized_hidden) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointGatedDeltaNet(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
