#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "orbi/streammoe/model/gqa_cpu.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

class QwenCheckpointGqa {
 public:
  QwenCheckpointGqa();
  ~QwenCheckpointGqa();

  QwenCheckpointGqa(const QwenCheckpointGqa&) = delete;
  QwenCheckpointGqa& operator=(const QwenCheckpointGqa&) = delete;

  QwenCheckpointGqa(QwenCheckpointGqa&&) noexcept;
  QwenCheckpointGqa& operator=(QwenCheckpointGqa&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointGqa> create(
      const QwenDenseLayerBinding& binding,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t layer_index() const noexcept;
  [[nodiscard]] const QwenGqaConfig& config() const noexcept;
  [[nodiscard]] const QwenGqaState& state() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenGqaStepResult run(
      std::span<const float> normalized_hidden) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointGqa(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
