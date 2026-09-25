#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace orbi::streammoe {

struct SharedExpertConfig {
  std::size_t hidden_size{};
  std::size_t intermediate_size{};
};

struct SharedExpertWeights {
  std::span<const float> gate_proj;
  std::span<const float> up_proj;
  std::span<const float> down_proj;
  std::span<const float> scalar_gate;
};

struct SharedExpertResult {
  bool executed{};
  float scalar_gate{};
  std::vector<float> values;
  std::string diagnostic;
};

/// CPU reference for Qwen3-Next shared expert:
/// sigmoid(shared_gate · x) * down(SiLU(gate(x)) * up(x))
[[nodiscard]] SharedExpertResult run_shared_expert_cpu(
    std::span<const float> hidden,
    SharedExpertWeights weights,
    SharedExpertConfig config) noexcept;

}  // namespace orbi::streammoe
