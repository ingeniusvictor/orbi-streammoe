#include "orbi/streammoe/model/shared_expert.hpp"

#include <cmath>
#include <exception>
#include <string>

#include "orbi/streammoe/cpu/reference_ops.hpp"

namespace orbi::streammoe {

SharedExpertResult run_shared_expert_cpu(
    std::span<const float> hidden,
    SharedExpertWeights weights,
    SharedExpertConfig config) noexcept {
  SharedExpertResult result;

  try {
    if (config.hidden_size == 0U || config.intermediate_size == 0U) {
      result.diagnostic = "shared expert dimensions must be non-zero";
      return result;
    }
    if (hidden.size() != config.hidden_size) {
      result.diagnostic = "shared expert hidden size mismatch";
      return result;
    }

    const auto gate_expected =
        config.intermediate_size * config.hidden_size;
    const auto down_expected =
        config.hidden_size * config.intermediate_size;

    if (weights.gate_proj.size() != gate_expected ||
        weights.up_proj.size() != gate_expected ||
        weights.down_proj.size() != down_expected ||
        weights.scalar_gate.size() != config.hidden_size) {
      result.diagnostic = "shared expert weight geometry mismatch";
      return result;
    }

    auto gate = cpu::matvec_row_major(
        weights.gate_proj,
        config.intermediate_size,
        config.hidden_size,
        hidden);
    const auto up = cpu::matvec_row_major(
        weights.up_proj,
        config.intermediate_size,
        config.hidden_size,
        hidden);

    cpu::swiglu_inplace(gate, up);

    auto values = cpu::matvec_row_major(
        weights.down_proj,
        config.hidden_size,
        config.intermediate_size,
        gate);

    float gate_logit = 0.0F;
    for (std::size_t i = 0; i < config.hidden_size; ++i) {
      gate_logit += weights.scalar_gate[i] * hidden[i];
    }

    const float scale = cpu::sigmoid(gate_logit);
    if (!std::isfinite(scale)) {
      result.diagnostic = "shared expert scalar gate is non-finite";
      return result;
    }

    for (auto& value : values) {
      value *= scale;
    }

    result.executed = true;
    result.scalar_gate = scale;
    result.values = std::move(values);
    result.diagnostic = "Qwen shared expert CPU semantic path executed.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("shared expert CPU exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic = "shared expert CPU encountered an unknown exception";
    return result;
  }
}

}  // namespace orbi::streammoe
