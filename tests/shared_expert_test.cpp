#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/cpu/reference_ops.hpp"
#include "orbi/streammoe/model/shared_expert.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

float max_abs_error(
    const std::vector<float>& a,
    const std::vector<float>& b) {
  require(a.size() == b.size(), "size mismatch");
  float worst = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, std::fabs(a[i] - b[i]));
  }
  return worst;
}

}  // namespace

int main() {
  try {
    constexpr std::size_t hidden = 8;
    constexpr std::size_t inter = 4;

    std::vector<float> x(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
      x[i] = static_cast<float>(static_cast<int>(i) - 3) * 0.25F;
    }

    std::vector<float> gate(inter * hidden);
    std::vector<float> up(inter * hidden);
    std::vector<float> down(hidden * inter);
    std::vector<float> scalar(hidden);

    for (std::size_t i = 0; i < gate.size(); ++i) {
      gate[i] = static_cast<float>(static_cast<int>(i % 9U) - 4) * 0.0625F;
      up[i] = static_cast<float>(static_cast<int>((i * 3U) % 11U) - 5) * 0.05F;
    }
    for (std::size_t i = 0; i < down.size(); ++i) {
      down[i] = static_cast<float>(static_cast<int>((i * 5U) % 13U) - 6) * 0.04F;
    }
    for (std::size_t i = 0; i < scalar.size(); ++i) {
      scalar[i] = static_cast<float>(static_cast<int>(i % 5U) - 2) * 0.125F;
    }

    auto gate_ref = cpu::matvec_row_major(gate, inter, hidden, x);
    const auto up_ref = cpu::matvec_row_major(up, inter, hidden, x);
    cpu::swiglu_inplace(gate_ref, up_ref);
    auto expected = cpu::matvec_row_major(down, hidden, inter, gate_ref);

    float logit = 0.0F;
    for (std::size_t i = 0; i < hidden; ++i) {
      logit += scalar[i] * x[i];
    }
    const float expected_scale = cpu::sigmoid(logit);
    for (auto& v : expected) v *= expected_scale;

    const auto result = run_shared_expert_cpu(
        x,
        SharedExpertWeights{
            .gate_proj = gate,
            .up_proj = up,
            .down_proj = down,
            .scalar_gate = scalar,
        },
        SharedExpertConfig{
            .hidden_size = hidden,
            .intermediate_size = inter,
        });

    require(result.executed, result.diagnostic);
    require(std::fabs(result.scalar_gate - expected_scale) <= 1e-7F,
            "shared gate mismatch");
    require(max_abs_error(result.values, expected) <= 1e-6F,
            "shared expert output mismatch");

    auto bad = run_shared_expert_cpu(
        std::span<const float>(x).first(hidden - 1),
        SharedExpertWeights{
            .gate_proj = gate,
            .up_proj = up,
            .down_proj = down,
            .scalar_gate = scalar,
        },
        SharedExpertConfig{
            .hidden_size = hidden,
            .intermediate_size = inter,
        });
    require(!bad.executed, "invalid hidden shape should fail");

    std::cout
        << "OSM-25A shared expert semantics: PASS\n"
        << "  scalar_gate=" << result.scalar_gate << "\n"
        << "  max_abs_error=" << max_abs_error(result.values, expected) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-25A shared expert semantics: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
