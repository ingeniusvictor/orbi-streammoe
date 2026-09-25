#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/model/gated_deltanet_cpu.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<float> generated(
    std::size_t count,
    float scale,
    int modulus,
    int shift) {
  std::vector<float> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] =
        static_cast<float>(
            (static_cast<int>(i * 3U) + shift) % modulus -
            modulus / 2) *
        scale;
  }
  return values;
}

void require_close(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance,
    const std::string& label) {
  require(actual.size() == expected.size(), label + " size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > tolerance) {
      throw std::runtime_error(
          label + " mismatch at index " + std::to_string(i));
    }
  }
}

}  // namespace

int main() {
  try {
    const QwenGatedDeltaNetConfig config{
        .hidden_size = 4,
        .num_key_heads = 1,
        .num_value_heads = 2,
        .key_head_dim = 2,
        .value_head_dim = 2,
        .conv_kernel_size = 2,
        .rms_eps = 1e-6F,
    };

    const auto qkvz = generated(12U * 4U, 0.04F, 13, 2);
    const auto ba = generated(4U * 4U, 0.03F, 11, 1);
    const auto conv = generated(8U * 2U, 0.05F, 9, 0);
    const std::vector<float> dt_bias{0.1F, -0.2F};
    const std::vector<float> a_log{
        std::log(0.7F),
        std::log(1.3F),
    };
    const std::vector<float> norm{1.1F, 0.9F};
    const auto out_proj = generated(4U * 4U, 0.035F, 15, 4);

    const QwenGatedDeltaNetWeights weights{
        .in_proj_qkvz = qkvz,
        .in_proj_ba = ba,
        .conv = conv,
        .dt_bias = dt_bias,
        .a_log = a_log,
        .norm = norm,
        .out_proj = out_proj,
    };

    QwenGatedDeltaNetState state;

    const std::vector<float> x1{0.5F, -0.25F, 0.75F, -0.5F};
    const auto first = run_qwen_gated_deltanet_cpu_step(
        x1, weights, config, state);
    require(first.executed, first.diagnostic);
    require_close(
        first.values,
        {-0.01956961F, 0.00510861F, -0.00215625F, 0.00397190F},
        2e-6F,
        "first decode step");

    require(state.conv_tail.size() == 8U, "conv-tail state size mismatch");
    require(state.recurrent.size() == 8U, "recurrent state size mismatch");

    const std::vector<float> x2{-0.125F, 0.625F, -0.375F, 0.25F};
    const auto second = run_qwen_gated_deltanet_cpu_step(
        x2, weights, config, state);
    require(second.executed, second.diagnostic);
    require_close(
        second.values,
        {-0.00224360F, -0.02006283F, 0.02170629F, 0.01140393F},
        2e-6F,
        "second decode step");

    require_close(
        state.recurrent,
        {0.00486176F, -0.00591133F,
         -0.00219204F, 0.00309173F,
         0.00050425F, -0.00254960F,
         -0.00718698F, 0.00870879F},
        3e-6F,
        "persistent recurrent state");

    require_close(
        state.conv_tail,
        {0.015F, 0.0F, -0.015F, -0.095F,
         -0.11F, -0.125F, 0.185F, 0.17F},
        2e-6F,
        "persistent conv tail");

    QwenGatedDeltaNetState bad_state;
    auto bad_config = config;
    bad_config.num_value_heads = 3;
    const auto bad = run_qwen_gated_deltanet_cpu_step(
        x1, weights, bad_config, bad_state);
    require(!bad.executed, "invalid head ratio must be rejected");

    std::cout
        << "OSM-29A Gated DeltaNet CPU decode semantics: PASS\n"
        << "  fused_interleaved_layout=PASS\n"
        << "  causal_conv_state=PASS\n"
        << "  recurrent_delta_state=PASS\n"
        << "  two_token_reference_vector=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-29A Gated DeltaNet CPU decode semantics: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
