#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/model/gqa_cpu.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<float> generated(
    std::size_t rows,
    std::size_t cols,
    float scale,
    int modulus,
    int shift) {
  std::vector<float> values(rows * cols);
  for (std::size_t i = 0; i < values.size(); ++i) {
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
          label + " mismatch at index " + std::to_string(i) +
          ": actual=" + std::to_string(actual[i]) +
          " expected=" + std::to_string(expected[i]));
    }
  }
}

}  // namespace

int main() {
  try {
    const QwenGqaConfig config{
        .hidden_size = 8,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 4,
        .partial_rotary_factor = 0.5F,
        .rope_theta = 10000.0F,
        .rms_eps = 1e-6F,
        .max_position_embeddings = 16,
    };

    const auto q_proj = generated(16, 8, 0.025F, 17, 2);
    const auto k_proj = generated(4, 8, 0.03F, 13, 1);
    const auto v_proj = generated(4, 8, 0.035F, 15, 4);
    const auto o_proj = generated(8, 8, 0.02F, 19, 3);
    const std::vector<float> q_norm{1.1F, 0.9F, 1.05F, 0.95F};
    const std::vector<float> k_norm{0.8F, 1.2F, 0.9F, 1.1F};

    const QwenGqaWeights weights{
        .q_proj = q_proj,
        .k_proj = k_proj,
        .v_proj = v_proj,
        .o_proj = o_proj,
        .q_norm = q_norm,
        .k_norm = k_norm,
    };

    QwenGqaState state;

    const std::vector<float> x1{
        0.5F, -0.25F, 0.75F, -0.5F,
        0.125F, -0.625F, 0.25F, 0.375F,
    };
    const auto first = run_qwen_gqa_cpu_step(
        x1, weights, config, state);
    require(first.executed, first.diagnostic);
    require_close(
        first.values,
        {-0.032192412F, 0.016934833F, -0.033163822F, 0.013018835F,
         0.000094564F, 0.021144900F, -0.028953755F, 0.017228902F},
        4e-6F,
        "first GQA token");

    require(state.position == 1U, "first GQA position mismatch");
    require_close(
        state.k_cache,
        {-0.151341217F, 1.333694476F, -0.595906042F, -1.664753388F},
        4e-6F,
        "first K cache");
    require_close(
        state.v_cache,
        {0.0F, -0.131250000F, 0.328125000F, 0.0F},
        2e-7F,
        "first V cache");

    const std::vector<float> x2{
        -0.125F, 0.625F, -0.375F, 0.25F,
        0.5F, 0.125F, -0.75F, 0.875F,
    };
    const auto second = run_qwen_gqa_cpu_step(
        x2, weights, config, state);
    require(second.executed, second.diagnostic);
    require_close(
        second.values,
        {-0.002020269F, 0.013718235F, -0.024726845F, -0.010926682F,
         0.000155870F, 0.017108827F, -0.021336254F, -0.007536091F},
        5e-6F,
        "second GQA token");

    require(state.position == 2U, "second GQA position mismatch");
    require_close(
        state.k_cache,
        {-0.151341217F, 1.333694476F, -0.595906042F, -1.664753388F,
         -0.590488111F, 0.189275858F, 0.299572397F, 2.087021036F},
        5e-6F,
        "two-token K cache");
    require_close(
        state.v_cache,
        {0.0F, -0.131250000F, 0.328125000F, 0.0F,
         0.0F, 0.026250000F, -0.013125000F, 0.144375000F},
        3e-7F,
        "two-token V cache");

    auto bad_config = config;
    bad_config.num_attention_heads = 3;
    QwenGqaState bad_state;
    const auto bad = run_qwen_gqa_cpu_step(
        x1, weights, bad_config, bad_state);
    require(!bad.executed, "invalid GQA head ratio must be rejected");

    auto window_config = config;
    window_config.max_position_embeddings = 1;
    QwenGqaState window_state;
    const auto allowed = run_qwen_gqa_cpu_step(
        x1, weights, window_config, window_state);
    require(allowed.executed, allowed.diagnostic);
    const auto blocked = run_qwen_gqa_cpu_step(
        x2, weights, window_config, window_state);
    require(!blocked.executed, "GQA context-window overflow must fail");

    std::cout
        << "OSM-31A gated GQA CPU decode semantics: PASS\n"
        << "  q_plus_output_gate_layout=PASS\n"
        << "  partial_RoPE=PASS\n"
        << "  grouped_query_mapping=PASS\n"
        << "  persistent_KV_cache=PASS\n"
        << "  two_token_reference_vector=PASS\n"
        << "  context_window_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-31A gated GQA CPU decode semantics: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
