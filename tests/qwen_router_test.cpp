#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/model/arch_config.hpp"
#include "orbi/streammoe/model/qwen_router.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    float actual,
    float expected,
    float tolerance,
    const std::string& message) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(
        message + ": actual=" + std::to_string(actual) +
        " expected=" + std::to_string(expected));
  }
}

float pick_sum(const QwenRouterResult& result) {
  float sum = 0.0F;
  for (const auto& pick : result.picks) sum += pick.weight;
  return sum;
}

}  // namespace

int main() {
  try {
    require(
        kQwen3Next80BA3B.expert_count == 512U,
        "Qwen3-Next-80B expert count contract changed");
    require(
        kQwen3Next80BA3B.expert_top_k == 10U,
        "Qwen3-Next-80B Top-K contract changed");
    require(
        kQwen3Next80BA3B.norm_topk_prob,
        "Qwen3-Next-80B must normalize selected router probabilities");

    const std::vector<float> hidden{1.0F};
    const std::vector<float> weight{
        1.0F,
        2.0F,
        3.0F,
        4.0F,
    };

    const QwenRouterConfig normalized_config{
        .hidden_size = 1,
        .expert_count = 4,
        .top_k = 2,
        .norm_topk_prob = true,
    };

    const auto normalized =
        route_qwen_top_k_cpu(hidden, weight, normalized_config);

    require(normalized.logits.size() == 4U, "router logits size mismatch");
    require(normalized.picks.size() == 2U, "router pick count mismatch");
    require(normalized.picks[0].expert == 3U, "top-1 expert mismatch");
    require(normalized.picks[1].expert == 2U, "top-2 expert mismatch");

    const float e1 = std::exp(1.0F);
    const float e2 = std::exp(2.0F);
    const float e3 = std::exp(3.0F);
    const float e4 = std::exp(4.0F);
    const float global_sum = e1 + e2 + e3 + e4;
    const float selected_mass = (e4 + e3) / global_sum;

    require_close(
        normalized.selected_probability_mass,
        selected_mass,
        1e-6F,
        "selected global-softmax mass mismatch");
    require_close(
        normalized.picks[0].weight,
        e4 / (e4 + e3),
        1e-6F,
        "renormalized top-1 weight mismatch");
    require_close(
        normalized.picks[1].weight,
        e3 / (e4 + e3),
        1e-6F,
        "renormalized top-2 weight mismatch");
    require_close(
        pick_sum(normalized),
        1.0F,
        1e-6F,
        "renormalized routed weights must sum to one");

    auto raw_config = normalized_config;
    raw_config.norm_topk_prob = false;
    const auto raw =
        route_qwen_top_k_cpu(hidden, weight, raw_config);

    require(raw.picks[0].expert == 3U && raw.picks[1].expert == 2U,
            "normalization policy must not change selected ids");
    require_close(
        raw.picks[0].weight,
        e4 / global_sum,
        1e-6F,
        "raw top-1 probability mismatch");
    require_close(
        raw.picks[1].weight,
        e3 / global_sum,
        1e-6F,
        "raw top-2 probability mismatch");
    require_close(
        pick_sum(raw),
        raw.selected_probability_mass,
        1e-6F,
        "raw selected weights must preserve selected probability mass");
    require(
        raw.selected_probability_mass < 1.0F,
        "raw selected probability mass should remain below one");

    // Deterministic tie contract: equal scores preserve the lower expert id.
    const std::vector<float> tie_weight{
        5.0F,
        5.0F,
        1.0F,
        0.0F,
    };
    const auto tie = route_qwen_top_k_cpu(
        hidden,
        tie_weight,
        QwenRouterConfig{
            .hidden_size = 1,
            .expert_count = 4,
            .top_k = 1,
            .norm_topk_prob = true,
        });
    require(tie.picks.size() == 1U, "tie pick count mismatch");
    require(tie.picks[0].expert == 0U, "lower expert id must win exact tie");
    require_close(tie.picks[0].weight, 1.0F, 1e-6F, "top-1 normalized weight");

    // Exercise the exact target router width and Top-K count without requiring
    // a full 2048-wide fixture. One scalar hidden dimension is enough to
    // certify 512-way selection semantics.
    std::vector<float> target_weights(512);
    for (std::size_t expert = 0; expert < target_weights.size(); ++expert) {
      target_weights[expert] =
          static_cast<float>(expert) * 0.01F;
    }

    const auto target = route_qwen_top_k_cpu(
        hidden,
        target_weights,
        QwenRouterConfig{
            .hidden_size = 1,
            .expert_count = 512,
            .top_k = 10,
            .norm_topk_prob = true,
        });

    require(target.picks.size() == 10U, "target Top-10 count mismatch");
    for (std::size_t i = 0; i < target.picks.size(); ++i) {
      const auto expected_expert =
          static_cast<std::uint32_t>(511U - i);
      require(
          target.picks[i].expert == expected_expert,
          "target 512-way router selected unexpected expert");
    }
    require_close(
        pick_sum(target),
        1.0F,
        1e-5F,
        "target Top-10 normalized weights must sum to one");

    bool rejected_shape = false;
    try {
      const std::vector<float> bad_weight{1.0F, 2.0F};
      (void)route_qwen_top_k_cpu(
          hidden,
          bad_weight,
          normalized_config);
    } catch (const std::invalid_argument&) {
      rejected_shape = true;
    }
    require(rejected_shape, "router must reject invalid weight shape");

    std::cout
        << "OSM-22 Qwen router Top-K: PASS\n"
        << "  qwen3_next_80b experts=512 top_k=10 norm_topk_prob=true\n"
        << "  normalized_top2="
        << normalized.picks[0].expert << ":"
        << normalized.picks[0].weight << ","
        << normalized.picks[1].expert << ":"
        << normalized.picks[1].weight << "\n"
        << "  selected_mass=" << normalized.selected_probability_mass << "\n"
        << "  tie_break=lower_expert_id PASS\n"
        << "  target_512_top10 PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-22 Qwen router Top-K: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
