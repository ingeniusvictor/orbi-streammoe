#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/conversion/qpack_package.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_qpack_package_plan_tests <metadata-dir>");
    }

    const auto plan = derive_qpack_expert_package_plan(
        fs::path(argv[1]),
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});

    require(plan.geometry.layer_count == 48U, "official layer count mismatch");
    require(plan.geometry.expert_count == 512U, "official expert count mismatch");
    require(plan.geometry.expert_stride == 1966080ULL, "official stride mismatch");
    require(
        plan.expert_payload_bytes == 48318382080ULL,
        "official expert payload total mismatch");
    require(
        plan.full_attention_interval == 4U,
        "official full-attention interval mismatch");
    require(plan.linear_layers.size() == 48U, "linear layer map size mismatch");
    require(
        std::count(plan.linear_layers.begin(), plan.linear_layers.end(), true) == 36,
        "official DeltaNet layer count mismatch");
    require(
        std::count(plan.linear_layers.begin(), plan.linear_layers.end(), false) == 12,
        "official full-attention layer count mismatch");
    require(
        plan.layer_relative_paths.front() == "packed_experts/layer_00.bin" &&
        plan.layer_relative_paths.back() == "packed_experts/layer_47.bin",
        "official layer path inventory mismatch");

    for (std::size_t layer = 0U; layer < 48U; ++layer) {
      const bool expected_linear = ((layer + 1U) % 4U) != 0U;
      require(
          plan.linear_layers[layer] == expected_linear,
          "official D-D-D-G layer pattern mismatch");
    }

    std::cout
        << "OSM-39G official QPACK expert package plan: PASS\n"
        << "  layers=" << plan.geometry.layer_count << "\n"
        << "  experts_per_layer=" << plan.geometry.expert_count << "\n"
        << "  expert_stride=" << plan.geometry.expert_stride << "\n"
        << "  expert_payload_bytes=" << plan.expert_payload_bytes << "\n"
        << "  linear_layers=36\n"
        << "  full_attention_layers=12\n"
        << "  deterministic_layer_paths=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39G official QPACK expert package plan: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
