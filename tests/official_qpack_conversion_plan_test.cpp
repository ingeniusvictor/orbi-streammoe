#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/conversion/qpack_plan.hpp"

using namespace orbi::streammoe;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_qpack_plan_tests <model-dir>");
    }

    const auto plan =
        build_qpack_conversion_plan(std::filesystem::path(argv[1]));

    require(plan.layer_count == 48U, "official layer count mismatch");
    require(plan.expert_count == 512U, "official expert count mismatch");
    require(plan.count(QpackConversionClass::global_dense) == 3U,
            "official global tensor count mismatch");
    require(
        plan.count(QpackConversionAction::split_packed_gate_up_experts) == 48U,
        "official packed gate/up inventory mismatch");
    require(
        plan.count(QpackConversionAction::split_packed_down_experts) == 48U,
        "official packed down inventory mismatch");
    require(
        plan.count(QpackConversionAction::direct_expert_quantize) == 0U,
        "official checkpoint unexpectedly uses direct expert tensors");

    std::cout
        << "OSM-39A official QPACK conversion plan: PASS\n"
        << "  entries=" << plan.entries.size() << "\n"
        << "  layers=" << plan.layer_count << "\n"
        << "  experts=" << plan.expert_count << "\n"
        << "  global_dense="
        << plan.count(QpackConversionClass::global_dense) << "\n"
        << "  layer_dense="
        << plan.count(QpackConversionClass::layer_dense) << "\n"
        << "  routed_expert="
        << plan.count(QpackConversionClass::routed_expert) << "\n"
        << "  packed_gate_up="
        << plan.count(QpackConversionAction::split_packed_gate_up_experts)
        << "\n"
        << "  packed_down="
        << plan.count(QpackConversionAction::split_packed_down_experts)
        << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39A official QPACK conversion plan: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
