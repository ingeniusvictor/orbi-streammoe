#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

const QpackSection& section(
    const QpackExpertGeometry& geometry,
    const std::string& name) {
  for (const auto& item : geometry.sections) {
    if (item.name == name) return item;
  }
  throw std::runtime_error("missing section: " + name);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_qpack_geometry_tests <model-dir>");
    }

    const auto geometry = derive_qpack_expert_geometry(
        std::filesystem::path(argv[1]),
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});

    require(geometry.hidden_size == 2048U, "official hidden size mismatch");
    require(
        geometry.moe_intermediate_size == 512U,
        "official MoE intermediate size mismatch");
    require(geometry.expert_count == 512U, "official expert count mismatch");
    require(geometry.layer_count == 48U, "official layer count mismatch");

    const auto& gate_w = section(geometry, "gate_proj.weight");
    const auto& gate_s = section(geometry, "gate_proj.scales");
    const auto& up_w = section(geometry, "up_proj.weight");
    const auto& down_w = section(geometry, "down_proj.weight");
    const auto& down_s = section(geometry, "down_proj.scales");

    require(
        gate_w.shape == std::vector<std::size_t>({512U, 256U}),
        "official gate packed shape mismatch");
    require(
        gate_s.shape == std::vector<std::size_t>({512U, 32U}),
        "official gate group shape mismatch");
    require(
        up_w.shape == std::vector<std::size_t>({512U, 256U}),
        "official up packed shape mismatch");
    require(
        down_w.shape == std::vector<std::size_t>({2048U, 64U}),
        "official down packed shape mismatch");
    require(
        down_s.shape == std::vector<std::size_t>({2048U, 8U}),
        "official down group shape mismatch");

    require(
        geometry.expert_stride == 1966080ULL,
        "official expert stride mismatch");
    require(
        geometry.layer_bytes == 1006632960ULL,
        "official per-layer expert bytes mismatch");
    require(
        geometry.all_layers_bytes == 48318382080ULL,
        "official all-layer expert bytes mismatch");

    std::cout
        << "OSM-39B official QPACK expert geometry: PASS\n"
        << "  hidden_size=" << geometry.hidden_size << "\n"
        << "  moe_intermediate_size=" << geometry.moe_intermediate_size << "\n"
        << "  quant_bits=" << geometry.quantization.bits << "\n"
        << "  quant_group_size=" << geometry.quantization.group_size << "\n"
        << "  expert_stride=" << geometry.expert_stride << "\n"
        << "  layer_bytes=" << geometry.layer_bytes << "\n"
        << "  all_layers_bytes=" << geometry.all_layers_bytes << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39B official QPACK expert geometry: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
