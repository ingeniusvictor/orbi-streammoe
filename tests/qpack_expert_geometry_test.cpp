#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write fixture");
  out << text;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm39b-qpack-geometry";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    write_text(
        root / "config.json",
        R"JSON({
          "model_type":"qwen3_next",
          "hidden_size":8,
          "moe_intermediate_size":8,
          "num_experts":3,
          "num_hidden_layers":4
        })JSON");

    const auto geometry = derive_qpack_expert_geometry(
        root,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});

    require(geometry.hidden_size == 8U, "hidden size mismatch");
    require(geometry.moe_intermediate_size == 8U, "intermediate mismatch");
    require(geometry.expert_count == 3U, "expert count mismatch");
    require(geometry.layer_count == 4U, "layer count mismatch");
    require(geometry.sections.size() == 9U, "section count mismatch");

    require(
        geometry.sections[0].name == "gate_proj.weight" &&
        geometry.sections[0].shape == std::vector<std::size_t>({8U, 1U}) &&
        geometry.sections[0].offset == 0U &&
        geometry.sections[0].size == 32U,
        "gate weight geometry mismatch");
    require(
        geometry.sections[1].shape == std::vector<std::size_t>({8U, 2U}) &&
        geometry.sections[1].offset == 32U &&
        geometry.sections[1].size == 64U,
        "gate scale geometry mismatch");
    require(
        geometry.sections[2].offset == 96U &&
        geometry.sections[2].size == 64U,
        "gate bias geometry mismatch");

    require(geometry.expert_stride == 480U, "expert stride mismatch");
    require(geometry.layer_bytes == 1440U, "layer bytes mismatch");
    require(geometry.all_layers_bytes == 5760U, "all-layer bytes mismatch");

    bool bad_bits = false;
    try {
      (void)derive_qpack_expert_geometry(
          root,
          QpackExpertQuantizationSpec{.bits = 3U, .group_size = 4U});
    } catch (const std::exception&) {
      bad_bits = true;
    }
    require(bad_bits, "invalid quant bits must fail");

    bool bad_group = false;
    try {
      (void)derive_qpack_expert_geometry(
          root,
          QpackExpertQuantizationSpec{.bits = 4U, .group_size = 3U});
    } catch (const std::exception&) {
      bad_group = true;
    }
    require(bad_group, "non-divisible group size must fail");

    fs::remove_all(root);
    std::cout
        << "OSM-39B QPACK expert geometry: PASS\n"
        << "  packed_word_geometry=PASS\n"
        << "  affine_group_geometry=PASS\n"
        << "  deterministic_offsets=PASS\n"
        << "  exact_expert_stride=PASS\n"
        << "  exact_layer_bytes=PASS\n"
        << "  invalid_quantization_guards=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-39B QPACK expert geometry: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
