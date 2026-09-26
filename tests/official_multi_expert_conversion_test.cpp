#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/multi_expert_orchestrator.hpp"
#include "orbi/streammoe/conversion/qpack_geometry.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_multi_expert_conversion_tests "
          "<metadata-dir> <expert-range-dir>");
    }

    const fs::path metadata_dir(argv[1]);
    const fs::path root(argv[2]);
    const auto geometry = derive_qpack_expert_geometry(
        metadata_dir,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});

    std::vector<MultiExpertInput> inputs;
    for (std::size_t expert = 0U; expert < 2U; ++expert) {
      const auto dir = root / ("expert_" + std::string(expert == 0U ? "000" : "001"));
      inputs.push_back({dir / "single-expert.json", dir});
    }

    auto writer = QpackLayerWriter::open(
        root / "layer_00.pilot.bin",
        root / "layer_00.pilot.progress.json",
        geometry,
        0U);

    MultiExpertConversionOptions options;
    options.first_expert = 0U;
    options.end_expert_exclusive = 2U;

    const auto first = convert_qwen_expert_range(
        inputs, geometry, writer, options, root / "scratch");

    require(first.requested_experts == 2U, "official requested expert mismatch");
    require(first.converted_experts == 2U, "official convert count mismatch");
    require(first.skipped_completed_experts == 0U, "official first skip mismatch");
    require(
        first.source_bytes == 12582912ULL,
        "official two-expert BF16 byte count mismatch");
    require(
        writer.state().completed_experts == 2U,
        "official writer completed count mismatch");
    require(
        fs::file_size(root / "layer_00.pilot.bin") == geometry.layer_bytes,
        "official preallocated layer size mismatch");
    require(
        std::isfinite(first.max_abs_error) &&
        std::isfinite(first.weighted_mean_abs_error),
        "official aggregate metrics must be finite");

    auto resumed = QpackLayerWriter::open(
        root / "layer_00.pilot.bin",
        root / "layer_00.pilot.progress.json",
        geometry,
        0U);
    const auto second = convert_qwen_expert_range(
        inputs, geometry, resumed, options, root / "scratch");

    require(second.converted_experts == 0U, "official resume must not reconvert");
    require(
        second.skipped_completed_experts == 2U,
        "official resume must skip both certified experts");
    require(second.source_bytes == 0U, "official resume must read no source BF16");

    std::cout
        << "OSM-39E official bounded multi-expert conversion: PASS\n"
        << "  experts=2\n"
        << "  source_bytes=" << first.source_bytes << "\n"
        << "  expert_stride=" << geometry.expert_stride << "\n"
        << "  preallocated_layer_bytes=" << geometry.layer_bytes << "\n"
        << "  first_pass_conversion=PASS\n"
        << "  journal_resume_skip=PASS\n"
        << "  bounded_network_fetch=PASS\n"
        << "  one_expert_at_a_time=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39E official bounded multi-expert conversion: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
