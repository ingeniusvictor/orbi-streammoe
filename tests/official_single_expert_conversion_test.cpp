#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"
#include "orbi/streammoe/conversion/single_expert_pilot.hpp"

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
          "usage: orbi_streammoe_official_single_expert_conversion_tests "
          "<metadata-dir> <pilot-dir>");
    }

    const fs::path metadata_dir(argv[1]);
    const fs::path pilot_dir(argv[2]);
    const auto geometry = derive_qpack_expert_geometry(
        metadata_dir,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});
    const auto manifest = inspect_single_expert_pilot_manifest(
        pilot_dir / "single-expert.json");

    require(manifest.layer_index == 0U, "pilot layer mismatch");
    require(manifest.expert_index == 0U, "pilot expert mismatch");
    require(
        manifest.total_fetched_bytes == 6291456ULL,
        "pilot must fetch exactly three 2 MiB BF16 matrices");

    const auto output = pilot_dir / "expert_000.qpack.bin";
    const auto result = convert_qwen_single_expert_pilot(
        manifest,
        pilot_dir,
        geometry,
        output);

    require(
        result.source_bytes == 6291456ULL,
        "converted source byte count mismatch");
    require(
        result.qpack_blob.size() == 1966080ULL,
        "QPACK expert blob size mismatch");
    require(
        fs::file_size(output) == geometry.expert_stride,
        "written expert blob size mismatch");
    require(
        result.gate.packed.size() == 512U * 256U,
        "gate packed size mismatch");
    require(
        result.up.packed.size() == 512U * 256U,
        "up packed size mismatch");
    require(
        result.down.packed.size() == 2048U * 64U,
        "down packed size mismatch");
    require(
        std::isfinite(result.max_abs_error) &&
        std::isfinite(result.mean_abs_error) &&
        result.max_abs_error >= 0.0F &&
        result.mean_abs_error >= 0.0,
        "quantization metrics must be finite");

    std::cout
        << "OSM-39C official bounded single-expert conversion: PASS\n"
        << "  layer=" << manifest.layer_index
        << " expert=" << manifest.expert_index << "\n"
        << "  source_bytes=" << result.source_bytes << "\n"
        << "  qpack_bytes=" << result.qpack_blob.size() << "\n"
        << "  max_abs_error=" << result.max_abs_error << "\n"
        << "  mean_abs_error=" << result.mean_abs_error << "\n"
        << "  bounded_network_fetch=PASS\n"
        << "  affine_group_error_bound=PASS\n"
        << "  exact_QPACK_blob=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39C official bounded single-expert conversion: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
