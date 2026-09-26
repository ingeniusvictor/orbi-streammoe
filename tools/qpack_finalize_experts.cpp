#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/conversion/qpack_package.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::runtime_error(
          "usage: orbi_streammoe_qpack_finalize_experts "
          "<metadata-dir> <output-dir> <source-checkpoint> <source-snapshot>");
    }

    const QpackExpertPackageOptions options{
        .model_name = "qwen3_next",
        .source_checkpoint = argv[3],
        .source_snapshot = argv[4],
    };
    const auto result = finalize_qpack_expert_package(
        fs::path(argv[1]),
        fs::path(argv[2]),
        options,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});

    std::cout
        << "OSM-39G QPACK expert package finalization: PASS\n"
        << "  manifest=" << result.manifest_path.string() << "\n"
        << "  layout=" << result.layout_path.string() << "\n"
        << "  provenance=" << result.provenance_path.string() << "\n"
        << "  layers=" << result.layer_count << "\n"
        << "  declared_files=" << result.declared_file_count << "\n"
        << "  expert_payload_bytes=" << result.expert_payload_bytes << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39G QPACK expert package finalization: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
