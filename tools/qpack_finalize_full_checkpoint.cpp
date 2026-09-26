#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "orbi/streammoe/conversion/full_checkpoint_package.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: orbi_streammoe_finalize_full_checkpoint "
          "<output-dir> <dense-journal-json>");
    }

    const auto result = finalize_full_qpack_checkpoint(
        fs::path(argv[1]),
        fs::path(argv[2]),
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});

    std::cout
        << "OSM-40F full checkpoint readiness: PASS\n"
        << "  manifest=" << result.manifest_path.string() << "\n"
        << "  dense=" << result.dense_path.string() << "\n"
        << "  dense_tensors=" << result.dense_tensor_count << "\n"
        << "  dense_bytes=" << result.dense_file_bytes << "\n"
        << "  expert_payload_bytes=" << result.expert_payload_bytes << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-40F full checkpoint readiness: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
