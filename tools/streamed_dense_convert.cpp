#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "orbi/streammoe/conversion/qpack_plan.hpp"
#include "orbi/streammoe/conversion/streamed_dense_orchestrator.hpp"
#include "orbi/streammoe/conversion/streamed_safetensors_builder.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    if (argc != 6) {
      throw std::runtime_error(
          "usage: orbi_streammoe_streamed_dense_convert "
          "<metadata-dir> <stream-manifest-json> <source-root> "
          "<output-safetensors> <journal-json>");
    }

    const fs::path metadata_dir(argv[1]);
    const fs::path manifest_path(argv[2]);
    const fs::path source_root(argv[3]);
    const fs::path output_path(argv[4]);
    const fs::path journal_path(argv[5]);

    const auto quantization =
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U};
    const auto plan = build_qpack_conversion_plan(metadata_dir);
    const auto manifest =
        inspect_streamed_dense_manifest(manifest_path);
    const auto specs =
        plan_streamed_dense_output_specs(plan, manifest, quantization);

    auto builder = StreamedSafetensorsBuilder::open(
        output_path,
        journal_path,
        specs);
    const auto result = execute_streamed_dense_manifest(
        plan,
        manifest,
        source_root,
        builder,
        quantization);

    if (builder.state().complete()) {
      builder.finalize();
    }

    std::cout
        << "OSM-40D streamed dense conversion CLI: PASS\n"
        << "  tensors=" << result.tensor_count << "\n"
        << "  requested_chunks=" << result.requested_chunks << "\n"
        << "  converted_chunks=" << result.converted_chunks << "\n"
        << "  skipped_chunks=" << result.skipped_chunks << "\n"
        << "  source_bytes=" << result.source_bytes << "\n"
        << "  output_complete="
        << (builder.state().complete() ? "true" : "false") << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-40D streamed dense conversion CLI: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
