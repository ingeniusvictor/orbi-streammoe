#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/multi_expert_orchestrator.hpp"
#include "orbi/streammoe/conversion/qpack_geometry.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {

std::size_t parse_size(const char* value, const char* label) {
  try {
    const auto parsed = std::stoull(value);
    return static_cast<std::size_t>(parsed);
  } catch (...) {
    throw std::runtime_error(std::string("invalid ") + label);
  }
}

std::string expert_dir_name(std::size_t expert) {
  std::ostringstream out;
  out << "expert_" << std::setfill('0') << std::setw(3) << expert;
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      throw std::runtime_error(
          "usage: orbi_streammoe_qpack_convert_range "
          "<metadata-dir> <expert-root> <layer-file> <journal-file> "
          "<layer> <first-expert> <end-expert-exclusive> <scratch-dir>");
    }

    const fs::path metadata_dir(argv[1]);
    const fs::path expert_root(argv[2]);
    const fs::path layer_file(argv[3]);
    const fs::path journal_file(argv[4]);
    const auto layer = parse_size(argv[5], "layer");
    const auto first = parse_size(argv[6], "first expert");
    const auto end = parse_size(argv[7], "end expert");
    const fs::path scratch_dir(argv[8]);

    const auto geometry = derive_qpack_expert_geometry(
        metadata_dir,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});

    if (layer >= geometry.layer_count ||
        first >= end ||
        end > geometry.expert_count) {
      throw std::runtime_error("requested layer/expert range is out of bounds");
    }

    std::vector<MultiExpertInput> inputs;
    inputs.reserve(end - first);
    for (std::size_t expert = first; expert < end; ++expert) {
      const auto dir = expert_root / expert_dir_name(expert);
      const auto manifest = dir / "single-expert.json";
      if (fs::exists(manifest)) {
        inputs.push_back({manifest, dir});
      }
    }

    auto writer = QpackLayerWriter::open(
        layer_file,
        journal_file,
        geometry,
        layer);

    MultiExpertConversionOptions options;
    options.first_expert = first;
    options.end_expert_exclusive = end;
    options.finalize_if_complete =
        first == 0U && end == geometry.expert_count;

    const auto result = convert_qwen_expert_range(
        inputs,
        geometry,
        writer,
        options,
        scratch_dir);

    std::cout
        << "OSM-39F production conversion CLI: PASS\n"
        << "  layer=" << layer << "\n"
        << "  requested=" << result.requested_experts << "\n"
        << "  converted=" << result.converted_experts << "\n"
        << "  skipped=" << result.skipped_completed_experts << "\n"
        << "  completed_total=" << writer.state().completed_experts << "\n"
        << "  expert_count=" << geometry.expert_count << "\n"
        << "  source_bytes=" << result.source_bytes << "\n"
        << "  max_abs_error=" << result.max_abs_error << "\n"
        << "  weighted_mean_abs_error=" << result.weighted_mean_abs_error
        << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39F production conversion CLI: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
