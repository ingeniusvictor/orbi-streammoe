#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/conversion/qpack_plan.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: orbi_streammoe_qpack_list_dense_sources "
          "<metadata-dir> <output-json>");
    }

    const fs::path metadata_dir(argv[1]);
    const fs::path output_path(argv[2]);
    const auto plan = build_qpack_conversion_plan(metadata_dir);

    json entries = json::array();
    for (const auto& entry : plan.entries) {
      if (entry.tensor_class != QpackConversionClass::global_dense &&
          entry.tensor_class != QpackConversionClass::layer_dense) {
        continue;
      }
      if (entry.target_file != "model.safetensors") {
        throw std::runtime_error(
            "dense source inventory: dense target file drift");
      }
      if (entry.action != QpackConversionAction::copy_bf16_to_f32 &&
          entry.action != QpackConversionAction::affine_quantize) {
        throw std::runtime_error(
            "dense source inventory: unsupported dense action");
      }
      entries.push_back({
          {"source_tensor", entry.source_tensor},
          {"source_shard", entry.source_shard},
          {"tensor_class", to_string(entry.tensor_class)},
          {"action", to_string(entry.action)},
          {"target_path", entry.target_path},
          {"layer_index",
           entry.layer_index.has_value()
               ? json(*entry.layer_index)
               : json(nullptr)},
      });
    }

    if (entries.empty()) {
      throw std::runtime_error("dense source inventory is empty");
    }

    json root = {
        {"schema_version", 1U},
        {"source_checkpoint", plan.source_checkpoint},
        {"layer_count", plan.layer_count},
        {"expert_count", plan.expert_count},
        {"dense_source_count", entries.size()},
        {"entries", entries},
    };

    if (!output_path.parent_path().empty()) {
      fs::create_directories(output_path.parent_path());
    }
    std::ofstream out(output_path);
    if (!out) {
      throw std::runtime_error("unable to create dense source inventory");
    }
    out << root.dump(2) << "\n";

    std::cout
        << "OSM-40D dense source inventory: PASS\n"
        << "  dense_sources=" << entries.size() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-40D dense source inventory: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
