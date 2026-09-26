#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/model/shard_header_contract.hpp"
#include "orbi/streammoe/model/sharded_checkpoint_contract.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_shape(
    const SafetensorsRangeTensorProbe& tensor,
    std::vector<std::size_t> expected,
    const std::string& label) {
  require(tensor.shape == expected, label + " shape mismatch");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_shard_range_tests "
          "<metadata-dir> <range-manifest>");
    }

    const auto checkpoint =
        inspect_qwen3_next_sharded_checkpoint(
            std::filesystem::path(argv[1]));
    const auto manifest =
        inspect_safetensors_range_manifest(
            std::filesystem::path(argv[2]));
    validate_qwen_shard_range_pilot(manifest, checkpoint);

    require(
        manifest.model == "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "official range pilot model mismatch");
    require(
        manifest.snapshot == "f5e99a3698d364cf77584543481b778afee26177",
        "official range pilot snapshot mismatch");
    require(
        manifest.selected_tensors.size() == 14U,
        "official selected tensor count mismatch");
    require(
        !manifest.shards.empty() && manifest.shards.size() <= 10U,
        "official range pilot shard count unexpected");
    require(
        manifest.total_fetched_bytes < 64ULL * 1024ULL * 1024ULL,
        "official range pilot fetched too much data");

    for (const auto& name : manifest.selected_tensors) {
      const auto& tensor = manifest.tensor(name);
      require(tensor.dtype == "BF16", "official tensor must be BF16: " + name);
      require(!tensor.shape.empty(), "official tensor shape must be non-empty: " + name);
      require(tensor.byte_size() > 0U, "official tensor range must be non-empty: " + name);
    }

    require_shape(
        manifest.tensor("model.embed_tokens.weight"),
        {151936U, 2048U},
        "embedding");
    require_shape(
        manifest.tensor("model.norm.weight"),
        {2048U},
        "final norm");
    require_shape(
        manifest.tensor("lm_head.weight"),
        {151936U, 2048U},
        "LM head");
    require_shape(
        manifest.tensor("model.layers.0.input_layernorm.weight"),
        {2048U},
        "layer-0 input norm");
    require_shape(
        manifest.tensor("model.layers.0.mlp.experts.1.gate_proj.weight"),
        {512U, 2048U},
        "layer-0 expert-1 gate");
    require_shape(
        manifest.tensor("model.layers.0.mlp.experts.1.up_proj.weight"),
        {512U, 2048U},
        "layer-0 expert-1 up");
    require_shape(
        manifest.tensor("model.layers.0.mlp.experts.1.down_proj.weight"),
        {2048U, 512U},
        "layer-0 expert-1 down");
    require_shape(
        manifest.tensor("model.layers.0.mlp.shared_expert_gate.weight"),
        {1U, 2048U},
        "layer-0 shared expert gate");

    std::cout
        << "OSM-38C official safetensors shard range pilot: PASS\n"
        << "  selected_tensors=" << manifest.selected_tensors.size() << "\n"
        << "  probed_shards=" << manifest.shards.size() << "\n"
        << "  fetched_bytes=" << manifest.total_fetched_bytes << "\n"
        << "  official_dtype=BF16\n"
        << "  global_shapes=PASS\n"
        << "  index_to_header_mapping=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-38C official safetensors shard range pilot: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
