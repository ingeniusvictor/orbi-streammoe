#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/conversion/qpack_plan.hpp"

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

std::string config() {
  return R"JSON({
    "model_type":"qwen3_next",
    "hidden_size":8,
    "vocab_size":16,
    "num_hidden_layers":4,
    "full_attention_interval":4,
    "num_experts":3,
    "num_experts_per_tok":2
  })JSON";
}

std::string index() {
  return R"JSON({
    "metadata":{"total_size":1234},
    "weight_map":{
      "model.norm.weight":"model-00001-of-00001.safetensors",
      "model.embed_tokens.weight":"model-00001-of-00001.safetensors",
      "lm_head.weight":"model-00001-of-00001.safetensors",

      "model.layers.0.input_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.post_attention_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.mlp.gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.mlp.shared_expert.gate_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.mlp.shared_expert.up_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.mlp.shared_expert.down_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.mlp.shared_expert_gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.mlp.experts.gate_up_proj":"model-00001-of-00001.safetensors",
      "model.layers.0.mlp.experts.down_proj":"model-00001-of-00001.safetensors",
      "model.layers.0.linear_attn.conv1d.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.linear_attn.dt_bias":"model-00001-of-00001.safetensors",
      "model.layers.0.linear_attn.A_log":"model-00001-of-00001.safetensors",
      "model.layers.0.linear_attn.norm.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.linear_attn.in_proj_qkvz.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.linear_attn.in_proj_ba.weight":"model-00001-of-00001.safetensors",
      "model.layers.0.linear_attn.out_proj.weight":"model-00001-of-00001.safetensors",

      "model.layers.1.input_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.post_attention_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.mlp.gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.mlp.shared_expert.gate_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.mlp.shared_expert.up_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.mlp.shared_expert.down_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.mlp.shared_expert_gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.mlp.experts.gate_up_proj":"model-00001-of-00001.safetensors",
      "model.layers.1.mlp.experts.down_proj":"model-00001-of-00001.safetensors",
      "model.layers.1.linear_attn.conv1d.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.linear_attn.dt_bias":"model-00001-of-00001.safetensors",
      "model.layers.1.linear_attn.A_log":"model-00001-of-00001.safetensors",
      "model.layers.1.linear_attn.norm.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.linear_attn.in_proj_qkvz.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.linear_attn.in_proj_ba.weight":"model-00001-of-00001.safetensors",
      "model.layers.1.linear_attn.out_proj.weight":"model-00001-of-00001.safetensors",

      "model.layers.2.input_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.post_attention_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.mlp.gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.mlp.shared_expert.gate_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.mlp.shared_expert.up_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.mlp.shared_expert.down_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.mlp.shared_expert_gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.mlp.experts.gate_up_proj":"model-00001-of-00001.safetensors",
      "model.layers.2.mlp.experts.down_proj":"model-00001-of-00001.safetensors",
      "model.layers.2.linear_attn.conv1d.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.linear_attn.dt_bias":"model-00001-of-00001.safetensors",
      "model.layers.2.linear_attn.A_log":"model-00001-of-00001.safetensors",
      "model.layers.2.linear_attn.norm.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.linear_attn.in_proj_qkvz.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.linear_attn.in_proj_ba.weight":"model-00001-of-00001.safetensors",
      "model.layers.2.linear_attn.out_proj.weight":"model-00001-of-00001.safetensors",

      "model.layers.3.input_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.post_attention_layernorm.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.mlp.gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.mlp.shared_expert.gate_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.mlp.shared_expert.up_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.mlp.shared_expert.down_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.mlp.shared_expert_gate.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.mlp.experts.gate_up_proj":"model-00001-of-00001.safetensors",
      "model.layers.3.mlp.experts.down_proj":"model-00001-of-00001.safetensors",
      "model.layers.3.self_attn.q_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.self_attn.k_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.self_attn.v_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.self_attn.o_proj.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.self_attn.q_norm.weight":"model-00001-of-00001.safetensors",
      "model.layers.3.self_attn.k_norm.weight":"model-00001-of-00001.safetensors"
    }
  })JSON";
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm39a-qpack-plan";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    write_text(root / "config.json", config());
    write_text(root / "model.safetensors.index.json", index());

    const auto plan = build_qpack_conversion_plan(root);
    require(plan.layer_count == 4U, "layer count mismatch");
    require(plan.expert_count == 3U, "expert count mismatch");
    require(plan.count(QpackConversionClass::global_dense) == 3U,
            "global tensor count mismatch");
    require(
        plan.count(QpackConversionAction::split_packed_gate_up_experts) == 4U,
        "packed gate/up plan count mismatch");
    require(
        plan.count(QpackConversionAction::split_packed_down_experts) == 4U,
        "packed down plan count mismatch");
    require(
        plan.count(QpackConversionAction::direct_expert_quantize) == 0U,
        "fixture must not use direct expert representation");

    for (std::size_t i = 1U; i < plan.entries.size(); ++i) {
      require(
          plan.entries[i - 1U].source_tensor < plan.entries[i].source_tensor,
          "plan must be deterministic and sorted");
    }

    const auto it = std::find_if(
        plan.entries.begin(), plan.entries.end(),
        [](const auto& entry) {
          return entry.source_tensor ==
              "model.layers.3.mlp.experts.gate_up_proj";
        });
    require(it != plan.entries.end(), "layer 3 expert plan missing");
    require(
        it->target_file == "packed_experts/layer_03.bin",
        "expert target file mismatch");
    require(
        it->target_path == "gate_proj+up_proj",
        "expert target section mismatch");

    fs::remove_all(root);
    std::cout
        << "OSM-39A QPACK conversion plan: PASS\n"
        << "  deterministic_order=PASS\n"
        << "  complete_source_inventory=PASS\n"
        << "  global_dense_mapping=PASS\n"
        << "  packed_expert_mapping=PASS\n"
        << "  layer_target_mapping=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-39A QPACK conversion plan: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
