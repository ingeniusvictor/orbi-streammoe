#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/model/sharded_checkpoint_contract.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write fixture");
  out << text;
}

std::string make_config() {
  return R"({
    "model_type":"qwen3_next",
    "hidden_size":8,
    "vocab_size":16,
    "num_hidden_layers":4,
    "full_attention_interval":4,
    "num_experts":3,
    "num_experts_per_tok":2
  })";
}

void add_common(
    std::string& map,
    std::size_t layer,
    const std::string& shard,
    bool& first) {
  const auto p = "model.layers." + std::to_string(layer) + ".";
  const std::string names[] = {
      p + "input_layernorm.weight",
      p + "post_attention_layernorm.weight",
      p + "mlp.gate.weight",
      p + "mlp.shared_expert.gate_proj.weight",
      p + "mlp.shared_expert.up_proj.weight",
      p + "mlp.shared_expert.down_proj.weight",
      p + "mlp.shared_expert_gate.weight",
      p + "mlp.experts.gate_up_proj",
      p + "mlp.experts.down_proj",
  };
  for (const auto& name : names) {
    if (!first) map += ",";
    first = false;
    map += "\"" + name + "\":\"" + shard + "\"";
  }
}

void add_linear(
    std::string& map,
    std::size_t layer,
    const std::string& shard,
    bool& first) {
  const auto p =
      "model.layers." + std::to_string(layer) + ".linear_attn.";
  const std::string names[] = {
      p + "conv1d.weight",
      p + "dt_bias",
      p + "A_log",
      p + "norm.weight",
      p + "in_proj_qkvz.weight",
      p + "in_proj_ba.weight",
      p + "out_proj.weight",
  };
  for (const auto& name : names) {
    if (!first) map += ",";
    first = false;
    map += "\"" + name + "\":\"" + shard + "\"";
  }
}

void add_attention(
    std::string& map,
    std::size_t layer,
    const std::string& shard,
    bool& first) {
  const auto p =
      "model.layers." + std::to_string(layer) + ".self_attn.";
  const std::string names[] = {
      p + "q_proj.weight",
      p + "k_proj.weight",
      p + "v_proj.weight",
      p + "o_proj.weight",
      p + "q_norm.weight",
      p + "k_norm.weight",
  };
  for (const auto& name : names) {
    if (!first) map += ",";
    first = false;
    map += "\"" + name + "\":\"" + shard + "\"";
  }
}

std::string make_index(bool omit_lm_head = false) {
  std::string map = "{";
  bool first = true;
  auto add = [&](const std::string& name, const std::string& shard) {
    if (!first) map += ",";
    first = false;
    map += "\"" + name + "\":\"" + shard + "\"";
  };

  add("model.embed_tokens.weight", "model-00001-of-00002.safetensors");
  add("model.norm.weight", "model-00002-of-00002.safetensors");
  if (!omit_lm_head) {
    add("lm_head.weight", "model-00002-of-00002.safetensors");
  }

  for (std::size_t layer = 0U; layer < 4U; ++layer) {
    const auto shard =
        layer < 2U
            ? "model-00001-of-00002.safetensors"
            : "model-00002-of-00002.safetensors";
    add_common(map, layer, shard, first);
    if (layer == 3U) add_attention(map, layer, shard, first);
    else add_linear(map, layer, shard, first);
  }
  map += "}";

  return std::string("{\"metadata\":{\"total_size\":123456},\"weight_map\":") +
      map + "}";
}

fs::path make_fixture(const fs::path& root, bool omit_lm_head = false) {
  fs::remove_all(root);
  fs::create_directories(root);
  write_text(root / "config.json", make_config());
  write_text(
      root / "model.safetensors.index.json",
      make_index(omit_lm_head));
  return root;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm38a-checkpoint-contract";

  try {
    const auto valid = make_fixture(root);
    const auto contract =
        inspect_qwen3_next_sharded_checkpoint(valid);

    require(contract.hidden_size == 8U, "hidden size mismatch");
    require(contract.vocab_size == 16U, "vocab size mismatch");
    require(contract.num_hidden_layers == 4U, "layer count mismatch");
    require(contract.full_attention_interval == 4U, "interval mismatch");
    require(contract.num_experts == 3U, "expert count mismatch");
    require(contract.num_experts_per_tok == 2U, "top-k mismatch");
    require(contract.shard_count == 2U, "shard count mismatch");
    require(contract.tensor_count > 50U, "tensor inventory unexpectedly small");
    require(contract.total_size_bytes == 123456U, "total size mismatch");
    require(contract.is_linear_layer(0U), "layer 0 must be linear");
    require(!contract.is_linear_layer(3U), "layer 3 must be full attention");

    make_fixture(root, true);
    bool missing_global = false;
    try {
      (void)inspect_qwen3_next_sharded_checkpoint(root);
    } catch (const std::exception&) {
      missing_global = true;
    }
    require(missing_global, "missing LM head must fail");

    make_fixture(root);
    write_text(
        root / "config.json",
        R"({"model_type":"other","hidden_size":8,"vocab_size":16,"num_hidden_layers":4,"full_attention_interval":4,"num_experts":3,"num_experts_per_tok":2})");
    bool wrong_model = false;
    try {
      (void)inspect_qwen3_next_sharded_checkpoint(root);
    } catch (const std::exception&) {
      wrong_model = true;
    }
    require(wrong_model, "wrong model type must fail");

    make_fixture(root);
    auto bad_index = make_index();
    const std::string needle = "model-00002-of-00002.safetensors";
    const auto pos = bad_index.find(needle);
    require(pos != std::string::npos, "fixture shard not found");
    bad_index.replace(pos, needle.size(), "bad-shard.bin");
    write_text(root / "model.safetensors.index.json", bad_index);
    bool bad_shard = false;
    try {
      (void)inspect_qwen3_next_sharded_checkpoint(root);
    } catch (const std::exception&) {
      bad_shard = true;
    }
    require(bad_shard, "invalid shard name must fail");

    fs::remove_all(root);
    std::cout
        << "OSM-38A sharded checkpoint contract: PASS\n"
        << "  config_contract=PASS\n"
        << "  D-D-D-G_inventory=PASS\n"
        << "  final_layer_guard=PASS\n"
        << "  shard_inventory=PASS\n"
        << "  malformed_guards=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-38A sharded checkpoint contract: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
