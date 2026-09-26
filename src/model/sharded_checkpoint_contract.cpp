#include "orbi/streammoe/model/sharded_checkpoint_contract.hpp"

#include <algorithm>
#include <fstream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

json read_json(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "checkpoint contract: unable to open " + path.string());
  }
  json out;
  try {
    input >> out;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "checkpoint contract: invalid JSON in " + path.string() +
        ": " + e.what());
  }
  return out;
}

std::size_t required_size(
    const json& root,
    const char* key) {
  try {
    const auto value = root.at(key).get<std::size_t>();
    if (value == 0U) {
      throw std::runtime_error(
          std::string("checkpoint contract: zero config field: ") + key);
    }
    return value;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("checkpoint contract: malformed config field ") +
        key + ": " + e.what());
  }
}

void require_tensor(
    const std::unordered_map<std::string, std::string>& weight_map,
    const std::string& name) {
  if (!weight_map.contains(name)) {
    throw std::runtime_error(
        "checkpoint contract: missing required tensor: " + name);
  }
}

std::string layer_prefix(std::size_t layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

void require_expert_inventory(
    const std::unordered_map<std::string, std::string>& weight_map,
    const std::string& prefix,
    std::size_t num_experts) {
  const auto packed_gate_up = prefix + "mlp.experts.gate_up_proj";
  const auto packed_down = prefix + "mlp.experts.down_proj";
  if (weight_map.contains(packed_gate_up) ||
      weight_map.contains(packed_down)) {
    require_tensor(weight_map, packed_gate_up);
    require_tensor(weight_map, packed_down);
    return;
  }

  for (std::size_t expert = 0U; expert < num_experts; ++expert) {
    const auto ep =
        prefix + "mlp.experts." + std::to_string(expert) + ".";
    require_tensor(weight_map, ep + "gate_proj.weight");
    require_tensor(weight_map, ep + "up_proj.weight");
    require_tensor(weight_map, ep + "down_proj.weight");
  }
}

void require_common_layer(
    const std::unordered_map<std::string, std::string>& weight_map,
    std::size_t layer,
    std::size_t num_experts) {
  const auto p = layer_prefix(layer);
  require_tensor(weight_map, p + "input_layernorm.weight");
  require_tensor(weight_map, p + "post_attention_layernorm.weight");
  require_tensor(weight_map, p + "mlp.gate.weight");
  require_tensor(weight_map, p + "mlp.shared_expert.gate_proj.weight");
  require_tensor(weight_map, p + "mlp.shared_expert.up_proj.weight");
  require_tensor(weight_map, p + "mlp.shared_expert.down_proj.weight");
  require_tensor(weight_map, p + "mlp.shared_expert_gate.weight");
  require_expert_inventory(weight_map, p, num_experts);
}

void require_linear_layer(
    const std::unordered_map<std::string, std::string>& weight_map,
    std::size_t layer) {
  const auto p = layer_prefix(layer) + "linear_attn.";
  require_tensor(weight_map, p + "conv1d.weight");
  require_tensor(weight_map, p + "dt_bias");
  require_tensor(weight_map, p + "A_log");
  require_tensor(weight_map, p + "norm.weight");
  require_tensor(weight_map, p + "in_proj_qkvz.weight");
  require_tensor(weight_map, p + "in_proj_ba.weight");
  require_tensor(weight_map, p + "out_proj.weight");
}

void require_attention_layer(
    const std::unordered_map<std::string, std::string>& weight_map,
    std::size_t layer) {
  const auto p = layer_prefix(layer) + "self_attn.";
  require_tensor(weight_map, p + "q_proj.weight");
  require_tensor(weight_map, p + "k_proj.weight");
  require_tensor(weight_map, p + "v_proj.weight");
  require_tensor(weight_map, p + "o_proj.weight");
  require_tensor(weight_map, p + "q_norm.weight");
  require_tensor(weight_map, p + "k_norm.weight");
}

}  // namespace

QwenShardedCheckpointContract
inspect_qwen3_next_sharded_checkpoint(
    const std::filesystem::path& model_dir) {
  const auto config = read_json(model_dir / "config.json");
  const auto index =
      read_json(model_dir / "model.safetensors.index.json");

  if (config.value("model_type", std::string{}) != "qwen3_next") {
    throw std::runtime_error(
        "checkpoint contract: model_type must be qwen3_next");
  }

  QwenShardedCheckpointContract out;
  out.hidden_size = required_size(config, "hidden_size");
  out.vocab_size = required_size(config, "vocab_size");
  out.num_hidden_layers = required_size(config, "num_hidden_layers");
  out.full_attention_interval =
      required_size(config, "full_attention_interval");
  out.num_experts = required_size(config, "num_experts");
  out.num_experts_per_tok =
      required_size(config, "num_experts_per_tok");

  if (out.num_experts_per_tok > out.num_experts) {
    throw std::runtime_error(
        "checkpoint contract: num_experts_per_tok exceeds num_experts");
  }
  if (out.num_hidden_layers < out.full_attention_interval ||
      out.num_hidden_layers % out.full_attention_interval != 0U) {
    throw std::runtime_error(
        "checkpoint contract: hidden layers must form complete attention intervals");
  }

  try {
    const auto& metadata = index.at("metadata");
    out.total_size_bytes =
        metadata.at("total_size").get<std::uint64_t>();
    if (out.total_size_bytes == 0U) {
      throw std::runtime_error(
          "checkpoint contract: metadata.total_size must be non-zero");
    }

    const auto& map = index.at("weight_map");
    if (!map.is_object() || map.empty()) {
      throw std::runtime_error(
          "checkpoint contract: weight_map must be a non-empty object");
    }

    std::set<std::string> unique_shards;
    const std::regex shard_pattern(
        R"(^model-[0-9]{5}-of-[0-9]{5}.safetensors$)");

    for (auto it = map.begin(); it != map.end(); ++it) {
      if (!it.value().is_string()) {
        throw std::runtime_error(
            "checkpoint contract: weight_map shard must be a string");
      }
      const auto shard = it.value().get<std::string>();
      if (!std::regex_match(shard, shard_pattern)) {
        throw std::runtime_error(
            "checkpoint contract: invalid shard filename: " + shard);
      }
      out.weight_map.emplace(it.key(), shard);
      unique_shards.insert(shard);
    }

    out.tensor_count = out.weight_map.size();
    out.shards.assign(unique_shards.begin(), unique_shards.end());
    out.shard_count = out.shards.size();
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("checkpoint contract: malformed safetensors index: ") +
        e.what());
  }

  if (out.shard_count == 0U) {
    throw std::runtime_error(
        "checkpoint contract: checkpoint has no shards");
  }

  require_tensor(out.weight_map, "model.embed_tokens.weight");
  require_tensor(out.weight_map, "model.norm.weight");
  require_tensor(out.weight_map, "lm_head.weight");

  // Inspect one complete D-D-D-G block from the checkpoint contract. The
  // numerical runtime already certifies repeating this selector for all layers.
  const auto block = std::min(out.full_attention_interval, out.num_hidden_layers);
  for (std::size_t layer = 0U; layer < block; ++layer) {
    require_common_layer(out.weight_map, layer, out.num_experts);
    if (out.is_linear_layer(layer)) {
      require_linear_layer(out.weight_map, layer);
    } else {
      require_attention_layer(out.weight_map, layer);
    }
  }

  // The final layer must exist too, guarding truncated indexes.
  require_common_layer(
      out.weight_map,
      out.num_hidden_layers - 1U,
      out.num_experts);
  if (out.is_linear_layer(out.num_hidden_layers - 1U)) {
    require_linear_layer(out.weight_map, out.num_hidden_layers - 1U);
  } else {
    require_attention_layer(out.weight_map, out.num_hidden_layers - 1U);
  }

  return out;
}

}  // namespace orbi::streammoe
