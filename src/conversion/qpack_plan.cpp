#include "orbi/streammoe/conversion/qpack_plan.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "orbi/streammoe/model/sharded_checkpoint_contract.hpp"

namespace orbi::streammoe {
namespace {

std::optional<std::size_t> parse_layer_index(const std::string& name) {
  static const std::regex pattern(R"(^model\.layers\.([0-9]+)\.)");
  std::smatch match;
  if (!std::regex_search(name, match, pattern)) return std::nullopt;
  try {
    return static_cast<std::size_t>(std::stoull(match[1].str()));
  } catch (...) {
    throw std::runtime_error("qpack plan: invalid layer index in tensor: " + name);
  }
}

std::string layer_file(std::size_t layer) {
  std::ostringstream out;
  out << "packed_experts/layer_" << std::setfill('0') << std::setw(2)
      << layer << ".bin";
  return out.str();
}

bool is_vector_f32_target(const std::string& name) {
  if (name == "model.norm.weight") return true;
  return name.ends_with("input_layernorm.weight") ||
         name.ends_with("post_attention_layernorm.weight") ||
         name.ends_with("linear_attn.conv1d.weight") ||
         name.ends_with("linear_attn.dt_bias") ||
         name.ends_with("linear_attn.A_log") ||
         name.ends_with("linear_attn.norm.weight") ||
         name.ends_with("self_attn.q_norm.weight") ||
         name.ends_with("self_attn.k_norm.weight");
}

bool is_global(const std::string& name) {
  return name == "model.embed_tokens.weight" ||
         name == "model.norm.weight" ||
         name == "lm_head.weight";
}

bool is_expert_tensor(const std::string& name) {
  return name.find(".mlp.experts.") != std::string::npos;
}

QpackConversionAction expert_action(const std::string& name) {
  if (name.ends_with(".mlp.experts.gate_up_proj")) {
    return QpackConversionAction::split_packed_gate_up_experts;
  }
  if (name.ends_with(".mlp.experts.down_proj")) {
    return QpackConversionAction::split_packed_down_experts;
  }

  static const std::regex direct(
      R"(\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)\.weight$)");
  if (std::regex_search(name, direct)) {
    return QpackConversionAction::direct_expert_quantize;
  }

  throw std::runtime_error(
      "qpack plan: unsupported routed-expert tensor naming: " + name);
}

std::string expert_target_path(
    const std::string& name,
    QpackConversionAction action) {
  switch (action) {
    case QpackConversionAction::split_packed_gate_up_experts:
      return "gate_proj+up_proj";
    case QpackConversionAction::split_packed_down_experts:
      return "down_proj";
    case QpackConversionAction::direct_expert_quantize: {
      const auto pos = name.rfind('.');
      const auto weight_base =
          pos == std::string::npos ? name : name.substr(0, pos);
      const auto section_pos = weight_base.rfind('.');
      if (section_pos == std::string::npos) {
        throw std::runtime_error("qpack plan: malformed direct expert tensor");
      }
      return weight_base.substr(section_pos + 1U);
    }
    default:
      break;
  }
  throw std::runtime_error("qpack plan: invalid expert action");
}

QpackConversionPlanEntry make_entry(
    const std::string& name,
    const std::string& shard,
    const QwenShardedCheckpointContract& checkpoint) {
  QpackConversionPlanEntry entry;
  entry.source_tensor = name;
  entry.source_shard = shard;

  if (name.starts_with("mtp.")) {
    entry.tensor_class = QpackConversionClass::auxiliary_mtp;
    entry.action = QpackConversionAction::exclude_auxiliary_mtp;
    entry.target_file = "excluded/mtp";
    entry.target_path = name;
    return entry;
  }

  entry.layer_index = parse_layer_index(name);

  if (entry.layer_index.has_value() &&
      *entry.layer_index >= checkpoint.num_hidden_layers) {
    throw std::runtime_error(
        "qpack plan: tensor layer exceeds checkpoint layer count: " + name);
  }

  if (is_expert_tensor(name)) {
    if (!entry.layer_index.has_value()) {
      throw std::runtime_error(
          "qpack plan: expert tensor has no layer index: " + name);
    }
    entry.tensor_class = QpackConversionClass::routed_expert;
    entry.action = expert_action(name);
    entry.target_file = layer_file(*entry.layer_index);
    entry.target_path = expert_target_path(name, entry.action);
    return entry;
  }

  entry.tensor_class =
      is_global(name)
          ? QpackConversionClass::global_dense
          : QpackConversionClass::layer_dense;
  entry.target_file = "model.safetensors";
  entry.target_path = name;

  if (is_vector_f32_target(name)) {
    entry.action = QpackConversionAction::copy_bf16_to_f32;
    return entry;
  }

  if (!name.ends_with(".weight")) {
    throw std::runtime_error(
        "qpack plan: unsupported non-expert tensor naming: " + name);
  }

  entry.action = QpackConversionAction::affine_quantize;
  entry.target_path.resize(entry.target_path.size() - std::string(".weight").size());
  return entry;
}

}  // namespace

std::size_t QpackConversionPlan::count(
    QpackConversionClass value) const noexcept {
  return static_cast<std::size_t>(std::count_if(
      entries.begin(), entries.end(),
      [value](const auto& entry) { return entry.tensor_class == value; }));
}

std::size_t QpackConversionPlan::count(
    QpackConversionAction value) const noexcept {
  return static_cast<std::size_t>(std::count_if(
      entries.begin(), entries.end(),
      [value](const auto& entry) { return entry.action == value; }));
}

QpackConversionPlan build_qpack_conversion_plan(
    const std::filesystem::path& model_dir) {
  const auto checkpoint = inspect_qwen3_next_sharded_checkpoint(model_dir);

  QpackConversionPlan plan;
  plan.source_checkpoint = model_dir.generic_string();
  plan.layer_count = checkpoint.num_hidden_layers;
  plan.expert_count = checkpoint.num_experts;
  plan.entries.reserve(checkpoint.weight_map.size());

  std::vector<std::pair<std::string, std::string>> weights;
  weights.reserve(checkpoint.weight_map.size());
  for (const auto& item : checkpoint.weight_map) {
    weights.push_back(item);
  }
  std::sort(
      weights.begin(), weights.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; });

  for (const auto& [name, shard] : weights) {
    plan.entries.push_back(make_entry(name, shard, checkpoint));
  }

  validate_qpack_conversion_plan(plan);
  return plan;
}

void validate_qpack_conversion_plan(const QpackConversionPlan& plan) {
  if (plan.layer_count == 0U || plan.expert_count == 0U) {
    throw std::runtime_error("qpack plan: layer/expert count must be non-zero");
  }
  if (plan.entries.empty()) {
    throw std::runtime_error("qpack plan: entries must not be empty");
  }

  std::set<std::string> sources;
  std::string previous;
  for (const auto& entry : plan.entries) {
    if (entry.source_tensor.empty() || entry.source_shard.empty() ||
        entry.target_file.empty() || entry.target_path.empty()) {
      throw std::runtime_error("qpack plan: empty required entry field");
    }
    if (!previous.empty() && entry.source_tensor <= previous) {
      throw std::runtime_error(
          "qpack plan: source tensors must be strictly sorted");
    }
    previous = entry.source_tensor;

    if (!sources.insert(entry.source_tensor).second) {
      throw std::runtime_error(
          "qpack plan: duplicate source tensor: " + entry.source_tensor);
    }

    if (entry.tensor_class == QpackConversionClass::auxiliary_mtp) {
      if (entry.action != QpackConversionAction::exclude_auxiliary_mtp ||
          entry.target_file != "excluded/mtp" ||
          !entry.source_tensor.starts_with("mtp.")) {
        throw std::runtime_error("qpack plan: invalid auxiliary MTP mapping");
      }
    } else if (entry.tensor_class == QpackConversionClass::routed_expert) {
      if (!entry.layer_index.has_value()) {
        throw std::runtime_error("qpack plan: routed expert missing layer index");
      }
      if (*entry.layer_index >= plan.layer_count) {
        throw std::runtime_error("qpack plan: routed expert layer out of range");
      }
      if (!entry.target_file.starts_with("packed_experts/layer_")) {
        throw std::runtime_error("qpack plan: routed expert target file mismatch");
      }
    } else if (entry.target_file != "model.safetensors") {
      throw std::runtime_error("qpack plan: dense tensor target file mismatch");
    }
  }

  if (plan.count(QpackConversionClass::global_dense) != 3U) {
    throw std::runtime_error(
        "qpack plan: exactly three model-global tensors are required");
  }

  const auto packed_gate_up =
      plan.count(QpackConversionAction::split_packed_gate_up_experts);
  const auto packed_down =
      plan.count(QpackConversionAction::split_packed_down_experts);
  const auto direct =
      plan.count(QpackConversionAction::direct_expert_quantize);

  if ((packed_gate_up == 0U) != (packed_down == 0U)) {
    throw std::runtime_error(
        "qpack plan: packed expert gate/up and down inventories disagree");
  }
  if (packed_gate_up != 0U && direct != 0U) {
    throw std::runtime_error(
        "qpack plan: mixed packed and direct expert representations");
  }
  if (packed_gate_up != 0U &&
      (packed_gate_up != plan.layer_count || packed_down != plan.layer_count)) {
    throw std::runtime_error(
        "qpack plan: packed expert tensors must exist once per layer");
  }
  if (direct != 0U) {
    const auto expected =
        plan.layer_count * plan.expert_count * std::size_t{3U};
    if (direct != expected) {
      throw std::runtime_error(
          "qpack plan: direct expert tensor inventory is incomplete");
    }
  }
}

const char* to_string(QpackConversionClass value) noexcept {
  switch (value) {
    case QpackConversionClass::global_dense: return "global_dense";
    case QpackConversionClass::layer_dense: return "layer_dense";
    case QpackConversionClass::routed_expert: return "routed_expert";
    case QpackConversionClass::auxiliary_mtp: return "auxiliary_mtp";
  }
  return "unknown";
}

const char* to_string(QpackConversionAction value) noexcept {
  switch (value) {
    case QpackConversionAction::copy_bf16_to_f32: return "copy_bf16_to_f32";
    case QpackConversionAction::affine_quantize: return "affine_quantize";
    case QpackConversionAction::split_packed_gate_up_experts:
      return "split_packed_gate_up_experts";
    case QpackConversionAction::split_packed_down_experts:
      return "split_packed_down_experts";
    case QpackConversionAction::direct_expert_quantize:
      return "direct_expert_quantize";
    case QpackConversionAction::exclude_auxiliary_mtp:
      return "exclude_auxiliary_mtp";
  }
  return "unknown";
}

}  // namespace orbi::streammoe
