#include "orbi/streammoe/model/qwen_dense_binding.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::filesystem::path validated_config_path(
    const QpackReader& qpack) {
  constexpr const char* kConfig = "config.json";
  const auto it = qpack.manifest().files.find(kConfig);
  if (it == qpack.manifest().files.end()) {
    throw std::runtime_error(
        "qwen dense binding: manifest does not declare config.json");
  }

  const auto path = qpack.container_dir() / kConfig;
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec || actual != it->second) {
    throw std::runtime_error(
        "qwen dense binding: config.json is missing or size-mismatched");
  }
  return path;
}

std::size_t required_size(
    const json& root,
    const char* key) {
  try {
    const auto value = root.at(key).get<std::size_t>();
    if (value == 0U) {
      throw std::runtime_error(
          std::string("qwen dense binding: zero config field: ") + key);
    }
    return value;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("qwen dense binding: malformed config field ") +
        key + ": " + e.what());
  }
}

void require_vector_size(
    const std::vector<float>& values,
    std::size_t expected,
    const std::string& label) {
  if (values.size() != expected) {
    throw std::runtime_error(
        "qwen dense binding: " + label +
        " element count mismatch");
  }
}

void require_matrix(
    const MlxAffineModule& module,
    std::size_t rows,
    std::size_t cols,
    const std::string& label) {
  if (module.logical_shape.size() != 2U ||
      module.logical_shape[0] != rows ||
      module.logical_shape[1] != cols) {
    throw std::runtime_error(
        "qwen dense binding: " + label +
        " logical matrix shape mismatch");
  }
}

std::string layer_prefix(std::size_t layer_index) {
  return "model.layers." + std::to_string(layer_index) + ".";
}

QwenMoeDenseBinding bind_moe(
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    const std::string& p) {
  QwenMoeDenseBinding moe{
      .router =
          checkpoint.read_affine_module(p + "mlp.gate"),
      .shared_gate =
          checkpoint.read_affine_module(
              p + "mlp.shared_expert.gate_proj"),
      .shared_up =
          checkpoint.read_affine_module(
              p + "mlp.shared_expert.up_proj"),
      .shared_down =
          checkpoint.read_affine_module(
              p + "mlp.shared_expert.down_proj"),
      .shared_expert_gate =
          checkpoint.read_affine_module(
              p + "mlp.shared_expert_gate"),
  };

  require_matrix(
      moe.router,
      config.num_experts,
      config.hidden_size,
      "mlp.gate");
  require_matrix(
      moe.shared_gate,
      config.shared_expert_intermediate_size,
      config.hidden_size,
      "shared_expert.gate_proj");
  require_matrix(
      moe.shared_up,
      config.shared_expert_intermediate_size,
      config.hidden_size,
      "shared_expert.up_proj");
  require_matrix(
      moe.shared_down,
      config.hidden_size,
      config.shared_expert_intermediate_size,
      "shared_expert.down_proj");
  require_matrix(
      moe.shared_expert_gate,
      1U,
      config.hidden_size,
      "shared_expert_gate");

  return moe;
}

}  // namespace

Qwen3NextDenseConfig parse_qwen3_next_dense_config(
    const QpackReader& qpack) {
  std::ifstream input(
      validated_config_path(qpack),
      std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "qwen dense binding: unable to open config.json");
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("qwen dense binding: invalid JSON: ") + e.what());
  }

  const auto model_type =
      root.value("model_type", std::string("qwen3_next"));
  if (model_type != "qwen3_next") {
    throw std::runtime_error(
        "qwen dense binding: OSM-26C requires model_type=qwen3_next");
  }

  Qwen3NextDenseConfig config{
      .hidden_size = required_size(root, "hidden_size"),
      .num_hidden_layers =
          required_size(root, "num_hidden_layers"),
      .full_attention_interval =
          required_size(root, "full_attention_interval"),
      .num_attention_heads =
          required_size(root, "num_attention_heads"),
      .num_key_value_heads =
          required_size(root, "num_key_value_heads"),
      .head_dim = required_size(root, "head_dim"),
      .linear_num_value_heads =
          required_size(root, "linear_num_value_heads"),
      .linear_num_key_heads =
          required_size(root, "linear_num_key_heads"),
      .linear_key_head_dim =
          required_size(root, "linear_key_head_dim"),
      .linear_value_head_dim =
          required_size(root, "linear_value_head_dim"),
      .linear_conv_kernel_dim =
          required_size(root, "linear_conv_kernel_dim"),
      .num_experts = required_size(root, "num_experts"),
      .num_experts_per_tok =
          required_size(root, "num_experts_per_tok"),
      .moe_intermediate_size =
          required_size(root, "moe_intermediate_size"),
      .shared_expert_intermediate_size =
          required_size(root, "shared_expert_intermediate_size"),
      .norm_topk_prob = root.value("norm_topk_prob", false),
  };

  if (config.num_experts_per_tok > config.num_experts) {
    throw std::runtime_error(
        "qwen dense binding: num_experts_per_tok exceeds num_experts");
  }
  if (config.linear_num_value_heads %
          config.linear_num_key_heads !=
      0U) {
    throw std::runtime_error(
        "qwen dense binding: value-head count must divide by key-head count");
  }

  return config;
}

QwenDenseLayerBinding bind_qwen3_next_dense_layer(
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    std::size_t layer_index) {
  if (layer_index >= config.num_hidden_layers) {
    throw std::out_of_range(
        "qwen dense binding: layer index out of range");
  }

  const auto p = layer_prefix(layer_index);
  QwenDenseLayerBinding layer{
      .layer_index = layer_index,
      .is_linear = config.is_linear_layer(layer_index),
      .input_norm =
          checkpoint.dense().read_floats(
              p + "input_layernorm.weight"),
      .post_attention_norm =
          checkpoint.dense().read_floats(
              p + "post_attention_layernorm.weight"),
      .moe = bind_moe(checkpoint, config, p),
  };

  require_vector_size(
      layer.input_norm,
      config.hidden_size,
      "input_layernorm.weight");
  require_vector_size(
      layer.post_attention_norm,
      config.hidden_size,
      "post_attention_layernorm.weight");

  if (layer.is_linear) {
    QwenDeltaDenseBinding delta{
        .conv =
            checkpoint.dense().read_floats(
                p + "linear_attn.conv1d.weight"),
        .dt_bias =
            checkpoint.dense().read_floats(
                p + "linear_attn.dt_bias"),
        .a_log =
            checkpoint.dense().read_floats(
                p + "linear_attn.A_log"),
        .norm =
            checkpoint.dense().read_floats(
                p + "linear_attn.norm.weight"),
        .out_proj =
            checkpoint.read_affine_module(
                p + "linear_attn.out_proj"),
        .in_proj_qkvz =
            checkpoint.read_affine_module(
                p + "linear_attn.in_proj_qkvz"),
        .in_proj_ba =
            checkpoint.read_affine_module(
                p + "linear_attn.in_proj_ba"),
    };

    const auto key_dim = config.key_dim();
    const auto value_dim = config.value_dim();

    require_vector_size(
        delta.conv,
        config.conv_dim() * config.linear_conv_kernel_dim,
        "linear_attn.conv1d.weight");
    require_vector_size(
        delta.dt_bias,
        config.linear_num_value_heads,
        "linear_attn.dt_bias");
    require_vector_size(
        delta.a_log,
        config.linear_num_value_heads,
        "linear_attn.A_log");
    require_vector_size(
        delta.norm,
        config.linear_value_head_dim,
        "linear_attn.norm.weight");

    require_matrix(
        delta.out_proj,
        config.hidden_size,
        value_dim,
        "linear_attn.out_proj");
    require_matrix(
        delta.in_proj_qkvz,
        2U * key_dim + 2U * value_dim,
        config.hidden_size,
        "linear_attn.in_proj_qkvz");
    require_matrix(
        delta.in_proj_ba,
        2U * config.linear_num_value_heads,
        config.hidden_size,
        "linear_attn.in_proj_ba");

    layer.delta = std::move(delta);
  } else {
    QwenAttentionDenseBinding attention{
        .q_proj =
            checkpoint.read_affine_module(
                p + "self_attn.q_proj"),
        .k_proj =
            checkpoint.read_affine_module(
                p + "self_attn.k_proj"),
        .v_proj =
            checkpoint.read_affine_module(
                p + "self_attn.v_proj"),
        .o_proj =
            checkpoint.read_affine_module(
                p + "self_attn.o_proj"),
        .q_norm =
            checkpoint.dense().read_floats(
                p + "self_attn.q_norm.weight"),
        .k_norm =
            checkpoint.dense().read_floats(
                p + "self_attn.k_norm.weight"),
    };

    const auto q_dim =
        config.num_attention_heads * config.head_dim;
    const auto kv_dim =
        config.num_key_value_heads * config.head_dim;

    require_matrix(
        attention.q_proj,
        2U * q_dim,
        config.hidden_size,
        "self_attn.q_proj");
    require_matrix(
        attention.k_proj,
        kv_dim,
        config.hidden_size,
        "self_attn.k_proj");
    require_matrix(
        attention.v_proj,
        kv_dim,
        config.hidden_size,
        "self_attn.v_proj");
    require_matrix(
        attention.o_proj,
        config.hidden_size,
        q_dim,
        "self_attn.o_proj");
    require_vector_size(
        attention.q_norm,
        config.head_dim,
        "self_attn.q_norm.weight");
    require_vector_size(
        attention.k_norm,
        config.head_dim,
        "self_attn.k_norm.weight");

    layer.attention = std::move(attention);
  }

  return layer;
}

}  // namespace orbi::streammoe
