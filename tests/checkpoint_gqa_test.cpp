#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/model/checkpoint_gqa.hpp"
#include "orbi/streammoe/model/checkpoint_sparse_moe.hpp"
#include "orbi/streammoe/model/gqa_cpu.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

struct TensorFixture {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::vector<std::byte> bytes;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void append_u32_le(
    std::vector<std::byte>& out,
    std::uint32_t value) {
  for (std::uint32_t shift = 0; shift < 32U; shift += 8U) {
    out.push_back(
        static_cast<std::byte>((value >> shift) & 0xFFU));
  }
}

void append_u64_le(
    std::ofstream& out,
    std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] =
        static_cast<char>((value >> (8U * i)) & 0xFFU);
  }
  out.write(
      bytes.data(),
      static_cast<std::streamsize>(bytes.size()));
}

void append_f32_le(
    std::vector<std::byte>& out,
    float value) {
  append_u32_le(out, std::bit_cast<std::uint32_t>(value));
}

std::string shape_json(
    const std::vector<std::size_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0U) out << ",";
    out << shape[i];
  }
  out << "]";
  return out.str();
}

void write_safetensors(
    const fs::path& path,
    const std::vector<TensorFixture>& tensors) {
  std::ostringstream header;
  header << "{";

  std::uint64_t offset = 0U;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (i != 0U) header << ",";
    const auto& tensor = tensors[i];
    header
        << "\"" << tensor.name << "\":{"
        << "\"dtype\":\"" << tensor.dtype << "\","
        << "\"shape\":" << shape_json(tensor.shape) << ","
        << "\"data_offsets\":["
        << offset << "," << (offset + tensor.bytes.size()) << "]}";
    offset += tensor.bytes.size();
  }
  header << "}";

  const auto header_text = header.str();
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("unable to create safetensors fixture");
  }
  append_u64_le(out, header_text.size());
  out.write(
      header_text.data(),
      static_cast<std::streamsize>(header_text.size()));
  for (const auto& tensor : tensors) {
    out.write(
        reinterpret_cast<const char*>(tensor.bytes.data()),
        static_cast<std::streamsize>(tensor.bytes.size()));
  }
}

void add_plain(
    std::vector<TensorFixture>& tensors,
    const std::string& name,
    std::vector<std::size_t> shape,
    std::size_t count,
    std::size_t seed) {
  std::vector<std::byte> bytes;
  for (std::size_t i = 0; i < count; ++i) {
    const auto value =
        static_cast<float>(
            static_cast<int>((i + seed) % 13U) - 6) *
        0.0625F;
    append_f32_le(bytes, value);
  }
  tensors.push_back({
      .name = name,
      .dtype = "F32",
      .shape = std::move(shape),
      .bytes = std::move(bytes),
  });
}

void add_affine(
    std::vector<TensorFixture>& tensors,
    const std::string& path,
    std::size_t rows,
    std::size_t logical_cols,
    std::uint32_t bits,
    std::uint32_t group_size,
    std::size_t seed) {
  const auto values_per_word = 32U / bits;
  require(
      logical_cols % values_per_word == 0U,
      "fixture logical cols not packable");
  require(
      logical_cols % group_size == 0U,
      "fixture logical cols not group aligned");

  const auto packed_cols = logical_cols / values_per_word;
  const auto groups = logical_cols / group_size;

  std::vector<std::byte> packed;
  for (std::size_t i = 0; i < rows * packed_cols; ++i) {
    append_u32_le(
        packed,
        static_cast<std::uint32_t>(
            0x01020304U + seed * 17U + i * 0x01010101U));
  }

  std::vector<std::byte> scales;
  std::vector<std::byte> biases;
  for (std::size_t i = 0; i < rows * groups; ++i) {
    append_f32_le(
        scales,
        0.125F *
            static_cast<float>((seed + i) % 7U + 1U));
    append_f32_le(
        biases,
        0.03125F *
            static_cast<float>(
                static_cast<int>((seed + 2U * i) % 9U) - 4));
  }

  tensors.push_back({
      .name = path + ".weight",
      .dtype = "U32",
      .shape = {rows, packed_cols},
      .bytes = std::move(packed),
  });
  tensors.push_back({
      .name = path + ".scales",
      .dtype = "F32",
      .shape = {rows, groups},
      .bytes = std::move(scales),
  });
  tensors.push_back({
      .name = path + ".biases",
      .dtype = "F32",
      .shape = {rows, groups},
      .bytes = std::move(biases),
  });
}

void add_common_layer(
    std::vector<TensorFixture>& tensors,
    std::size_t layer,
    std::size_t seed) {
  const auto p =
      "model.layers." + std::to_string(layer) + ".";

  add_plain(
      tensors,
      p + "input_layernorm.weight",
      {8},
      8,
      seed + 1U);
  add_plain(
      tensors,
      p + "post_attention_layernorm.weight",
      {8},
      8,
      seed + 2U);

  add_affine(
      tensors,
      p + "mlp.gate",
      3,
      8,
      8,
      4,
      seed + 3U);
  add_affine(
      tensors,
      p + "mlp.shared_expert.gate_proj",
      8,
      8,
      4,
      4,
      seed + 4U);
  add_affine(
      tensors,
      p + "mlp.shared_expert.up_proj",
      8,
      8,
      4,
      4,
      seed + 5U);
  add_affine(
      tensors,
      p + "mlp.shared_expert.down_proj",
      8,
      8,
      4,
      4,
      seed + 6U);
  add_affine(
      tensors,
      p + "mlp.shared_expert_gate",
      1,
      8,
      8,
      4,
      seed + 7U);
}

void add_delta_layer(
    std::vector<TensorFixture>& tensors,
    std::size_t layer,
    std::size_t seed) {
  const auto p =
      "model.layers." + std::to_string(layer) + ".linear_attn.";

  add_plain(
      tensors,
      p + "conv1d.weight",
      {24, 2},
      48,
      seed + 1U);
  add_plain(
      tensors,
      p + "dt_bias",
      {1},
      1,
      seed + 2U);
  add_plain(
      tensors,
      p + "A_log",
      {1},
      1,
      seed + 3U);
  add_plain(
      tensors,
      p + "norm.weight",
      {8},
      8,
      seed + 4U);

  add_affine(tensors, p + "out_proj", 8, 8, 4, 4, seed + 5U);
  add_affine(tensors, p + "in_proj_qkvz", 32, 8, 4, 4, seed + 6U);
  add_affine(tensors, p + "in_proj_ba", 2, 8, 4, 4, seed + 7U);
}

void add_attention_layer(
    std::vector<TensorFixture>& tensors,
    std::size_t layer,
    std::size_t seed) {
  const auto p =
      "model.layers." + std::to_string(layer) + ".self_attn.";

  add_affine(tensors, p + "q_proj", 16, 8, 4, 4, seed + 1U);
  add_affine(tensors, p + "k_proj", 8, 8, 4, 4, seed + 2U);
  add_affine(tensors, p + "v_proj", 8, 8, 4, 4, seed + 3U);
  add_affine(tensors, p + "o_proj", 8, 8, 4, 4, seed + 4U);

  add_plain(
      tensors,
      p + "q_norm.weight",
      {8},
      8,
      seed + 5U);
  add_plain(
      tensors,
      p + "k_norm.weight",
      {8},
      8,
      seed + 6U);
}

void write_text(
    const fs::path& path,
    const std::string& value) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write fixture");
  out << value;
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  std::vector<TensorFixture> tensors;
  add_common_layer(tensors, 0, 10);
  add_delta_layer(tensors, 0, 30);
  add_common_layer(tensors, 3, 50);
  add_attention_layer(tensors, 3, 70);
  write_safetensors(root / "model.safetensors", tensors);

  const std::string config =
      "{"
      "\"model_type\":\"qwen3_next\","
      "\"hidden_size\":8,"
      "\"num_hidden_layers\":4,"
      "\"full_attention_interval\":4,"
      "\"num_attention_heads\":1,"
      "\"num_key_value_heads\":1,"
      "\"head_dim\":8,"
      "\"partial_rotary_factor\":0.5,"
      "\"rope_theta\":10000,"
      "\"rms_norm_eps\":0.00001,"
      "\"max_position_embeddings\":8,"
      "\"linear_num_value_heads\":1,"
      "\"linear_num_key_heads\":1,"
      "\"linear_key_head_dim\":8,"
      "\"linear_value_head_dim\":8,"
      "\"linear_conv_kernel_dim\":2,"
      "\"num_experts\":3,"
      "\"num_experts_per_tok\":2,"
      "\"moe_intermediate_size\":8,"
      "\"shared_expert_intermediate_size\":8,"
      "\"norm_topk_prob\":true,"
      "\"quantization\":{"
      "\"group_size\":4,"
      "\"bits\":4,"
      "\"mode\":\"affine\","
      "\"model.layers.0.mlp.gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.0.mlp.shared_expert_gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.3.mlp.gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.3.mlp.shared_expert_gate\":{\"group_size\":4,\"bits\":8}"
      "}"
      "}";
  write_text(root / "config.json", config);

  const std::string layout =
      "{"
      "\"expertCount\":1,"
      "\"layerCount\":1,"
      "\"expertStride\":16,"
      "\"sections\":["
      "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\","
      "\"shape\":[1],\"offset\":0,\"size\":4}"
      "],"
      "\"linearLayers\":[true]"
      "}";
  write_text(root / "packed_experts" / "layout.json", layout);

  std::vector<char> expert_bytes(16, 0);
  std::ofstream layer(
      root / "packed_experts" / "layer_00.bin",
      std::ios::binary);
  layer.write(
      expert_bytes.data(),
      static_cast<std::streamsize>(expert_bytes.size()));
  layer.close();

  const auto dense_size = fs::file_size(root / "model.safetensors");
  const auto config_size = fs::file_size(root / "config.json");
  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");
  const auto layer_size =
      fs::file_size(root / "packed_experts" / "layer_00.bin");

  const std::string manifest =
      "{"
      "\"magic\":\"QPACK\","
      "\"version\":1,"
      "\"modelName\":\"qwen3_next\","
      "\"sourceCheckpoint\":\"osm26c\","
      "\"files\":{"
      "\"model.safetensors\":" + std::to_string(dense_size) + ","
      "\"config.json\":" + std::to_string(config_size) + ","
      "\"packed_experts/layout.json\":" + std::to_string(layout_size) + ","
      "\"packed_experts/layer_00.bin\":" + std::to_string(layer_size) +
      "}"
      "}";

  write_text(root / "manifest.json", manifest);
  return root;
}


float max_abs_error(
    std::span<const float> actual,
    std::span<const float> expected) {
  require(actual.size() == expected.size(), "output size mismatch");
  float worst = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    worst = std::max(worst, std::fabs(actual[i] - expected[i]));
  }
  return worst;
}

}  // namespace


int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm31b-checkpoint-gqa";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);

    require(
        std::fabs(config.partial_rotary_factor - 0.5F) <= 1e-7F,
        "partial rotary factor parse mismatch");
    require(
        std::fabs(config.rope_theta - 10000.0F) <= 1e-4F,
        "rope theta parse mismatch");
    require(
        std::fabs(config.rms_norm_eps - 1e-5F) <= 1e-9F,
        "RMS epsilon parse mismatch");
    require(
        config.max_position_embeddings == 8U,
        "max position parse mismatch");
    require(config.rotary_dims() == 4U, "rotary dims mismatch");

    const auto binding =
        bind_qwen3_next_dense_layer(checkpoint, config, 3);
    require(!binding.is_linear, "layer 3 must be full attention");
    require(binding.attention.has_value(), "GQA binding missing");

    std::string diagnostic;
    auto runtime = QwenCheckpointGqa::create(
        binding,
        config,
        &diagnostic);
    require(runtime.has_value(), diagnostic);
    require(runtime->valid(), "checkpoint GQA runtime must be valid");
    require(runtime->layer_index() == 3U, "checkpoint GQA layer mismatch");

    const auto& attn = *binding.attention;
    const auto q = dequantize_mlx_affine_module_cpu(attn.q_proj);
    const auto k = dequantize_mlx_affine_module_cpu(attn.k_proj);
    const auto v = dequantize_mlx_affine_module_cpu(attn.v_proj);
    const auto o = dequantize_mlx_affine_module_cpu(attn.o_proj);

    const QwenGqaConfig direct_config{
        .hidden_size = config.hidden_size,
        .num_attention_heads = config.num_attention_heads,
        .num_key_value_heads = config.num_key_value_heads,
        .head_dim = config.head_dim,
        .partial_rotary_factor = config.partial_rotary_factor,
        .rope_theta = config.rope_theta,
        .rms_eps = config.rms_norm_eps,
        .max_position_embeddings = config.max_position_embeddings,
    };
    const QwenGqaWeights direct_weights{
        .q_proj = q,
        .k_proj = k,
        .v_proj = v,
        .o_proj = o,
        .q_norm = attn.q_norm,
        .k_norm = attn.k_norm,
    };

    QwenGqaState direct_state;

    const std::vector<float> x1{
        0.5F, -0.25F, 0.75F, -0.5F,
        0.125F, -0.625F, 0.25F, 0.375F,
    };
    const std::vector<float> x2{
        -0.125F, 0.625F, -0.375F, 0.25F,
        0.5F, 0.125F, -0.75F, 0.875F,
    };

    const auto direct1 = run_qwen_gqa_cpu_step(
        x1, direct_weights, direct_config, direct_state);
    require(direct1.executed, direct1.diagnostic);
    const auto wrapped1 = runtime->run(x1);
    require(wrapped1.executed, wrapped1.diagnostic);

    require(
        max_abs_error(wrapped1.values, direct1.values) <= 1e-6F,
        "checkpoint GQA token1 output mismatch");
    require(runtime->state().position == direct_state.position,
            "checkpoint GQA token1 position mismatch");
    require(
        max_abs_error(runtime->state().k_cache, direct_state.k_cache) <= 1e-7F,
        "checkpoint GQA token1 K cache mismatch");
    require(
        max_abs_error(runtime->state().v_cache, direct_state.v_cache) <= 1e-7F,
        "checkpoint GQA token1 V cache mismatch");

    const auto direct2 = run_qwen_gqa_cpu_step(
        x2, direct_weights, direct_config, direct_state);
    require(direct2.executed, direct2.diagnostic);
    const auto wrapped2 = runtime->run(x2);
    require(wrapped2.executed, wrapped2.diagnostic);

    require(
        max_abs_error(wrapped2.values, direct2.values) <= 1e-6F,
        "checkpoint GQA token2 output mismatch");
    require(runtime->state().position == direct_state.position,
            "checkpoint GQA token2 position mismatch");
    require(
        max_abs_error(runtime->state().k_cache, direct_state.k_cache) <= 1e-7F,
        "checkpoint GQA token2 K cache mismatch");
    require(
        max_abs_error(runtime->state().v_cache, direct_state.v_cache) <= 1e-7F,
        "checkpoint GQA token2 V cache mismatch");

    runtime->reset_state();
    require(runtime->state().position == 0U, "checkpoint GQA reset position");
    require(runtime->state().k_cache.empty(), "checkpoint GQA reset K cache");
    require(runtime->state().v_cache.empty(), "checkpoint GQA reset V cache");

    QwenGqaState fresh;
    const auto fresh1 = run_qwen_gqa_cpu_step(
        x1, direct_weights, direct_config, fresh);
    require(fresh1.executed, fresh1.diagnostic);
    const auto reset1 = runtime->run(x1);
    require(reset1.executed, reset1.diagnostic);
    require(
        max_abs_error(reset1.values, fresh1.values) <= 1e-6F,
        "checkpoint GQA reset output mismatch");

    const auto linear =
        bind_qwen3_next_dense_layer(checkpoint, config, 0);
    const auto rejected = QwenCheckpointGqa::create(
        linear,
        config,
        &diagnostic);
    require(
        !rejected.has_value(),
        "DeltaNet binding must be rejected by checkpoint GQA");

    fs::remove_all(root);
    std::cout
        << "OSM-31B checkpoint-bound gated GQA: PASS\n"
        << "  qkv_o_source=model.safetensors affine Q4\n"
        << "  qk_norm_source=model.safetensors\n"
        << "  rope_theta=" << config.rope_theta << "\n"
        << "  rotary_dims=" << config.rotary_dims() << "\n"
        << "  max_position_embeddings="
        << config.max_position_embeddings << "\n"
        << "  two_token_KV_parity=PASS\n"
        << "  reset_state=PASS\n"
        << "  layer_classification_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-31B checkpoint-bound gated GQA: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
