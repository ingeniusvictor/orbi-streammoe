#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

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

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm26c-qwen-binding";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);

    require(config.hidden_size == 8U, "hidden size mismatch");
    require(config.num_hidden_layers == 4U, "layer count mismatch");
    require(config.is_linear_layer(0), "layer 0 must be DeltaNet");
    require(!config.is_linear_layer(3), "layer 3 must be full attention");
    require(config.key_dim() == 8U, "key dim mismatch");
    require(config.value_dim() == 8U, "value dim mismatch");
    require(config.conv_dim() == 24U, "conv dim mismatch");
    require(config.norm_topk_prob, "norm_topk_prob mismatch");

    const auto linear =
        bind_qwen3_next_dense_layer(checkpoint, config, 0);
    require(linear.is_linear, "bound layer 0 must be linear");
    require(linear.delta.has_value(), "DeltaNet binding missing");
    require(!linear.attention.has_value(), "unexpected attention binding");
    require(linear.input_norm.size() == 8U, "input norm size mismatch");
    require(linear.moe.router.quant.bits == 8U, "router override missing");
    require(
        linear.moe.router.path == "model.layers.0.mlp.gate",
        "router exact path mismatch");
    require(
        linear.moe.shared_expert_gate.quant.bits == 8U,
        "shared scalar gate override missing");
    require(
        linear.delta->in_proj_qkvz.logical_shape ==
            std::vector<std::size_t>({32, 8}),
        "fused qkvz shape mismatch");
    require(
        linear.delta->in_proj_ba.logical_shape ==
            std::vector<std::size_t>({2, 8}),
        "fused ba shape mismatch");
    require(linear.delta->conv.size() == 48U, "conv tensor size mismatch");

    const auto attention =
        bind_qwen3_next_dense_layer(checkpoint, config, 3);
    require(!attention.is_linear, "bound layer 3 must be attention");
    require(!attention.delta.has_value(), "unexpected DeltaNet binding");
    require(attention.attention.has_value(), "attention binding missing");
    require(
        attention.attention->q_proj.logical_shape ==
            std::vector<std::size_t>({16, 8}),
        "q_proj shape mismatch");
    require(
        attention.attention->k_proj.logical_shape ==
            std::vector<std::size_t>({8, 8}),
        "k_proj shape mismatch");
    require(
        attention.attention->o_proj.logical_shape ==
            std::vector<std::size_t>({8, 8}),
        "o_proj shape mismatch");
    require(
        attention.attention->q_norm.size() == 8U &&
        attention.attention->k_norm.size() == 8U,
        "attention norm size mismatch");

    bool out_of_range_rejected = false;
    try {
      (void)bind_qwen3_next_dense_layer(
          checkpoint, config, config.num_hidden_layers);
    } catch (const std::exception&) {
      out_of_range_rejected = true;
    }
    require(out_of_range_rejected, "out-of-range layer must be rejected");

    bool missing_layer_rejected = false;
    try {
      (void)bind_qwen3_next_dense_layer(checkpoint, config, 1);
    } catch (const std::exception&) {
      missing_layer_rejected = true;
    }
    require(
        missing_layer_rejected,
        "incomplete layer tensor set must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-26C Qwen3-Next dense layer binding: PASS\n"
        << "  layer0=DeltaNet fused layout PASS\n"
        << "  layer3=GQA full attention PASS\n"
        << "  exact_MoE_names=PASS\n"
        << "  router/shared_gate_8bit_overrides=PASS\n"
        << "  config_shape_validation=PASS\n"
        << "  incomplete_layer_rejected=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-26C Qwen3-Next dense layer binding: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
