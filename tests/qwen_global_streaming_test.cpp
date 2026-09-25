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
#include "orbi/streammoe/model/qwen_global_binding.hpp"
#include "orbi/streammoe/model/qwen_global_streaming.hpp"
#include "orbi/streammoe/model/checkpoint_sparse_moe.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"
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

  add_affine(
      tensors,
      "model.embed_tokens",
      16,
      8,
      4,
      4,
      90);
  add_plain(
      tensors,
      "model.norm.weight",
      {8},
      8,
      91);
  add_affine(
      tensors,
      "lm_head",
      16,
      8,
      4,
      4,
      92);

  write_safetensors(root / "model.safetensors", tensors);

  const std::string config =
      "{"
      "\"model_type\":\"qwen3_next\","
      "\"hidden_size\":8,"
      "\"vocab_size\":16,"
      "\"tie_word_embeddings\":false,"
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
      fs::temp_directory_path() / "orbi-streammoe-osm35b-global-streaming";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);
    const auto global =
        bind_qwen3_next_global_checkpoint(checkpoint, config);

    const auto embed_dense =
        dequantize_mlx_affine_module_cpu(
            checkpoint.read_affine_module("model.embed_tokens"));
    const auto lm_dense =
        dequantize_mlx_affine_module_cpu(
            checkpoint.read_affine_module("lm_head"));

    for (const std::size_t token : {0U, 5U, 15U}) {
      const auto row =
          read_qwen_embedding_row(checkpoint, global, token);
      require(row.size() == global.hidden_size,
              "streamed embedding row width mismatch");

      const auto first = token * global.hidden_size;
      std::vector<float> expected(
          embed_dense.begin() + static_cast<std::ptrdiff_t>(first),
          embed_dense.begin() +
              static_cast<std::ptrdiff_t>(first + global.hidden_size));
      require(
          max_abs_error(row, expected) <= 1e-7F,
          "streamed embedding row differs from full dequant reference");
    }

    const std::vector<float> hidden{
        0.5F, -0.25F, 0.75F, -0.5F,
        0.125F, -0.625F, 0.25F, 0.375F,
    };

    const auto full_logits = cpu::matvec_row_major(
        lm_dense,
        global.vocab_size,
        global.hidden_size,
        hidden);

    const auto chunk = project_qwen_lm_head_chunk(
        checkpoint,
        global,
        hidden,
        4U,
        5U);
    require(chunk.first_token == 4U, "LM-head chunk start mismatch");
    require(chunk.logits.size() == 5U, "LM-head chunk length mismatch");

    const std::vector<float> expected_chunk(
        full_logits.begin() + 4,
        full_logits.begin() + 9);
    require(
        max_abs_error(chunk.logits, expected_chunk) <= 1e-6F,
        "streamed LM-head chunk differs from full projection");

    const auto greedy = greedy_qwen_lm_head_streaming(
        checkpoint,
        global,
        hidden,
        3U);
    const auto expected_it =
        std::max_element(full_logits.begin(), full_logits.end());
    const auto expected_token =
        static_cast<std::size_t>(
            std::distance(full_logits.begin(), expected_it));
    require(
        greedy.token_id == expected_token,
        "streamed greedy token mismatch");
    require(
        std::fabs(greedy.logit - *expected_it) <= 1e-6F,
        "streamed greedy logit mismatch");

    auto tied_config = config;
    tied_config.tie_word_embeddings = true;
    const auto tied =
        bind_qwen3_next_global_checkpoint(checkpoint, tied_config);
    const auto tied_chunk = project_qwen_lm_head_chunk(
        checkpoint,
        tied,
        hidden,
        2U,
        4U);
    const auto tied_full = cpu::matvec_row_major(
        embed_dense,
        tied.vocab_size,
        tied.hidden_size,
        hidden);
    const std::vector<float> tied_expected(
        tied_full.begin() + 2,
        tied_full.begin() + 6);
    require(
        max_abs_error(tied_chunk.logits, tied_expected) <= 1e-6F,
        "tied streamed LM head mismatch");

    const auto final_norm_slice =
        checkpoint.dense().read_floats(
            "model.norm.weight",
            2U,
            3U);
    const auto final_norm_full =
        checkpoint.dense().read_floats("model.norm.weight");
    require(final_norm_slice.size() == 3U,
            "bounded float read size mismatch");
    require(
        max_abs_error(
            final_norm_slice,
            std::span<const float>(final_norm_full).subspan(2U, 3U)) <= 1e-7F,
        "bounded float read parity failed");

    const auto packed_slice =
        checkpoint.dense().read_u32(
            "model.embed_tokens.weight",
            3U,
            4U);
    const auto packed_full =
        checkpoint.dense().read_u32("model.embed_tokens.weight");
    require(packed_slice.size() == 4U,
            "bounded U32 read size mismatch");
    require(
        std::equal(
            packed_slice.begin(),
            packed_slice.end(),
            packed_full.begin() + 3),
        "bounded U32 read parity failed");

    bool bad_token_rejected = false;
    try {
      (void)read_qwen_embedding_row(
          checkpoint,
          global,
          global.vocab_size);
    } catch (const std::exception&) {
      bad_token_rejected = true;
    }
    require(bad_token_rejected, "out-of-range token id must fail");

    bool bad_chunk_rejected = false;
    try {
      (void)project_qwen_lm_head_chunk(
          checkpoint,
          global,
          hidden,
          global.vocab_size - 1U,
          2U);
    } catch (const std::exception&) {
      bad_chunk_rejected = true;
    }
    require(bad_chunk_rejected, "out-of-range LM-head chunk must fail");

    bool zero_chunk_rejected = false;
    try {
      (void)greedy_qwen_lm_head_streaming(
          checkpoint,
          global,
          hidden,
          0U);
    } catch (const std::exception&) {
      zero_chunk_rejected = true;
    }
    require(zero_chunk_rejected, "zero greedy chunk must fail");

    fs::remove_all(root);
    std::cout
        << "OSM-35B streamed global matrices: PASS\n"
        << "  embedding_one_row_reads=PASS\n"
        << "  lm_head_bounded_chunk=PASS\n"
        << "  greedy_without_full_vocab_logits=PASS\n"
        << "  tied_lm_head_alias=PASS\n"
        << "  bounded_F32_reads=PASS\n"
        << "  bounded_U32_reads=PASS\n"
        << "  range_guards=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-35B streamed global matrices: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
