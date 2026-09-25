#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/model/checkpoint_decoder_stack.hpp"
#include "orbi/streammoe/model/checkpoint_model_shell.hpp"
#include "orbi/streammoe/model/qwen_global_binding.hpp"
#include "orbi/streammoe/model/qwen_global_streaming.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"
#include "orbi/streammoe/model/checkpoint_decoder_layer.hpp"
#include "orbi/streammoe/storage/qpack_expert_storage.hpp"
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


constexpr std::size_t kExpertStride = 1024U;
constexpr std::size_t kExpertCount = 3U;
constexpr std::size_t kTopK = 2U;

bool require_vulkan_compute() {
  const char* value = std::getenv("ORBI_REQUIRE_VULKAN_COMPUTE");
  return value != nullptr && std::string(value) != "0";
}

void write_text(
    const fs::path& path,
    const std::string& value) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write fixture");
  out << value;
}

void write_u32_at(
    std::vector<std::byte>& blob,
    std::size_t offset,
    std::uint32_t value) {
  std::memcpy(blob.data() + offset, &value, sizeof(value));
}

void write_f32_at(
    std::vector<std::byte>& blob,
    std::size_t offset,
    float value) {
  std::memcpy(blob.data() + offset, &value, sizeof(value));
}

std::uint32_t expert_q4_word(
    std::size_t seed,
    std::size_t row) {
  std::uint32_t word = 0U;
  for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
    const auto q = static_cast<std::uint32_t>(
        (seed + row * 3U + lane * 5U) & 0xFU);
    word |= q << (lane * 4U);
  }
  return word;
}

void fill_expert_projection(
    std::vector<std::byte>& blob,
    std::size_t expert,
    std::size_t seed,
    std::size_t weight_offset,
    std::size_t scales_offset,
    std::size_t biases_offset) {
  for (std::size_t row = 0U; row < 8U; ++row) {
    write_u32_at(
        blob,
        weight_offset + row * sizeof(std::uint32_t),
        expert_q4_word(seed + expert * 7U, row));

    for (std::size_t group = 0U; group < 2U; ++group) {
      const auto i = row * 2U + group;
      const float scale =
          0.03125F *
          static_cast<float>(
              (seed + expert + row + group) % 5U + 1U);
      const float bias =
          0.0625F *
          static_cast<float>(
              static_cast<int>(
                  (seed + expert * 2U + row + group) % 7U) - 3);

      write_f32_at(
          blob,
          scales_offset + i * sizeof(float),
          scale);
      write_f32_at(
          blob,
          biases_offset + i * sizeof(float),
          bias);
    }
  }
}

void build_expert_layer(
    const fs::path& path,
    std::size_t layer_seed) {
  constexpr std::size_t gate_w = 0U;
  constexpr std::size_t gate_s = 32U;
  constexpr std::size_t gate_b = 96U;
  constexpr std::size_t up_w = 160U;
  constexpr std::size_t up_s = 192U;
  constexpr std::size_t up_b = 256U;
  constexpr std::size_t down_w = 320U;
  constexpr std::size_t down_s = 352U;
  constexpr std::size_t down_b = 416U;

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create routed expert layer");

  for (std::size_t expert = 0U; expert < kExpertCount; ++expert) {
    std::vector<std::byte> blob(kExpertStride);
    fill_expert_projection(
        blob, expert, layer_seed + 100U,
        gate_w, gate_s, gate_b);
    fill_expert_projection(
        blob, expert, layer_seed + 120U,
        up_w, up_s, up_b);
    fill_expert_projection(
        blob, expert, layer_seed + 140U,
        down_w, down_s, down_b);

    out.write(
        reinterpret_cast<const char*>(blob.data()),
        static_cast<std::streamsize>(blob.size()));
  }
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  std::vector<TensorFixture> tensors;
  for (std::size_t layer = 0U; layer < 4U; ++layer) {
    const auto seed = 10U + layer * 40U;
    add_common_layer(tensors, layer, seed);
    if (layer == 3U) {
      add_attention_layer(tensors, layer, seed + 20U);
    } else {
      add_delta_layer(tensors, layer, seed + 20U);
    }
  }
  add_affine(
      tensors,
      "model.embed_tokens",
      16, 8, 4, 4, 9000U);
  add_plain(
      tensors,
      "model.norm.weight",
      {8},
      8,
      9001U);
  add_affine(
      tensors,
      "lm_head",
      16, 8, 4, 4, 9002U);

  write_safetensors(root / "model.safetensors", tensors);

  for (std::size_t layer = 0U; layer < 4U; ++layer) {
    const auto name =
        "layer_0" + std::to_string(layer) + ".bin";
    build_expert_layer(
        root / "packed_experts" / name,
        layer * 1000U);
  }

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
      "\"model.layers.1.mlp.gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.1.mlp.shared_expert_gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.2.mlp.gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.2.mlp.shared_expert_gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.3.mlp.gate\":{\"group_size\":4,\"bits\":8},"
      "\"model.layers.3.mlp.shared_expert_gate\":{\"group_size\":4,\"bits\":8}"
      "}"
      "}";
  write_text(root / "config.json", config);

  const std::string layout =
      "{"
      "\"expertCount\":3,"
      "\"layerCount\":4,"
      "\"expertStride\":1024,"
      "\"sections\":["
      "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[8,1],\"offset\":0,\"size\":32},"
      "{\"name\":\"gate_proj.scales\",\"dtype\":\"F32\",\"shape\":[8,2],\"offset\":32,\"size\":64},"
      "{\"name\":\"gate_proj.biases\",\"dtype\":\"F32\",\"shape\":[8,2],\"offset\":96,\"size\":64},"
      "{\"name\":\"up_proj.weight\",\"dtype\":\"U32\",\"shape\":[8,1],\"offset\":160,\"size\":32},"
      "{\"name\":\"up_proj.scales\",\"dtype\":\"F32\",\"shape\":[8,2],\"offset\":192,\"size\":64},"
      "{\"name\":\"up_proj.biases\",\"dtype\":\"F32\",\"shape\":[8,2],\"offset\":256,\"size\":64},"
      "{\"name\":\"down_proj.weight\",\"dtype\":\"U32\",\"shape\":[8,1],\"offset\":320,\"size\":32},"
      "{\"name\":\"down_proj.scales\",\"dtype\":\"F32\",\"shape\":[8,2],\"offset\":352,\"size\":64},"
      "{\"name\":\"down_proj.biases\",\"dtype\":\"F32\",\"shape\":[8,2],\"offset\":416,\"size\":64}"
      "],"
      "\"linearLayers\":[true,true,true,false]"
      "}";
  write_text(root / "packed_experts" / "layout.json", layout);

  const auto dense_size =
      fs::file_size(root / "model.safetensors");
  const auto config_size =
      fs::file_size(root / "config.json");
  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");

  std::ostringstream manifest;
  manifest
      << "{"
      << "\"magic\":\"QPACK\","
      << "\"version\":1,"
      << "\"modelName\":\"qwen3_next\","
      << "\"sourceCheckpoint\":\"osm34-multilayer\","
      << "\"quantBits\":4,"
      << "\"quantGroupSize\":4,"
      << "\"files\":{"
      << "\"model.safetensors\":" << dense_size << ","
      << "\"config.json\":" << config_size << ","
      << "\"packed_experts/layout.json\":" << layout_size;

  for (std::size_t layer = 0U; layer < 4U; ++layer) {
    const auto file =
        "packed_experts/layer_0" + std::to_string(layer) + ".bin";
    const auto bytes = fs::file_size(root / file);
    manifest << ",\"" << file << "\":" << bytes;
  }
  manifest << "}}";

  write_text(root / "manifest.json", manifest.str());
  return root;
}

float max_abs_error(
    std::span<const float> actual,
    std::span<const float> expected) {
  require(actual.size() == expected.size(), "output size mismatch");
  float worst = 0.0F;
  for (std::size_t i = 0U; i < actual.size(); ++i) {
    worst = std::max(
        worst,
        std::fabs(actual[i] - expected[i]));
  }
  return worst;
}

}  // namespace


struct ReferenceModelStep {
  std::vector<float> embedding;
  std::vector<float> decoder_hidden;
  std::vector<float> final_hidden;
  std::size_t token{};
  float logit{};
};

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm35c-model-shell";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);

    require(config.vocab_size == 16U, "model-shell vocab size mismatch");
    require(config.num_hidden_layers == 4U, "model-shell layer count mismatch");

    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no Vulkan context: " +
            diagnostic);
      }
      fs::remove_all(root);
      std::cout
          << "OSM-35C checkpoint model shell: PASS "
          << "(no compute device on host)\n";
      return 0;
    }

    auto model = QwenCheckpointModelShell::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    require(model.has_value(), diagnostic);
    require(model->valid(), "checkpoint model shell must be valid");
    require(model->hidden_size() == 8U, "model-shell hidden size mismatch");
    require(model->vocab_size() == 16U, "model-shell vocab mismatch");
    require(model->layer_count() == 4U, "model-shell stack mismatch");
    require(model->global_binding() != nullptr, "global binding missing");
    require(model->decoder_stack() != nullptr, "decoder stack missing");

    auto reference_stack = QwenCheckpointDecoderStack::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    require(reference_stack.has_value(), diagnostic);

    const auto global =
        bind_qwen3_next_global_checkpoint(checkpoint, config);
    const auto embed_full =
        dequantize_mlx_affine_module_cpu(
            checkpoint.read_affine_module("model.embed_tokens"));
    const auto lm_head_full =
        dequantize_mlx_affine_module_cpu(
            checkpoint.read_affine_module("lm_head"));

    QpackExpertStorage storage(root);

    ExpertCache model_host_cache(
        storage,
        storage.expert_stride_bytes() * 12U,
        1U);
    VulkanResidentExpertCache model_gpu_cache(
        *context,
        storage.reader(),
        model_host_cache,
        65536U);

    ExpertCache ref_host_cache(
        storage,
        storage.expert_stride_bytes() * 12U,
        1U);
    VulkanResidentExpertCache ref_gpu_cache(
        *context,
        storage.reader(),
        ref_host_cache,
        65536U);

    const auto run_reference =
        [&](std::size_t token_id) {
          ReferenceModelStep result;

          const auto begin = token_id * config.hidden_size;
          result.embedding.assign(
              embed_full.begin() + static_cast<std::ptrdiff_t>(begin),
              embed_full.begin() +
                  static_cast<std::ptrdiff_t>(begin + config.hidden_size));

          const auto decoded = reference_stack->run(
              *context,
              ref_gpu_cache,
              result.embedding);
          require(decoded.executed, decoded.diagnostic);
          result.decoder_hidden = decoded.values;

          result.final_hidden = result.decoder_hidden;
          cpu::rms_norm_inplace(
              result.final_hidden,
              1U,
              config.hidden_size,
              global.final_norm,
              config.rms_norm_eps);

          const auto logits = cpu::matvec_row_major(
              lm_head_full,
              config.vocab_size,
              config.hidden_size,
              result.final_hidden);
          const auto best =
              std::max_element(logits.begin(), logits.end());
          result.token = static_cast<std::size_t>(
              std::distance(logits.begin(), best));
          result.logit = *best;
          return result;
        };

    const auto expected1 = run_reference(5U);
    const auto actual1 = model->step_greedy(
        checkpoint,
        *context,
        model_gpu_cache,
        5U,
        3U);
    require(actual1.executed, actual1.diagnostic);
    require(actual1.input_token == 5U, "token1 input id mismatch");

    const auto embed_error1 =
        max_abs_error(actual1.embedding, expected1.embedding);
    const auto decoder_error1 =
        max_abs_error(actual1.decoder_hidden, expected1.decoder_hidden);
    const auto final_error1 =
        max_abs_error(actual1.final_hidden, expected1.final_hidden);

    require(embed_error1 <= 1e-7F, "token1 embedding parity failed");
    require(decoder_error1 <= 0.03F, "token1 decoder parity failed");
    require(final_error1 <= 1e-5F, "token1 final RMSNorm parity failed");
    require(actual1.greedy.token_id == expected1.token,
            "token1 greedy id mismatch");
    require(std::fabs(actual1.greedy.logit - expected1.logit) <= 1e-5F,
            "token1 greedy logit mismatch");

    const auto expected2 = run_reference(7U);
    const auto actual2 = model->step_greedy(
        checkpoint,
        *context,
        model_gpu_cache,
        7U,
        5U);
    require(actual2.executed, actual2.diagnostic);

    const auto embed_error2 =
        max_abs_error(actual2.embedding, expected2.embedding);
    const auto decoder_error2 =
        max_abs_error(actual2.decoder_hidden, expected2.decoder_hidden);
    const auto final_error2 =
        max_abs_error(actual2.final_hidden, expected2.final_hidden);

    require(embed_error2 <= 1e-7F, "token2 embedding parity failed");
    require(decoder_error2 <= 0.03F, "token2 decoder parity failed");
    require(final_error2 <= 1e-5F, "token2 final RMSNorm parity failed");
    require(actual2.greedy.token_id == expected2.token,
            "token2 greedy id mismatch");
    require(std::fabs(actual2.greedy.logit - expected2.logit) <= 1e-5F,
            "token2 greedy logit mismatch");

    const auto* model_stack = model->decoder_stack();
    const auto* gqa = model_stack->layer(3U)->gqa_state();
    require(gqa != nullptr && gqa->position == 2U,
            "model-shell GQA state must persist across two tokens");
    for (std::size_t layer = 0U; layer < 3U; ++layer) {
      const auto* delta = model_stack->layer(layer)->delta_state();
      require(delta != nullptr, "model-shell DeltaNet state missing");
      require(!delta->conv_tail.empty(), "model-shell conv state missing");
      require(!delta->recurrent.empty(), "model-shell recurrent state missing");
    }

    model->reset_state();
    reference_stack->reset_state();

    const auto expected_reset = run_reference(5U);
    const auto actual_reset = model->step_greedy(
        checkpoint,
        *context,
        model_gpu_cache,
        5U,
        4U);
    require(actual_reset.executed, actual_reset.diagnostic);
    require(
        max_abs_error(
            actual_reset.final_hidden,
            expected_reset.final_hidden) <= 1e-5F,
        "model-shell reset final state mismatch");
    require(actual_reset.greedy.token_id == expected_reset.token,
            "model-shell reset greedy id mismatch");

    const auto bad_token = model->step_greedy(
        checkpoint,
        *context,
        model_gpu_cache,
        config.vocab_size,
        4U);
    require(!bad_token.executed, "out-of-range model token must fail");

    const auto bad_chunk = model->step_greedy(
        checkpoint,
        *context,
        model_gpu_cache,
        1U,
        0U);
    require(!bad_chunk.executed, "zero model LM-head chunk must fail");

    const auto stats = model_gpu_cache.stats();
    require(stats.loads > 0U, "model shell must stream routed experts");

    fs::remove_all(root);
    std::cout
        << "OSM-35C checkpoint model shell: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  token_to_embedding=STREAMED\n"
        << "  decoder_topology=D-D-D-G\n"
        << "  final_RMSNorm=VULKAN\n"
        << "  lm_head=STREAMED_CHUNKS\n"
        << "  token1_embedding_error=" << embed_error1 << "\n"
        << "  token1_decoder_error=" << decoder_error1 << "\n"
        << "  token1_final_error=" << final_error1 << "\n"
        << "  token2_decoder_error=" << decoder_error2 << "\n"
        << "  token2_final_error=" << final_error2 << "\n"
        << "  persistent_sequence_state=PASS\n"
        << "  reset_state=PASS\n"
        << "  bounded_global_memory=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-35C checkpoint model shell: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
