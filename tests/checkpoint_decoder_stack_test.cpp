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

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm34-decoder-stack";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);

    require(config.num_hidden_layers == 4U, "fixture layer count mismatch");
    require(config.full_attention_interval == 4U, "fixture interval mismatch");

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
          << "OSM-34 multi-layer decoder stack: PASS "
          << "(no compute device on host)\n";
      return 0;
    }

    auto stack = QwenCheckpointDecoderStack::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    require(stack.has_value(), diagnostic);
    require(stack->valid(), "decoder stack must be valid");
    require(stack->layer_count() == 4U, "stack layer count mismatch");
    require(stack->hidden_size() == 8U, "stack hidden size mismatch");
    require(stack->delta_layer_count() == 3U, "DeltaNet layer count mismatch");
    require(stack->gqa_layer_count() == 1U, "GQA layer count mismatch");

    for (std::size_t layer = 0U; layer < 4U; ++layer) {
      const auto* item = stack->layer(layer);
      require(item != nullptr, "missing stack layer");
      const auto expected =
          layer == 3U
              ? QwenCheckpointDecoderLayerKind::gated_gqa
              : QwenCheckpointDecoderLayerKind::gated_deltanet;
      require(item->kind() == expected, "D-D-D-G layer order mismatch");
    }
    require(stack->layer(4U) == nullptr, "out-of-range stack layer must be null");

    std::vector<QwenCheckpointDecoderLayer> reference;
    reference.reserve(4U);
    for (std::size_t layer = 0U; layer < 4U; ++layer) {
      const auto binding =
          bind_qwen3_next_dense_layer(checkpoint, config, layer);
      auto item = QwenCheckpointDecoderLayer::create(
          *context,
          binding,
          config,
          &diagnostic);
      require(item.has_value(), diagnostic);
      reference.push_back(std::move(*item));
    }

    QpackExpertStorage storage(root);

    ExpertCache stack_host_cache(
        storage,
        storage.expert_stride_bytes() * 12U,
        1U);
    VulkanResidentExpertCache stack_gpu_cache(
        *context,
        storage.reader(),
        stack_host_cache,
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
        [&](std::span<const float> input) {
          std::vector<float> values(input.begin(), input.end());
          for (auto& layer : reference) {
            const auto step =
                layer.run(*context, ref_gpu_cache, values);
            require(step.executed, step.diagnostic);
            values = step.values;
          }
          return values;
        };

    const std::vector<float> x1{
        0.5F, -0.25F, 0.75F, -0.5F,
        0.125F, -0.625F, 0.25F, 0.375F,
    };
    const std::vector<float> x2{
        -0.125F, 0.625F, -0.375F, 0.25F,
        0.5F, 0.125F, -0.75F, 0.875F,
    };

    const auto expected1 = run_reference(x1);
    const auto actual1 =
        stack->run(*context, stack_gpu_cache, x1);
    require(actual1.executed, actual1.diagnostic);
    require(actual1.layers_executed == 4U, "token1 did not execute four layers");
    const auto error1 =
        max_abs_error(actual1.values, expected1);
    require(error1 <= 0.03F, "token1 stack parity failed");

    const auto stats_after_first = stack_gpu_cache.stats();
    require(
        stats_after_first.loads >= 4U * kTopK,
        "first stack token must load Top-K experts for every layer");

    const auto expected2 = run_reference(x2);
    const auto actual2 =
        stack->run(*context, stack_gpu_cache, x2);
    require(actual2.executed, actual2.diagnostic);
    require(actual2.layers_executed == 4U, "token2 did not execute four layers");
    const auto error2 =
        max_abs_error(actual2.values, expected2);
    require(error2 <= 0.03F, "token2 stack parity failed");

    for (std::size_t layer = 0U; layer < 3U; ++layer) {
      const auto* state = stack->layer(layer)->delta_state();
      require(state != nullptr, "DeltaNet state pointer missing");
      require(!state->conv_tail.empty(), "DeltaNet conv state did not persist");
      require(!state->recurrent.empty(), "DeltaNet recurrent state did not persist");
    }

    const auto* gqa_state = stack->layer(3U)->gqa_state();
    require(gqa_state != nullptr, "GQA state pointer missing");
    require(gqa_state->position == 2U, "GQA position must be two after two tokens");
    require(!gqa_state->k_cache.empty(), "GQA K cache did not persist");
    require(!gqa_state->v_cache.empty(), "GQA V cache did not persist");

    const auto bad = stack->run(
        *context,
        stack_gpu_cache,
        std::span<const float>(x1).first(7U));
    require(!bad.executed, "invalid stack hidden shape must be rejected");

    stack->reset_state();
    for (std::size_t layer = 0U; layer < 3U; ++layer) {
      const auto* state = stack->layer(layer)->delta_state();
      require(state != nullptr, "DeltaNet reset state pointer missing");
      require(state->conv_tail.empty(), "DeltaNet conv reset failed");
      require(state->recurrent.empty(), "DeltaNet recurrent reset failed");
    }
    gqa_state = stack->layer(3U)->gqa_state();
    require(gqa_state != nullptr, "GQA reset state pointer missing");
    require(gqa_state->position == 0U, "GQA reset position failed");
    require(gqa_state->k_cache.empty(), "GQA reset K cache failed");
    require(gqa_state->v_cache.empty(), "GQA reset V cache failed");

    fs::remove_all(root);
    std::cout
        << "OSM-34 multi-layer alternating decoder stack: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  topology=D-D-D-G\n"
        << "  token1_layers=4\n"
        << "  token2_layers=4\n"
        << "  token1_max_abs_error=" << error1 << "\n"
        << "  token2_max_abs_error=" << error2 << "\n"
        << "  first_token_expert_loads=" << stats_after_first.loads << "\n"
        << "  shared_expert_cache=PASS\n"
        << "  persistent_layer_state=PASS\n"
        << "  reset_state=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-34 multi-layer alternating decoder stack: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
