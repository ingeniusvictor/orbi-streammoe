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

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"
#include "orbi/streammoe/model/checkpoint_sparse_moe.hpp"
#include "orbi/streammoe/model/checkpoint_full_attention_decoder_layer.hpp"
#include "orbi/streammoe/model/checkpoint_gqa_sublayer.hpp"
#include "orbi/streammoe/model/checkpoint_moe_sublayer.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/model/qwen_router.hpp"
#include "orbi/streammoe/model/shared_expert.hpp"
#include "orbi/streammoe/storage/qpack_expert_storage.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

constexpr std::size_t kHidden = 8U;
constexpr std::size_t kIntermediate = 8U;
constexpr std::size_t kExperts = 3U;
constexpr std::size_t kTopK = 2U;
constexpr std::size_t kGroupSize = 4U;
constexpr std::size_t kExpertStride = 1024U;

struct TensorFixture {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::vector<std::byte> bytes;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool require_vulkan_compute() {
  const char* value = std::getenv("ORBI_REQUIRE_VULKAN_COMPUTE");
  return value != nullptr && std::string(value) != "0";
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

  const auto text = header.str();
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create dense safetensors");
  append_u64_le(out, text.size());
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
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
    append_f32_le(
        bytes,
        static_cast<float>(
            static_cast<int>((i + seed) % 11U) - 5) *
            0.0625F);
  }
  tensors.push_back({
      .name = name,
      .dtype = "F32",
      .shape = std::move(shape),
      .bytes = std::move(bytes),
  });
}

std::uint32_t packed_q4_word(
    std::size_t seed,
    std::size_t row) {
  std::uint32_t word = 0U;
  for (std::uint32_t lane = 0; lane < 8U; ++lane) {
    const auto q = static_cast<std::uint32_t>(
        (seed + row * 3U + lane * 2U) & 0xFU);
    word |= q << (lane * 4U);
  }
  return word;
}

std::pair<float, float> affine_metadata(
    std::size_t seed,
    std::size_t row,
    std::size_t group) {
  const float scale =
      0.03125F *
      static_cast<float>((seed + row + group) % 5U + 1U);
  const float bias =
      0.0625F *
      static_cast<float>(
          static_cast<int>((seed + row * 2U + group) % 7U) - 3);
  return {scale, bias};
}

void add_q4_affine(
    std::vector<TensorFixture>& tensors,
    const std::string& path,
    std::size_t rows,
    std::size_t logical_cols,
    std::size_t seed) {
  require(logical_cols == 8U, "OSM-27 fixture expects Q4 logical cols=8");

  std::vector<std::byte> packed;
  std::vector<std::byte> scales;
  std::vector<std::byte> biases;

  for (std::size_t row = 0; row < rows; ++row) {
    append_u32_le(packed, packed_q4_word(seed, row));
    for (std::size_t group = 0; group < 2U; ++group) {
      const auto [scale, bias] =
          affine_metadata(seed, row, group);
      append_f32_le(scales, scale);
      append_f32_le(biases, bias);
    }
  }

  tensors.push_back({
      .name = path + ".weight",
      .dtype = "U32",
      .shape = {rows, 1},
      .bytes = std::move(packed),
  });
  tensors.push_back({
      .name = path + ".scales",
      .dtype = "F32",
      .shape = {rows, 2},
      .bytes = std::move(scales),
  });
  tensors.push_back({
      .name = path + ".biases",
      .dtype = "F32",
      .shape = {rows, 2},
      .bytes = std::move(biases),
  });
}

void add_q8_affine_explicit(
    std::vector<TensorFixture>& tensors,
    const std::string& path,
    const std::vector<std::array<std::uint8_t, 8>>& rows) {
  std::vector<std::byte> packed;
  std::vector<std::byte> scales;
  std::vector<std::byte> biases;

  for (const auto& row : rows) {
    const std::uint32_t w0 =
        static_cast<std::uint32_t>(row[0]) |
        (static_cast<std::uint32_t>(row[1]) << 8U) |
        (static_cast<std::uint32_t>(row[2]) << 16U) |
        (static_cast<std::uint32_t>(row[3]) << 24U);
    const std::uint32_t w1 =
        static_cast<std::uint32_t>(row[4]) |
        (static_cast<std::uint32_t>(row[5]) << 8U) |
        (static_cast<std::uint32_t>(row[6]) << 16U) |
        (static_cast<std::uint32_t>(row[7]) << 24U);
    append_u32_le(packed, w0);
    append_u32_le(packed, w1);

    append_f32_le(scales, 1.0F);
    append_f32_le(scales, 1.0F);
    append_f32_le(biases, 0.0F);
    append_f32_le(biases, 0.0F);
  }

  tensors.push_back({
      .name = path + ".weight",
      .dtype = "U32",
      .shape = {rows.size(), 2},
      .bytes = std::move(packed),
  });
  tensors.push_back({
      .name = path + ".scales",
      .dtype = "F32",
      .shape = {rows.size(), 2},
      .bytes = std::move(scales),
  });
  tensors.push_back({
      .name = path + ".biases",
      .dtype = "F32",
      .shape = {rows.size(), 2},
      .bytes = std::move(biases),
  });
}

void build_dense_checkpoint(
    const fs::path& path) {
  const std::string p = "model.layers.0.";
  std::vector<TensorFixture> tensors;

  add_plain(tensors, p + "input_layernorm.weight", {8}, 8, 1);
  add_plain(tensors, p + "post_attention_layernorm.weight", {8}, 8, 2);

  add_q8_affine_explicit(
      tensors,
      p + "mlp.gate",
      {
          {1,0,0,0,0,0,0,0},
          {3,0,0,0,0,0,0,0},
          {2,0,0,0,0,0,0,0},
      });

  add_q4_affine(
      tensors,
      p + "mlp.shared_expert.gate_proj",
      8, 8, 10);
  add_q4_affine(
      tensors,
      p + "mlp.shared_expert.up_proj",
      8, 8, 20);
  add_q4_affine(
      tensors,
      p + "mlp.shared_expert.down_proj",
      8, 8, 30);

  add_q8_affine_explicit(
      tensors,
      p + "mlp.shared_expert_gate",
      {
          {1,2,0,0,0,0,0,0},
      });

  add_plain(
      tensors,
      p + "linear_attn.conv1d.weight",
      {24,2},
      48,
      40);
  add_plain(
      tensors,
      p + "linear_attn.dt_bias",
      {1},
      1,
      41);
  add_plain(
      tensors,
      p + "linear_attn.A_log",
      {1},
      1,
      42);
  add_plain(
      tensors,
      p + "linear_attn.norm.weight",
      {8},
      8,
      43);

  add_q4_affine(
      tensors,
      p + "linear_attn.out_proj",
      8, 8, 50);
  add_q4_affine(
      tensors,
      p + "linear_attn.in_proj_qkvz",
      32, 8, 60);
  add_q4_affine(
      tensors,
      p + "linear_attn.in_proj_ba",
      2, 8, 70);

  add_q4_affine(
      tensors,
      p + "self_attn.q_proj",
      16, 8, 80);
  add_q4_affine(
      tensors,
      p + "self_attn.k_proj",
      8, 8, 81);
  add_q4_affine(
      tensors,
      p + "self_attn.v_proj",
      8, 8, 82);
  add_q4_affine(
      tensors,
      p + "self_attn.o_proj",
      8, 8, 83);
  add_plain(
      tensors,
      p + "self_attn.q_norm.weight",
      {8},
      8,
      84);
  add_plain(
      tensors,
      p + "self_attn.k_norm.weight",
      {8},
      8,
      85);

  write_safetensors(path, tensors);
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

void fill_expert_projection(
    std::vector<std::byte>& blob,
    std::size_t expert,
    std::size_t seed,
    std::size_t weight_offset,
    std::size_t scales_offset,
    std::size_t biases_offset) {
  for (std::size_t row = 0; row < 8U; ++row) {
    write_u32_at(
        blob,
        weight_offset + row * sizeof(std::uint32_t),
        packed_q4_word(seed + expert * 5U, row));

    for (std::size_t group = 0; group < 2U; ++group) {
      const auto [scale, bias] =
          affine_metadata(seed + expert * 7U, row, group);
      const auto i = row * 2U + group;
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
    const fs::path& path) {
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

  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    std::vector<std::byte> blob(kExpertStride);
    fill_expert_projection(
        blob, expert, 100, gate_w, gate_s, gate_b);
    fill_expert_projection(
        blob, expert, 120, up_w, up_s, up_b);
    fill_expert_projection(
        blob, expert, 140, down_w, down_s, down_b);

    out.write(
        reinterpret_cast<const char*>(blob.data()),
        static_cast<std::streamsize>(blob.size()));
  }
}

void write_text(
    const fs::path& path,
    const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write qpack fixture");
  out << text;
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  build_dense_checkpoint(root / "model.safetensors");
  build_expert_layer(root / "packed_experts" / "layer_00.bin");

  const std::string config =
      "{"
      "\"model_type\":\"qwen3_next\","
      "\"hidden_size\":8,"
      "\"num_hidden_layers\":1,"
      "\"full_attention_interval\":1,"
      "\"num_attention_heads\":1,"
      "\"num_key_value_heads\":1,"
      "\"head_dim\":8,"
      "\"partial_rotary_factor\":0.5,"
      "\"rope_theta\":10000,"
      "\"rms_norm_eps\":0.000001,"
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
      "\"model.layers.0.mlp.shared_expert_gate\":{\"group_size\":4,\"bits\":8}"
      "}"
      "}";
  write_text(root / "config.json", config);

  const std::string layout =
      "{"
      "\"expertCount\":3,"
      "\"layerCount\":1,"
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
      "\"linearLayers\":[true]"
      "}";
  write_text(root / "packed_experts" / "layout.json", layout);

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
      "\"sourceCheckpoint\":\"osm27-checkpoint-bound\","
      "\"quantBits\":4,"
      "\"quantGroupSize\":4,"
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

std::vector<float> cpu_expert(
    QpackExpertStorage& storage,
    std::uint32_t expert,
    std::span<const float> x) {
  ExpertCache cache(storage, storage.expert_stride_bytes(), 1U);
  const std::vector<std::uint32_t> pick{expert};
  const auto entries = cache.fetch(0, pick);
  require(entries.size() == 1U, "CPU expert fetch failed");

  const auto projection = [&](std::string_view name,
                              std::span<const float> input) {
    const auto view =
        bind_qpack_q4_projection(
            storage.reader(), entries.front(), name);
    const auto dense = cpu::dequantize_affine_rows(
        view.packed,
        view.out_dim,
        view.packed_cols,
        view.scales,
        view.biases,
        cpu::AffineQuantSpec{
            .bits = 4,
            .group_size =
                static_cast<std::uint32_t>(view.group_size),
        });
    return cpu::matvec_row_major(
        dense,
        view.out_dim,
        view.packed_cols * 8U,
        input);
  };

  auto gate = projection("gate_proj", x);
  const auto up = projection("up_proj", x);
  cpu::swiglu_inplace(gate, up);
  return projection("down_proj", gate);
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


struct ReferenceLayerResult {
  QwenCheckpointGqaSublayerResult gqa;
  QwenCheckpointMoeSublayerResult moe;
};

int main() {
  const auto root =
      fs::temp_directory_path() /
      "orbi-streammoe-osm32-full-attention-decoder";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);
    const auto binding =
        bind_qwen3_next_dense_layer(checkpoint, config, 0);

    require(!binding.is_linear, "fixture layer must be full attention");
    require(binding.attention.has_value(), "full-attention binding missing");

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
          << "OSM-32 full-attention decoder layer: PASS "
          << "(no compute device on host)\n";
      return 0;
    }

    auto layer = QwenCheckpointFullAttentionDecoderLayer::create(
        *context,
        binding,
        config,
        &diagnostic);
    require(layer.has_value(), diagnostic);
    require(layer->valid(), "full-attention decoder layer must be valid");
    require(layer->layer_index() == 0U, "decoder layer index mismatch");
    require(layer->hidden_size() == kHidden, "decoder hidden size mismatch");

    auto ref_gqa = QwenCheckpointGqaSublayer::create(
        *context,
        binding,
        config,
        &diagnostic);
    require(ref_gqa.has_value(), diagnostic);

    auto ref_moe = QwenCheckpointMoeSublayer::create(
        *context,
        binding,
        config,
        config.rms_norm_eps,
        &diagnostic);
    require(ref_moe.has_value(), diagnostic);

    QpackExpertStorage storage(root);

    ExpertCache full_host_cache(
        storage,
        storage.expert_stride_bytes(),
        1U);
    VulkanResidentExpertCache full_gpu_cache(
        *context,
        storage.reader(),
        full_host_cache,
        4096U);

    ExpertCache ref_host_cache(
        storage,
        storage.expert_stride_bytes(),
        1U);
    VulkanResidentExpertCache ref_gpu_cache(
        *context,
        storage.reader(),
        ref_host_cache,
        4096U);

    const auto run_reference =
        [&](std::span<const float> x) {
          ReferenceLayerResult result;
          result.gqa = ref_gqa->run(*context, x);
          require(result.gqa.executed, result.gqa.diagnostic);

          result.moe = ref_moe->run(
              *context,
              ref_gpu_cache,
              result.gqa.values);
          require(result.moe.executed, result.moe.diagnostic);
          return result;
        };

    const std::vector<float> x1{
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F,
    };
    const std::vector<float> x2{
        0.5F, -0.25F, 0.75F, -0.5F,
        0.125F, -0.625F, 0.25F, 0.375F,
    };

    const auto expected1 = run_reference(x1);
    const auto actual1 = layer->run(*context, full_gpu_cache, x1);
    require(actual1.executed, actual1.diagnostic);

    const auto first_gqa_error =
        max_abs_error(
            actual1.attention_branch.values,
            expected1.gqa.values);
    const auto first_moe_error =
        max_abs_error(
            actual1.moe_branch.values,
            expected1.moe.values);
    const auto first_final_error =
        max_abs_error(actual1.values, expected1.moe.values);

    require(first_gqa_error <= 1e-6F, "token1 GQA sublayer mismatch");
    require(first_moe_error <= 0.03F, "token1 MoE sublayer mismatch");
    require(first_final_error <= 0.03F, "token1 full layer mismatch");

    require(
        layer->gqa_state().position == ref_gqa->state().position,
        "token1 GQA position mismatch");
    require(
        max_abs_error(
            layer->gqa_state().k_cache,
            ref_gqa->state().k_cache) <= 1e-7F,
        "token1 K cache mismatch");
    require(
        max_abs_error(
            layer->gqa_state().v_cache,
            ref_gqa->state().v_cache) <= 1e-7F,
        "token1 V cache mismatch");

    const auto expected2 = run_reference(x2);
    const auto actual2 = layer->run(*context, full_gpu_cache, x2);
    require(actual2.executed, actual2.diagnostic);

    const auto second_gqa_error =
        max_abs_error(
            actual2.attention_branch.values,
            expected2.gqa.values);
    const auto second_moe_error =
        max_abs_error(
            actual2.moe_branch.values,
            expected2.moe.values);
    const auto second_final_error =
        max_abs_error(actual2.values, expected2.moe.values);

    require(second_gqa_error <= 1e-6F, "token2 GQA sublayer mismatch");
    require(second_moe_error <= 0.03F, "token2 MoE sublayer mismatch");
    require(second_final_error <= 0.03F, "token2 full layer mismatch");

    require(
        layer->gqa_state().position == ref_gqa->state().position,
        "token2 GQA position mismatch");
    require(
        max_abs_error(
            layer->gqa_state().k_cache,
            ref_gqa->state().k_cache) <= 1e-7F,
        "token2 K cache mismatch");
    require(
        max_abs_error(
            layer->gqa_state().v_cache,
            ref_gqa->state().v_cache) <= 1e-7F,
        "token2 V cache mismatch");

    layer->reset_state();
    ref_gqa->reset_state();
    require(layer->gqa_state().position == 0U, "reset position mismatch");
    require(layer->gqa_state().k_cache.empty(), "reset K cache mismatch");
    require(layer->gqa_state().v_cache.empty(), "reset V cache mismatch");

    const auto expected_reset = run_reference(x1);
    const auto actual_reset = layer->run(
        *context,
        full_gpu_cache,
        x1);
    require(actual_reset.executed, actual_reset.diagnostic);
    const auto reset_error =
        max_abs_error(actual_reset.values, expected_reset.moe.values);
    require(reset_error <= 0.03F, "reset full-attention layer mismatch");

    const auto bad = layer->run(
        *context,
        full_gpu_cache,
        std::span<const float>(x1).first(kHidden - 1U));
    require(!bad.executed, "invalid hidden geometry must be rejected");

    const auto stats = full_gpu_cache.stats();
    require(stats.loads > 0U, "full-attention layer must stream experts");
    require(
        stats.resident_entries > 0U,
        "full-attention layer must retain Vulkan expert residents");

    fs::remove_all(root);
    std::cout
        << "OSM-32 full-attention decoder layer: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  semantics=h=x+GQA(RMSNorm(x)); "
        << "out=h+MoE(RMSNorm(h))\n"
        << "  token1_gqa_error=" << first_gqa_error << "\n"
        << "  token1_final_error=" << first_final_error << "\n"
        << "  token2_gqa_error=" << second_gqa_error << "\n"
        << "  token2_final_error=" << second_final_error << "\n"
        << "  reset_final_error=" << reset_error << "\n"
        << "  streamed_expert_loads=" << stats.loads << "\n"
        << "  persistent_KV_cache=PASS\n"
        << "  Vulkan_MoE_cache=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-32 full-attention decoder layer: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
