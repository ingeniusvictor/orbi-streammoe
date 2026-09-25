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
#include <tuple>
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
#include "orbi/streammoe/model/checkpoint_deltanet_sublayer.hpp"
#include "orbi/streammoe/model/checkpoint_gated_deltanet.hpp"
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


int main() {
  const auto root =
      fs::temp_directory_path() /
      "orbi-streammoe-osm29c-deltanet-residual";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);
    const auto binding =
        bind_qwen3_next_dense_layer(checkpoint, config, 0);

    require(binding.is_linear, "fixture layer must be linear");
    require(binding.delta.has_value(), "fixture DeltaNet binding missing");

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
          << "OSM-29C checkpoint DeltaNet residual sublayer: PASS "
          << "(no compute device on host)\n";
      return 0;
    }

    constexpr float kRmsEps = 1e-6F;

    auto sublayer = QwenCheckpointDeltaNetSublayer::create(
        *context,
        binding,
        config,
        kRmsEps,
        &diagnostic);
    require(sublayer.has_value(), diagnostic);
    require(sublayer->valid(), "DeltaNet sublayer must be valid");
    require(sublayer->layer_index() == 0U, "layer index mismatch");
    require(sublayer->hidden_size() == kHidden, "hidden size mismatch");

    auto reference = QwenCheckpointGatedDeltaNet::create(
        binding,
        config,
        kRmsEps,
        &diagnostic);
    require(reference.has_value(), diagnostic);
    require(reference->valid(), "reference DeltaNet must be valid");

    const std::vector<float> x1{
        0.5F, -0.25F, 0.75F, -0.5F,
        0.125F, -0.625F, 0.25F, 0.375F,
    };
    const std::vector<float> x2{
        -0.125F, 0.625F, -0.375F, 0.25F,
        0.5F, 0.125F, -0.75F, 0.875F,
    };

    const auto run_reference =
        [&](std::span<const float> residual) {
          std::vector<float> normalized(residual.begin(), residual.end());
          cpu::rms_norm_inplace(
              normalized,
              1U,
              kHidden,
              binding.input_norm,
              kRmsEps);

          const auto branch = reference->run(normalized);
          require(branch.executed, branch.diagnostic);

          std::vector<float> output(kHidden);
          for (std::size_t i = 0; i < kHidden; ++i) {
            output[i] = residual[i] + branch.values[i];
          }

          return std::tuple{
              std::move(normalized),
              branch,
              std::move(output),
          };
        };

    const auto [norm1, branch1, expected1] = run_reference(x1);
    const auto first = sublayer->run(*context, x1);
    require(first.executed, first.diagnostic);

    const auto norm_error1 =
        max_abs_error(first.normalized, norm1);
    const auto branch_error1 =
        max_abs_error(first.branch.values, branch1.values);
    const auto final_error1 =
        max_abs_error(first.values, expected1);

    require(norm_error1 <= 1e-5F, "token1 input RMSNorm parity failed");
    require(branch_error1 <= 1e-6F, "token1 DeltaNet branch parity failed");
    require(final_error1 <= 1e-6F, "token1 residual parity failed");

    require(
        max_abs_error(
            sublayer->state().conv_tail,
            reference->state().conv_tail) <= 1e-7F,
        "token1 conv-tail state mismatch");
    require(
        max_abs_error(
            sublayer->state().recurrent,
            reference->state().recurrent) <= 1e-7F,
        "token1 recurrent state mismatch");

    const auto [norm2, branch2, expected2] = run_reference(x2);
    const auto second = sublayer->run(*context, x2);
    require(second.executed, second.diagnostic);

    const auto norm_error2 =
        max_abs_error(second.normalized, norm2);
    const auto branch_error2 =
        max_abs_error(second.branch.values, branch2.values);
    const auto final_error2 =
        max_abs_error(second.values, expected2);

    require(norm_error2 <= 1e-5F, "token2 input RMSNorm parity failed");
    require(branch_error2 <= 1e-6F, "token2 DeltaNet branch parity failed");
    require(final_error2 <= 1e-6F, "token2 residual parity failed");

    require(
        max_abs_error(
            sublayer->state().conv_tail,
            reference->state().conv_tail) <= 1e-7F,
        "token2 conv-tail state mismatch");
    require(
        max_abs_error(
            sublayer->state().recurrent,
            reference->state().recurrent) <= 1e-7F,
        "token2 recurrent state mismatch");

    sublayer->reset_state();
    reference->reset_state();
    require(
        sublayer->state().conv_tail.empty() &&
        sublayer->state().recurrent.empty(),
        "sublayer reset must clear persistent state");

    const auto [norm_reset, branch_reset, expected_reset] =
        run_reference(x1);
    const auto reset = sublayer->run(*context, x1);
    require(reset.executed, reset.diagnostic);
    require(
        max_abs_error(reset.normalized, norm_reset) <= 1e-5F,
        "reset RMSNorm parity failed");
    require(
        max_abs_error(reset.branch.values, branch_reset.values) <= 1e-6F,
        "reset DeltaNet branch parity failed");
    require(
        max_abs_error(reset.values, expected_reset) <= 1e-6F,
        "reset residual output parity failed");

    const auto bad = sublayer->run(
        *context,
        std::span<const float>(x1).first(kHidden - 1U));
    require(!bad.executed, "invalid residual shape must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-29C checkpoint DeltaNet residual sublayer: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  input_norm_source=model.safetensors\n"
        << "  token1_norm_max_abs_error=" << norm_error1 << "\n"
        << "  token1_branch_max_abs_error=" << branch_error1 << "\n"
        << "  token1_final_max_abs_error=" << final_error1 << "\n"
        << "  token2_norm_max_abs_error=" << norm_error2 << "\n"
        << "  token2_branch_max_abs_error=" << branch_error2 << "\n"
        << "  token2_final_max_abs_error=" << final_error2 << "\n"
        << "  persistent_state=PASS\n"
        << "  reset_state=PASS\n"
        << "  residual_shape_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-29C checkpoint DeltaNet residual sublayer: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
