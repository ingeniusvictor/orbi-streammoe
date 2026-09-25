#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/model/checkpoint_gqa.hpp"
#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"
#include "orbi/streammoe/model/checkpoint_gqa_sublayer.hpp"
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
      fs::temp_directory_path() / "orbi-streammoe-osm31c-gqa-sublayer";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);
    const auto binding =
        bind_qwen3_next_dense_layer(checkpoint, config, 3);

    require(!binding.is_linear, "fixture layer must be full attention");
    require(binding.attention.has_value(), "attention binding missing");

    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      if (const char* require_env =
              std::getenv("ORBI_REQUIRE_VULKAN_COMPUTE");
          require_env != nullptr && std::string(require_env) != "0") {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no Vulkan context: " +
            diagnostic);
      }
      fs::remove_all(root);
      std::cout
          << "OSM-31C checkpoint GQA residual sublayer: PASS "
          << "(no compute device on host)\n";
      return 0;
    }

    auto sublayer = QwenCheckpointGqaSublayer::create(
        *context,
        binding,
        config,
        &diagnostic);
    require(sublayer.has_value(), diagnostic);
    require(sublayer->valid(), "GQA sublayer must be valid");
    require(sublayer->layer_index() == 3U, "GQA sublayer layer mismatch");
    require(sublayer->hidden_size() == config.hidden_size,
            "GQA sublayer hidden size mismatch");

    auto reference = QwenCheckpointGqa::create(
        binding,
        config,
        &diagnostic);
    require(reference.has_value(), diagnostic);
    require(reference->valid(), "reference checkpoint GQA must be valid");

    const auto run_reference =
        [&](std::span<const float> residual) {
          std::vector<float> normalized(
              residual.begin(), residual.end());
          cpu::rms_norm_inplace(
              normalized,
              1U,
              config.hidden_size,
              binding.input_norm,
              config.rms_norm_eps);

          const auto branch = reference->run(normalized);
          require(branch.executed, branch.diagnostic);

          std::vector<float> output(residual.size());
          for (std::size_t i = 0; i < residual.size(); ++i) {
            output[i] = residual[i] + branch.values[i];
          }

          return std::tuple{
              std::move(normalized),
              branch,
              std::move(output),
          };
        };

    const std::vector<float> x1{
        0.5F, -0.25F, 0.75F, -0.5F,
        0.125F, -0.625F, 0.25F, 0.375F,
    };
    const std::vector<float> x2{
        -0.125F, 0.625F, -0.375F, 0.25F,
        0.5F, 0.125F, -0.75F, 0.875F,
    };

    const auto [norm1, branch1, expected1] = run_reference(x1);
    const auto actual1 = sublayer->run(*context, x1);
    require(actual1.executed, actual1.diagnostic);

    const auto norm_error1 =
        max_abs_error(actual1.normalized, norm1);
    const auto branch_error1 =
        max_abs_error(actual1.branch.values, branch1.values);
    const auto final_error1 =
        max_abs_error(actual1.values, expected1);

    require(norm_error1 <= 1e-5F, "token1 GQA RMSNorm parity failed");
    require(branch_error1 <= 1e-6F, "token1 GQA branch parity failed");
    require(final_error1 <= 1e-6F, "token1 GQA residual parity failed");
    require(
        sublayer->state().position == reference->state().position,
        "token1 GQA position mismatch");
    require(
        max_abs_error(
            sublayer->state().k_cache,
            reference->state().k_cache) <= 1e-7F,
        "token1 GQA K cache mismatch");
    require(
        max_abs_error(
            sublayer->state().v_cache,
            reference->state().v_cache) <= 1e-7F,
        "token1 GQA V cache mismatch");

    const auto [norm2, branch2, expected2] = run_reference(x2);
    const auto actual2 = sublayer->run(*context, x2);
    require(actual2.executed, actual2.diagnostic);

    const auto norm_error2 =
        max_abs_error(actual2.normalized, norm2);
    const auto branch_error2 =
        max_abs_error(actual2.branch.values, branch2.values);
    const auto final_error2 =
        max_abs_error(actual2.values, expected2);

    require(norm_error2 <= 1e-5F, "token2 GQA RMSNorm parity failed");
    require(branch_error2 <= 1e-6F, "token2 GQA branch parity failed");
    require(final_error2 <= 1e-6F, "token2 GQA residual parity failed");
    require(
        sublayer->state().position == reference->state().position,
        "token2 GQA position mismatch");
    require(
        max_abs_error(
            sublayer->state().k_cache,
            reference->state().k_cache) <= 1e-7F,
        "token2 GQA K cache mismatch");
    require(
        max_abs_error(
            sublayer->state().v_cache,
            reference->state().v_cache) <= 1e-7F,
        "token2 GQA V cache mismatch");

    sublayer->reset_state();
    reference->reset_state();
    require(sublayer->state().position == 0U, "reset GQA position mismatch");
    require(sublayer->state().k_cache.empty(), "reset GQA K cache mismatch");
    require(sublayer->state().v_cache.empty(), "reset GQA V cache mismatch");

    const auto [norm_reset, branch_reset, expected_reset] =
        run_reference(x1);
    const auto reset = sublayer->run(*context, x1);
    require(reset.executed, reset.diagnostic);
    require(
        max_abs_error(reset.normalized, norm_reset) <= 1e-5F,
        "reset GQA RMSNorm parity failed");
    require(
        max_abs_error(reset.branch.values, branch_reset.values) <= 1e-6F,
        "reset GQA branch parity failed");
    require(
        max_abs_error(reset.values, expected_reset) <= 1e-6F,
        "reset GQA residual parity failed");

    const auto bad = sublayer->run(
        *context,
        std::span<const float>(x1).first(config.hidden_size - 1U));
    require(!bad.executed, "invalid GQA residual shape must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-31C checkpoint GQA residual sublayer: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  input_norm_source=model.safetensors\n"
        << "  token1_norm_error=" << norm_error1 << "\n"
        << "  token1_branch_error=" << branch_error1 << "\n"
        << "  token1_final_error=" << final_error1 << "\n"
        << "  token2_norm_error=" << norm_error2 << "\n"
        << "  token2_branch_error=" << branch_error2 << "\n"
        << "  token2_final_error=" << final_error2 << "\n"
        << "  persistent_KV_cache=PASS\n"
        << "  reset_state=PASS\n"
        << "  residual_shape_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-31C checkpoint GQA residual sublayer: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
