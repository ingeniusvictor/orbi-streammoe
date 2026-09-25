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
#include "orbi/streammoe/session/greedy_token_session.hpp"
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



int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm36a-token-session";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);

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
          << "OSM-36A autoregressive token session: PASS "
          << "(no compute device on host)\n";
      return 0;
    }

    auto direct = QwenCheckpointModelShell::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    require(direct.has_value(), diagnostic);

    auto session = QwenGreedyTokenSession::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    require(session.has_value(), diagnostic);
    require(session->valid(), "session must be valid");
    require(session->vocab_size() == 16U, "session vocab mismatch");

    QpackExpertStorage storage(root);

    ExpertCache direct_host(
        storage,
        storage.expert_stride_bytes() * 12U,
        1U);
    VulkanResidentExpertCache direct_gpu(
        *context,
        storage.reader(),
        direct_host,
        65536U);

    ExpertCache session_host(
        storage,
        storage.expert_stride_bytes() * 12U,
        1U);
    VulkanResidentExpertCache session_gpu(
        *context,
        storage.reader(),
        session_host,
        65536U);

    const std::vector<std::size_t> prompt{5U, 7U};
    const std::size_t max_new = 3U;
    const std::size_t chunk_rows = 3U;

    direct->reset_state();
    std::vector<std::size_t> expected_tokens;
    std::vector<float> expected_logits;

    QwenCheckpointModelStepResult step;
    for (const auto token : prompt) {
      step = direct->step_greedy(
          checkpoint,
          *context,
          direct_gpu,
          token,
          chunk_rows);
      require(step.executed, step.diagnostic);
    }

    expected_tokens.push_back(step.greedy.token_id);
    expected_logits.push_back(step.greedy.logit);

    while (expected_tokens.size() < max_new) {
      step = direct->step_greedy(
          checkpoint,
          *context,
          direct_gpu,
          expected_tokens.back(),
          chunk_rows);
      require(step.executed, step.diagnostic);
      expected_tokens.push_back(step.greedy.token_id);
      expected_logits.push_back(step.greedy.logit);
    }

    QwenGreedySessionOptions options;
    options.max_new_tokens = max_new;
    options.lm_head_chunk_rows = chunk_rows;
    options.reset_before_prompt = true;

    const auto result = session->generate(
        checkpoint,
        *context,
        session_gpu,
        prompt,
        options);

    require(result.executed, result.diagnostic);
    require(
        result.stop_reason ==
            QwenGreedySessionStopReason::max_new_tokens,
        "session must stop on max_new_tokens");
    require(result.prompt_tokens == prompt, "prompt echo mismatch");
    require(
        result.generated_tokens == expected_tokens,
        "autoregressive generated token sequence mismatch");
    require(
        result.generated_logits.size() == expected_logits.size(),
        "autoregressive generated logit count mismatch");
    for (std::size_t i = 0U; i < expected_logits.size(); ++i) {
      require(
          std::fabs(result.generated_logits[i] - expected_logits[i]) <= 1e-5F,
          "autoregressive generated logit mismatch");
    }
    require(
        result.model_steps ==
            prompt.size() + result.generated_tokens.size() - 1U,
        "autoregressive model-step accounting mismatch");

    QwenGreedySessionOptions stop_options = options;
    stop_options.max_new_tokens = 5U;
    stop_options.stop_token_ids = {expected_tokens.front()};

    const auto stopped = session->generate(
        checkpoint,
        *context,
        session_gpu,
        prompt,
        stop_options);
    require(stopped.executed, stopped.diagnostic);
    require(
        stopped.stop_reason == QwenGreedySessionStopReason::stop_token,
        "session stop-token reason mismatch");
    require(
        stopped.generated_tokens.size() == 1U &&
        stopped.generated_tokens.front() == expected_tokens.front(),
        "session must stop immediately on first generated stop token");
    require(
        stopped.model_steps == prompt.size(),
        "stop-token prompt step count mismatch");

    QwenGreedySessionOptions one_options = options;
    one_options.max_new_tokens = 1U;

    const auto reset_again = session->generate(
        checkpoint,
        *context,
        session_gpu,
        prompt,
        one_options);
    require(reset_again.executed, reset_again.diagnostic);
    require(
        reset_again.generated_tokens.front() == expected_tokens.front(),
        "session reset must reproduce first continuation token");

    const std::vector<std::size_t> first_prompt{5U};
    const auto first_half = session->generate(
        checkpoint,
        *context,
        session_gpu,
        first_prompt,
        one_options);
    require(first_half.executed, first_half.diagnostic);

    QwenGreedySessionOptions continuation_options = one_options;
    continuation_options.reset_before_prompt = false;
    const std::vector<std::size_t> second_prompt{7U};
    const auto continued = session->generate(
        checkpoint,
        *context,
        session_gpu,
        second_prompt,
        continuation_options);
    require(continued.executed, continued.diagnostic);
    require(
        continued.generated_tokens.front() == expected_tokens.front(),
        "non-reset continuation must preserve sequence state");

    const std::vector<std::size_t> empty_prompt;
    const auto empty = session->generate(
        checkpoint,
        *context,
        session_gpu,
        empty_prompt,
        continuation_options);
    require(!empty.executed, "empty prompt must fail");
    require(
        empty.stop_reason ==
            QwenGreedySessionStopReason::invalid_request,
        "empty prompt stop reason mismatch");
    require(empty.model_steps == 0U, "empty prompt must not execute model");

    const auto continued_after_invalid = session->generate(
        checkpoint,
        *context,
        session_gpu,
        second_prompt,
        continuation_options);
    require(
        continued_after_invalid.executed,
        continued_after_invalid.diagnostic);

    QwenGreedySessionOptions zero_options = one_options;
    zero_options.max_new_tokens = 0U;
    const auto zero = session->generate(
        checkpoint,
        *context,
        session_gpu,
        prompt,
        zero_options);
    require(!zero.executed, "zero max_new_tokens must fail");

    const std::vector<std::size_t> bad_prompt{config.vocab_size};
    const auto bad_token = session->generate(
        checkpoint,
        *context,
        session_gpu,
        bad_prompt,
        one_options);
    require(!bad_token.executed, "out-of-vocab prompt token must fail");

    QwenGreedySessionOptions bad_stop_options = one_options;
    bad_stop_options.stop_token_ids = {config.vocab_size};
    const auto bad_stop = session->generate(
        checkpoint,
        *context,
        session_gpu,
        prompt,
        bad_stop_options);
    require(!bad_stop.executed, "out-of-vocab stop token must fail");

    QwenGreedySessionOptions bad_chunk_options = one_options;
    bad_chunk_options.lm_head_chunk_rows = 0U;
    const auto bad_chunk = session->generate(
        checkpoint,
        *context,
        session_gpu,
        prompt,
        bad_chunk_options);
    require(!bad_chunk.executed, "zero LM-head chunk must fail");

    const auto stats = session_gpu.stats();
    require(stats.loads > 0U, "session must stream routed experts");

    fs::remove_all(root);
    std::cout
        << "OSM-36A autoregressive token session: PASS\n"
        << "  prompt_tokens=2\n"
        << "  generated_tokens=" << result.generated_tokens.size() << "\n"
        << "  model_steps=" << result.model_steps << "\n"
        << "  direct_step_parity=PASS\n"
        << "  stop_token=PASS\n"
        << "  max_new_tokens=PASS\n"
        << "  reset_reproducibility=PASS\n"
        << "  continuation_without_reset=PASS\n"
        << "  invalid_request_guards=PASS\n"
        << "  expert_streaming=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-36A autoregressive token session: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
