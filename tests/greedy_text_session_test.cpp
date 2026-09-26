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
#include "orbi/streammoe/session/greedy_text_session.hpp"
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




namespace {

class FixtureTextTokenizer final : public Tokenizer {
 public:
  std::optional<std::size_t> vocab{16U};
  bool fail_encode{};
  bool fail_decode{};

  [[nodiscard]] std::optional<std::size_t> vocab_size() const noexcept override {
    return vocab;
  }

  [[nodiscard]] TokenizerEncodeResult encode(
      std::string_view text,
      bool add_special_tokens) const override {
    if (fail_encode) {
      return {false, {}, "fixture encode failure"};
    }
    if (text != "hello") {
      return {false, {}, "unexpected fixture prompt"};
    }

    TokenizerEncodeResult result;
    result.encoded = true;
    result.token_ids = add_special_tokens
        ? std::vector<std::size_t>{1U, 5U, 7U}
        : std::vector<std::size_t>{5U, 7U};
    result.diagnostic = "fixture encoded";
    return result;
  }

  [[nodiscard]] TokenizerDecodeResult decode(
      std::span<const std::size_t> token_ids,
      bool skip_special_tokens) const override {
    if (fail_decode) {
      return {false, {}, "fixture decode failure"};
    }

    std::ostringstream out;
    if (!skip_special_tokens) out << "<RAW>";
    for (std::size_t i = 0U; i < token_ids.size(); ++i) {
      if (i != 0U) out << ",";
      out << token_ids[i];
    }

    return {true, out.str(), "fixture decoded"};
  }
};

std::string fixture_decode(
    std::span<const std::size_t> token_ids) {
  std::ostringstream out;
  for (std::size_t i = 0U; i < token_ids.size(); ++i) {
    if (i != 0U) out << ",";
    out << token_ids[i];
  }
  return out.str();
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm36c-text-session";

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
          << "OSM-36C greedy text session: PASS "
          << "(no compute device on host)\n";
      return 0;
    }

    auto direct = QwenGreedyTokenSession::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    require(direct.has_value(), diagnostic);

    auto text_session = QwenGreedyTextSession::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    require(text_session.has_value(), diagnostic);
    require(text_session->valid(), "text session must be valid");
    require(text_session->vocab_size() == 16U, "text session vocab mismatch");
    require(
        text_session->token_session() != nullptr,
        "text session must expose token runtime diagnostics");

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

    ExpertCache text_host(
        storage,
        storage.expert_stride_bytes() * 12U,
        1U);
    VulkanResidentExpertCache text_gpu(
        *context,
        storage.reader(),
        text_host,
        65536U);

    const std::vector<std::size_t> prompt{5U, 7U};

    QwenGreedySessionOptions token_options;
    token_options.max_new_tokens = 3U;
    token_options.lm_head_chunk_rows = 3U;
    token_options.reset_before_prompt = true;

    const auto expected = direct->generate(
        checkpoint,
        *context,
        direct_gpu,
        prompt,
        token_options);
    require(expected.executed, expected.diagnostic);

    FixtureTextTokenizer tokenizer;

    QwenGreedyTextSessionOptions text_options;
    text_options.generation = token_options;
    text_options.tokenizer.add_special_tokens = false;
    text_options.tokenizer.skip_special_tokens = true;
    text_options.tokenizer.use_tokenizer_eos_as_stop = false;

    const auto result = text_session->generate(
        checkpoint,
        *context,
        text_gpu,
        tokenizer,
        "hello",
        text_options);

    require(result.completed, result.diagnostic);
    require(
        result.status == QwenGreedyTextSessionStatus::completed,
        "text session completion status mismatch");
    require(
        result.token_session_executed,
        "text session must report numerical execution");
    require(result.prompt_tokens == prompt, "encoded prompt mismatch");
    require(
        result.generated_tokens == expected.generated_tokens,
        "text facade generated token mismatch");
    require(
        result.generated_logits.size() == expected.generated_logits.size(),
        "text facade generated logit count mismatch");
    for (std::size_t i = 0U; i < expected.generated_logits.size(); ++i) {
      require(
          std::fabs(
              result.generated_logits[i] -
              expected.generated_logits[i]) <= 1e-5F,
          "text facade generated logit mismatch");
    }
    require(
        result.model_steps == expected.model_steps,
        "text facade model-step accounting mismatch");
    require(
        result.stop_reason == expected.stop_reason,
        "text facade stop reason mismatch");
    require(
        result.text == fixture_decode(expected.generated_tokens),
        "decoded text mismatch");

    const auto reset_again = text_session->generate(
        checkpoint,
        *context,
        text_gpu,
        tokenizer,
        "hello",
        text_options);
    require(reset_again.completed, reset_again.diagnostic);
    require(
        reset_again.generated_tokens == result.generated_tokens,
        "reset text session must reproduce deterministic greedy output");

    tokenizer.vocab = 17U;
    const auto vocab_mismatch = text_session->generate(
        checkpoint,
        *context,
        text_gpu,
        tokenizer,
        "hello",
        text_options);
    require(!vocab_mismatch.completed, "vocab mismatch must fail");
    require(
        vocab_mismatch.status ==
            QwenGreedyTextSessionStatus::invalid_request,
        "vocab mismatch status mismatch");
    require(
        !vocab_mismatch.token_session_executed &&
        vocab_mismatch.model_steps == 0U,
        "oversized tokenizer domain must fail before model execution");
    tokenizer.vocab = 16U;

    tokenizer.fail_encode = true;
    const auto encode_failure = text_session->generate(
        checkpoint,
        *context,
        text_gpu,
        tokenizer,
        "hello",
        text_options);
    require(!encode_failure.completed, "encode failure must fail");
    require(
        encode_failure.status ==
            QwenGreedyTextSessionStatus::tokenizer_error,
        "encode failure status mismatch");
    require(
        !encode_failure.token_session_executed &&
        encode_failure.model_steps == 0U,
        "encode failure must happen before model execution");
    tokenizer.fail_encode = false;

    QwenGreedyTextSessionOptions invalid_options = text_options;
    invalid_options.generation.max_new_tokens = 0U;
    const auto invalid_generation = text_session->generate(
        checkpoint,
        *context,
        text_gpu,
        tokenizer,
        "hello",
        invalid_options);
    require(
        !invalid_generation.completed,
        "invalid generation options must fail");
    require(
        invalid_generation.status ==
            QwenGreedyTextSessionStatus::invalid_request,
        "invalid generation status mismatch");
    require(
        !invalid_generation.token_session_executed &&
        invalid_generation.model_steps == 0U,
        "invalid generation options must not advance model");

    tokenizer.fail_decode = true;
    const auto decode_failure = text_session->generate(
        checkpoint,
        *context,
        text_gpu,
        tokenizer,
        "hello",
        text_options);
    require(!decode_failure.completed, "decode failure must fail text request");
    require(
        decode_failure.status ==
            QwenGreedyTextSessionStatus::tokenizer_error,
        "decode failure status mismatch");
    require(
        decode_failure.token_session_executed,
        "decode failure must preserve evidence that model executed");
    require(
        decode_failure.generated_tokens == expected.generated_tokens,
        "decode failure must preserve generated token evidence");
    require(
        decode_failure.model_steps == expected.model_steps,
        "decode failure must preserve model-step evidence");
    tokenizer.fail_decode = false;

    text_session->reset();
    const auto after_explicit_reset = text_session->generate(
        checkpoint,
        *context,
        text_gpu,
        tokenizer,
        "hello",
        text_options);
    require(after_explicit_reset.completed, after_explicit_reset.diagnostic);
    require(
        after_explicit_reset.generated_tokens == expected.generated_tokens,
        "explicit reset must reproduce deterministic text continuation");

    const auto stats = text_gpu.stats();
    require(stats.loads > 0U, "text session must stream routed experts");

    fs::remove_all(root);
    std::cout
        << "OSM-36C greedy text session: PASS\n"
        << "  direct_token_parity=PASS\n"
        << "  prompt_token_passthrough=PASS\n"
        << "  generated_logit_passthrough=PASS\n"
        << "  model_step_accounting=PASS\n"
        << "  deterministic_reset=PASS\n"
        << "  vocab_mismatch_preflight=PASS\n"
        << "  encode_failure_preflight=PASS\n"
        << "  invalid_generation_guard=PASS\n"
        << "  decode_failure_evidence=PASS\n"
        << "  expert_streaming=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-36C greedy text session: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
