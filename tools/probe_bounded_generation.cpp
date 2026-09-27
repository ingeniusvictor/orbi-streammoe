#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/session/greedy_text_session.hpp"
#include "orbi/streammoe/storage/qpack_expert_storage.hpp"
#include "orbi/streammoe/tokenizer/tokenizers_cpp_qwen_tokenizer.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::size_t parse_size(const char* value, const char* label) {
  try {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != std::string(value).size() || parsed == 0U ||
        parsed > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("invalid");
    }
    return static_cast<std::size_t>(parsed);
  } catch (...) {
    throw std::runtime_error(std::string("invalid ") + label);
  }
}

std::string read_text(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("unable to open prompt file");
  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr
        << "usage: orbi_streammoe_probe_bounded_generation "
        << "<checkpoint_dir> <tokenizer_asset_dir> <prompt_file> "
        << "<max_prompt_tokens> <max_new_tokens> <lm_head_chunk_rows> "
        << "<host_cache_bytes> <gpu_cache_bytes>\n";
    return 2;
  }

  try {
    if (!qwen_tokenizers_cpp_backend_available()) {
      throw std::runtime_error(
          "bounded generation requires ORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP=ON");
    }

    const fs::path checkpoint_dir(argv[1]);
    const fs::path tokenizer_dir(argv[2]);
    const fs::path prompt_file(argv[3]);
    const auto max_prompt_tokens = parse_size(argv[4], "max_prompt_tokens");
    const auto max_new_tokens = parse_size(argv[5], "max_new_tokens");
    const auto lm_head_chunk_rows = parse_size(argv[6], "lm_head_chunk_rows");
    const auto host_cache_bytes = parse_size(argv[7], "host_cache_bytes");
    const auto gpu_cache_bytes = parse_size(argv[8], "gpu_cache_bytes");

    if (max_new_tokens < 2U || max_new_tokens > 8U) {
      throw std::runtime_error("OSM-43D requires 2 <= max_new_tokens <= 8");
    }

    const auto prompt = read_text(prompt_file);
    if (prompt.empty()) throw std::runtime_error("prompt must not be empty");

    QpackReader qpack(checkpoint_dir);
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);
    if (config.num_hidden_layers != 48U) {
      throw std::runtime_error("bounded generation requires 48 layers");
    }

    std::string diagnostic;
    auto tokenizer = QwenTokenizersCppTokenizer::create(tokenizer_dir, &diagnostic);
    if (!tokenizer.has_value() || !tokenizer->valid()) {
      throw std::runtime_error("official tokenizer creation failed: " + diagnostic);
    }

    const auto encoded = tokenizer->encode(prompt, true);
    if (!encoded.encoded || encoded.token_ids.empty()) {
      throw std::runtime_error("official tokenizer encode failed: " + encoded.diagnostic);
    }
    if (encoded.token_ids.size() > max_prompt_tokens) {
      throw std::runtime_error("encoded prompt exceeds max_prompt_tokens");
    }
    if (std::any_of(
            encoded.token_ids.begin(), encoded.token_ids.end(),
            [&](std::size_t token) { return token >= config.vocab_size; })) {
      throw std::runtime_error("official tokenizer produced token outside model vocabulary");
    }

    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      throw std::runtime_error("bounded generation requires Vulkan: " + diagnostic);
    }

    auto session = QwenGreedyTextSession::create(
        *context, checkpoint, config, &diagnostic);
    if (!session.has_value() || !session->valid()) {
      throw std::runtime_error("greedy text session creation failed: " + diagnostic);
    }

    QpackExpertStorage storage(checkpoint_dir);
    ExpertCache host_cache(storage, host_cache_bytes, 1U);
    VulkanResidentExpertCache gpu_cache(
        *context, storage.reader(), host_cache, gpu_cache_bytes);

    QwenGreedyTextSessionOptions options;
    options.generation.max_new_tokens = max_new_tokens;
    options.generation.lm_head_chunk_rows = lm_head_chunk_rows;
    options.generation.reset_before_prompt = true;
    options.tokenizer.add_special_tokens = true;
    options.tokenizer.skip_special_tokens = true;
    options.tokenizer.use_tokenizer_eos_as_stop = true;

    const auto result = session->generate(
        checkpoint, *context, gpu_cache, *tokenizer, prompt, options);
    if (!result.completed || !result.token_session_executed) {
      throw std::runtime_error("bounded generation failed: " + result.diagnostic);
    }
    if (result.prompt_tokens != encoded.token_ids) {
      throw std::runtime_error("prompt tokens differ from official tokenizer");
    }
    if (result.generated_tokens.size() < 2U ||
        result.generated_tokens.size() > max_new_tokens) {
      throw std::runtime_error("bounded generation did not produce 2..max_new_tokens tokens");
    }
    if (result.generated_logits.size() != result.generated_tokens.size()) {
      throw std::runtime_error("generated token/logit cardinality mismatch");
    }
    if (!std::all_of(
            result.generated_logits.begin(),
            result.generated_logits.end(),
            [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error("generated logits contain non-finite values");
    }

    const auto expected_steps =
        result.prompt_tokens.size() + result.generated_tokens.size() - 1U;
    if (result.model_steps != expected_steps) {
      throw std::runtime_error("autoregressive model-step accounting mismatch");
    }

    const auto host_stats = host_cache.stats();
    const auto gpu_stats = gpu_cache.stats();
    if (host_stats.misses == 0U || gpu_stats.loads == 0U) {
      throw std::runtime_error("bounded generation did not stream routed experts");
    }

    json output = {
        {"stage", "official-bounded-multitoken-generation"},
        {"executed", true},
        {"checkpoint_dir", fs::absolute(checkpoint_dir).generic_string()},
        {"tokenizer_asset_dir", fs::absolute(tokenizer_dir).generic_string()},
        {"prompt_file", fs::absolute(prompt_file).generic_string()},
        {"prompt_utf8_bytes", prompt.size()},
        {"prompt_token_count", result.prompt_tokens.size()},
        {"prompt_token_ids", result.prompt_tokens},
        {"generated_token_count", result.generated_tokens.size()},
        {"generated_token_ids", result.generated_tokens},
        {"generated_logits", result.generated_logits},
        {"generated_text", result.text},
        {"model_steps", result.model_steps},
        {"stop_reason", result.stop_reason == QwenGreedySessionStopReason::stop_token
            ? "stop_token"
            : "max_new_tokens"},
        {"model", {
            {"hidden_size", session->token_session()->model_shell()->hidden_size()},
            {"vocab_size", session->vocab_size()},
            {"layer_count", session->token_session()->model_shell()->layer_count()},
        }},
        {"limits", {
            {"max_prompt_tokens", max_prompt_tokens},
            {"max_new_tokens", max_new_tokens},
            {"lm_head_chunk_rows", lm_head_chunk_rows},
            {"host_cache_bytes", host_cache_bytes},
            {"gpu_cache_bytes", gpu_cache_bytes},
        }},
        {"host_cache", {
            {"hits", host_stats.hits},
            {"misses", host_stats.misses},
            {"budget_bytes", host_stats.budget_bytes},
        }},
        {"gpu_cache", {
            {"hits", gpu_stats.hits},
            {"misses", gpu_stats.misses},
            {"loads", gpu_stats.loads},
            {"evictions", gpu_stats.evictions},
            {"resident_entries", gpu_stats.resident_entries},
            {"resident_bytes", gpu_stats.resident_bytes},
            {"budget_bytes", gpu_stats.budget_bytes},
        }},
        {"vulkan", {
            {"device_name", context->info().device_name},
            {"vendor_id", context->info().vendor_id},
            {"device_id", context->info().device_id},
            {"api_version", context->info().api_version},
            {"queue_family_index", context->info().queue_family_index},
        }},
        {"claims", {
            {"official_tokenizer_used", true},
            {"text_prompt_tokenized", true},
            {"teacher_forced_prompt_executed", true},
            {"checkpoint_backed_48_layer_inference_executed", true},
            {"bounded_multitoken_generation_validated", true},
            {"generated_token_sequence_decoded", true},
            {"chat_template_validated", false},
            {"tokens_per_second_measured", false},
            {"ram_vram_profile_measured", false},
        }},
    };

    std::cout << output.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "OSM-43D bounded multi-token generation gate: FAIL: "
              << e.what() << "\n";
    return 1;
  }
}
