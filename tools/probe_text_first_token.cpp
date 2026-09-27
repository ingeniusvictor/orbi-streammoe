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
  if (argc != 8) {
    std::cerr
        << "usage: orbi_streammoe_probe_text_first_token "
        << "<checkpoint_dir> <tokenizer_asset_dir> <prompt_file> "
        << "<max_prompt_tokens> <lm_head_chunk_rows> "
        << "<host_cache_bytes> <gpu_cache_bytes>\n";
    return 2;
  }

  try {
    if (!qwen_tokenizers_cpp_backend_available()) {
      throw std::runtime_error(
          "official text gate requires ORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP=ON");
    }

    const fs::path checkpoint_dir(argv[1]);
    const fs::path tokenizer_dir(argv[2]);
    const fs::path prompt_file(argv[3]);
    const auto max_prompt_tokens = parse_size(argv[4], "max_prompt_tokens");
    const auto lm_head_chunk_rows = parse_size(argv[5], "lm_head_chunk_rows");
    const auto host_cache_bytes = parse_size(argv[6], "host_cache_bytes");
    const auto gpu_cache_bytes = parse_size(argv[7], "gpu_cache_bytes");
    const auto prompt = read_text(prompt_file);
    if (prompt.empty()) throw std::runtime_error("prompt must not be empty");

    QpackReader qpack(checkpoint_dir);
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);
    if (config.num_hidden_layers != 48U) {
      throw std::runtime_error("official text gate requires 48 layers");
    }

    std::string diagnostic;
    auto tokenizer = QwenTokenizersCppTokenizer::create(
        tokenizer_dir, &diagnostic);
    if (!tokenizer.has_value() || !tokenizer->valid()) {
      throw std::runtime_error(
          "official tokenizer creation failed: " + diagnostic);
    }

    const auto encoded = tokenizer->encode(prompt, true);
    if (!encoded.encoded || encoded.token_ids.empty()) {
      throw std::runtime_error(
          "official tokenizer encode failed: " + encoded.diagnostic);
    }
    if (encoded.token_ids.size() > max_prompt_tokens) {
      throw std::runtime_error("encoded prompt exceeds max_prompt_tokens");
    }
    for (const auto token : encoded.token_ids) {
      if (token >= config.vocab_size) {
        throw std::runtime_error("official tokenizer produced token outside model vocabulary");
      }
    }

    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      throw std::runtime_error(
          "text first-token gate requires Vulkan: " + diagnostic);
    }

    auto session = QwenGreedyTextSession::create(
        *context, checkpoint, config, &diagnostic);
    if (!session.has_value() || !session->valid()) {
      throw std::runtime_error(
          "greedy text session creation failed: " + diagnostic);
    }

    QpackExpertStorage storage(checkpoint_dir);
    ExpertCache host_cache(storage, host_cache_bytes, 1U);
    VulkanResidentExpertCache gpu_cache(
        *context, storage.reader(), host_cache, gpu_cache_bytes);

    QwenGreedyTextSessionOptions options;
    options.generation.max_new_tokens = 1U;
    options.generation.lm_head_chunk_rows = lm_head_chunk_rows;
    options.generation.reset_before_prompt = true;
    options.tokenizer.add_special_tokens = true;
    options.tokenizer.skip_special_tokens = true;
    options.tokenizer.use_tokenizer_eos_as_stop = true;

    const auto result = session->generate(
        checkpoint,
        *context,
        gpu_cache,
        *tokenizer,
        prompt,
        options);
    if (!result.completed || !result.token_session_executed) {
      throw std::runtime_error(
          "text first-token generation failed: " + result.diagnostic);
    }
    if (result.prompt_tokens != encoded.token_ids) {
      throw std::runtime_error("text session prompt tokens differ from official tokenizer");
    }
    if (result.generated_tokens.size() != 1U ||
        result.generated_logits.size() != 1U) {
      throw std::runtime_error("text gate must produce exactly one token/logit");
    }
    if (!std::isfinite(result.generated_logits.front())) {
      throw std::runtime_error("generated logit is not finite");
    }
    if (result.model_steps != result.prompt_tokens.size()) {
      throw std::runtime_error("teacher-forced model-step accounting mismatch");
    }

    const auto host_stats = host_cache.stats();
    const auto gpu_stats = gpu_cache.stats();
    if (host_stats.misses == 0U || gpu_stats.loads == 0U) {
      throw std::runtime_error("text gate did not stream routed experts");
    }

    json output = {
        {"stage", "official-text-prompt-first-token"},
        {"executed", true},
        {"checkpoint_dir", fs::absolute(checkpoint_dir).generic_string()},
        {"tokenizer_asset_dir", fs::absolute(tokenizer_dir).generic_string()},
        {"prompt_file", fs::absolute(prompt_file).generic_string()},
        {"prompt_utf8_bytes", prompt.size()},
        {"prompt_token_count", result.prompt_tokens.size()},
        {"prompt_token_ids", result.prompt_tokens},
        {"generated_token_id", result.generated_tokens.front()},
        {"generated_logit", result.generated_logits.front()},
        {"generated_text", result.text},
        {"model_steps", result.model_steps},
        {"model", {
            {"hidden_size", session->token_session()->model_shell()->hidden_size()},
            {"vocab_size", session->vocab_size()},
            {"layer_count", session->token_session()->model_shell()->layer_count()},
        }},
        {"limits", {
            {"max_prompt_tokens", max_prompt_tokens},
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
            {"first_continuation_token_generated", true},
            {"multi_token_generation_validated", false},
            {"tokens_per_second_measured", false},
        }},
    };

    std::cout << output.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-43C official text-prompt first-token gate: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
