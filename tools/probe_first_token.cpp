#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/model/checkpoint_model_shell.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/storage/qpack_expert_storage.hpp"

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

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr
        << "usage: orbi_streammoe_probe_first_token "
        << "<checkpoint_dir> <token_id> <lm_head_chunk_rows> "
        << "<host_cache_bytes> <gpu_cache_bytes>\n";
    return 2;
  }

  try {
    const fs::path root(argv[1]);
    const auto token_id = parse_size(argv[2], "token_id");
    const auto lm_head_chunk_rows = parse_size(argv[3], "lm_head_chunk_rows");
    const auto host_cache_bytes = parse_size(argv[4], "host_cache_bytes");
    const auto gpu_cache_bytes = parse_size(argv[5], "gpu_cache_bytes");

    QpackReader qpack(root);
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);
    if (token_id >= config.vocab_size) {
      throw std::runtime_error("input token exceeds official vocabulary");
    }
    if (config.num_hidden_layers != 48U) {
      throw std::runtime_error("official first-token gate requires 48 layers");
    }

    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      throw std::runtime_error(
          "first-token gate requires Vulkan compute context: " + diagnostic);
    }

    auto model = QwenCheckpointModelShell::create(
        *context, checkpoint, config, &diagnostic);
    if (!model.has_value() || !model->valid()) {
      throw std::runtime_error(
          "first-token model-shell creation failed: " + diagnostic);
    }

    QpackExpertStorage storage(root);
    ExpertCache host_cache(storage, host_cache_bytes, 1U);
    VulkanResidentExpertCache gpu_cache(
        *context, storage.reader(), host_cache, gpu_cache_bytes);

    const auto step = model->step_greedy(
        checkpoint,
        *context,
        gpu_cache,
        token_id,
        lm_head_chunk_rows);
    if (!step.executed) {
      throw std::runtime_error(
          "first-token inference failed: " + step.diagnostic);
    }
    if (step.greedy.token_id >= config.vocab_size) {
      throw std::runtime_error("generated token exceeds vocabulary");
    }
    if (!std::isfinite(step.greedy.logit)) {
      throw std::runtime_error("generated logit is not finite");
    }

    const auto host_stats = host_cache.stats();
    const auto gpu_stats = gpu_cache.stats();
    if (host_stats.misses == 0U || gpu_stats.loads == 0U) {
      throw std::runtime_error(
          "first-token gate did not stream routed experts");
    }

    json result = {
        {"stage", "official-first-token-inference"},
        {"executed", true},
        {"checkpoint_dir", fs::absolute(root).generic_string()},
        {"input_token_id", token_id},
        {"generated_token_id", step.greedy.token_id},
        {"generated_logit", step.greedy.logit},
        {"model", {
            {"hidden_size", model->hidden_size()},
            {"vocab_size", model->vocab_size()},
            {"layer_count", model->layer_count()},
        }},
        {"memory_limits", {
            {"host_cache_bytes", host_cache_bytes},
            {"gpu_cache_bytes", gpu_cache_bytes},
            {"lm_head_chunk_rows", lm_head_chunk_rows},
        }},
        {"host_cache", {
            {"hits", host_stats.hits},
            {"misses", host_stats.misses},
            {"allocated_slots", host_stats.allocated_slots},
            {"capacity_slots", host_stats.capacity_slots},
            {"logical_bytes", host_stats.logical_bytes},
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
            {"one_checkpoint_backed_step_executed", true},
            {"streamed_embedding_executed", true},
            {"decoder_48_layers_executed", true},
            {"final_rmsnorm_executed", true},
            {"streamed_lm_head_executed", true},
            {"first_token_generated", true},
            {"text_prompt_tokenized", false},
            {"multi_token_generation_validated", false},
            {"tokens_per_second_measured", false},
        }},
    };

    std::cout << result.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-43B official first-token inference gate: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
