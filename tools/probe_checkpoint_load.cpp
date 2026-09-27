#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/model/checkpoint_model_shell.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: orbi_streammoe_probe_checkpoint_load <checkpoint_dir>\n";
    return 2;
  }

  try {
    const fs::path root(argv[1]);
    QpackReader qpack(root);
    QpackMlxCheckpoint checkpoint(qpack);
    const auto config = parse_qwen3_next_dense_config(qpack);

    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      throw std::runtime_error(
          "runtime load gate requires Vulkan compute context: " + diagnostic);
    }

    auto model = QwenCheckpointModelShell::create(
        *context,
        checkpoint,
        config,
        &diagnostic);
    if (!model.has_value() || !model->valid()) {
      throw std::runtime_error(
          "checkpoint model shell load failed: " + diagnostic);
    }

    json result = {
        {"stage", "official-full-checkpoint-runtime-load"},
        {"loaded", true},
        {"checkpoint_dir", fs::absolute(root).generic_string()},
        {"model_name", qpack.manifest().model_name},
        {"source_checkpoint", qpack.manifest().source_checkpoint},
        {"hidden_size", model->hidden_size()},
        {"vocab_size", model->vocab_size()},
        {"layer_count", model->layer_count()},
        {"vulkan", {
            {"device_name", context->info().device_name},
            {"vendor_id", context->info().vendor_id},
            {"device_id", context->info().device_id},
            {"api_version", context->info().api_version},
            {"queue_family_index", context->info().queue_family_index},
        }},
        {"claims", {
            {"checkpoint_opened", true},
            {"dense_inventory_bound", true},
            {"decoder_stack_created", true},
            {"vulkan_context_created", true},
            {"inference_executed", false},
            {"token_generated", false},
        }},
    };
    std::cout << result.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-43A official runtime load gate: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
