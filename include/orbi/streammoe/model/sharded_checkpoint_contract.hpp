#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace orbi::streammoe {

struct QwenShardedCheckpointContract {
  std::size_t hidden_size{};
  std::size_t vocab_size{};
  std::size_t num_hidden_layers{};
  std::size_t full_attention_interval{};
  std::size_t num_experts{};
  std::size_t num_experts_per_tok{};
  std::uint64_t total_size_bytes{};
  std::size_t tensor_count{};
  std::size_t shard_count{};
  std::vector<std::string> shards;
  std::unordered_map<std::string, std::string> weight_map;

  [[nodiscard]] bool is_linear_layer(std::size_t layer) const noexcept {
    return ((layer + 1U) % full_attention_interval) != 0U;
  }
};

/// Validate a Hugging Face sharded Qwen3-Next checkpoint without reading any
/// tensor payload. Only config.json and model.safetensors.index.json are read.
[[nodiscard]] QwenShardedCheckpointContract
inspect_qwen3_next_sharded_checkpoint(
    const std::filesystem::path& model_dir);

}  // namespace orbi::streammoe
