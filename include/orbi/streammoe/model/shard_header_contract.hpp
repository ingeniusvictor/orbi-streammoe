#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "orbi/streammoe/model/sharded_checkpoint_contract.hpp"

namespace orbi::streammoe {

struct SafetensorsRangeTensorProbe {
  std::string name;
  std::string shard;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::uint64_t data_begin{};
  std::uint64_t data_end{};

  [[nodiscard]] std::uint64_t byte_size() const noexcept {
    return data_end - data_begin;
  }
};

struct SafetensorsRangeShardProbe {
  std::string filename;
  std::uint64_t file_size{};
  std::uint64_t header_size{};
  std::uint64_t fetched_bytes{};
  std::size_t tensor_count{};
};

struct SafetensorsRangeManifest {
  std::string model;
  std::string snapshot;
  std::vector<std::string> selected_tensors;
  std::vector<SafetensorsRangeShardProbe> shards;
  std::unordered_map<std::string, SafetensorsRangeTensorProbe> tensors;
  std::uint64_t total_fetched_bytes{};

  [[nodiscard]] const SafetensorsRangeTensorProbe& tensor(
      const std::string& name) const;
};

/// Parse and validate a manifest produced from HTTP safetensors header ranges.
///
/// The manifest contains only metadata recovered from the 8-byte safetensors
/// prefix plus the exact JSON header range. No tensor payload is required.
[[nodiscard]] SafetensorsRangeManifest inspect_safetensors_range_manifest(
    const std::filesystem::path& manifest_path);

/// Cross-check probed tensor metadata against the sharded checkpoint index.
///
/// This proves each selected tensor was discovered in the exact shard named by
/// model.safetensors.index.json and that every selected tensor was actually
/// covered by the range pilot.
void validate_qwen_shard_range_pilot(
    const SafetensorsRangeManifest& manifest,
    const QwenShardedCheckpointContract& checkpoint);

}  // namespace orbi::streammoe
