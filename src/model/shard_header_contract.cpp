#include "orbi/streammoe/model/shard_header_contract.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::size_t bytes_per_element(const std::string& dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8U;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4U;
  if (dtype == "F16" || dtype == "BF16" ||
      dtype == "I16" || dtype == "U16") return 2U;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1U;
  throw std::runtime_error(
      "shard range manifest: unsupported dtype: " + dtype);
}

std::uint64_t checked_elements(const std::vector<std::size_t>& shape) {
  std::uint64_t product = 1U;
  for (const auto dim : shape) {
    if (dim == 0U) {
      throw std::runtime_error(
          "shard range manifest: tensor shape contains zero dimension");
    }
    if (product >
        std::numeric_limits<std::uint64_t>::max() /
            static_cast<std::uint64_t>(dim)) {
      throw std::runtime_error(
          "shard range manifest: tensor shape overflows uint64");
    }
    product *= static_cast<std::uint64_t>(dim);
  }
  return product;
}

std::uint64_t required_u64(
    const json& value,
    const char* key) {
  try {
    return value.at(key).get<std::uint64_t>();
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("shard range manifest: malformed field ") +
        key + ": " + e.what());
  }
}

}  // namespace

const SafetensorsRangeTensorProbe&
SafetensorsRangeManifest::tensor(const std::string& name) const {
  const auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::runtime_error(
        "shard range manifest: missing probed tensor: " + name);
  }
  return it->second;
}

SafetensorsRangeManifest inspect_safetensors_range_manifest(
    const std::filesystem::path& manifest_path) {
  std::ifstream input(manifest_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "shard range manifest: unable to open " + manifest_path.string());
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("shard range manifest: invalid JSON: ") + e.what());
  }

  if (root.value("schema_version", 0U) != 1U) {
    throw std::runtime_error(
        "shard range manifest: unsupported schema_version");
  }

  SafetensorsRangeManifest out;
  try {
    out.model = root.at("model").get<std::string>();
    out.snapshot = root.at("snapshot").get<std::string>();
    out.selected_tensors =
        root.at("selected_tensors").get<std::vector<std::string>>();
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("shard range manifest: malformed root: ") + e.what());
  }

  if (out.model.empty() || out.snapshot.empty()) {
    throw std::runtime_error(
        "shard range manifest: model and snapshot must be non-empty");
  }
  if (out.selected_tensors.empty()) {
    throw std::runtime_error(
        "shard range manifest: selected_tensors must be non-empty");
  }

  std::set<std::string> unique_selected(
      out.selected_tensors.begin(),
      out.selected_tensors.end());
  if (unique_selected.size() != out.selected_tensors.size()) {
    throw std::runtime_error(
        "shard range manifest: selected_tensors contains duplicates");
  }

  const auto& shards = root.at("shards");
  if (!shards.is_array() || shards.empty()) {
    throw std::runtime_error(
        "shard range manifest: shards must be a non-empty array");
  }

  std::set<std::string> shard_names;
  for (const auto& shard_json : shards) {
    SafetensorsRangeShardProbe shard;
    try {
      shard.filename = shard_json.at("filename").get<std::string>();
      shard.file_size = required_u64(shard_json, "file_size");
      shard.header_size = required_u64(shard_json, "header_size");
      shard.fetched_bytes = required_u64(shard_json, "fetched_bytes");
      shard.tensor_count =
          shard_json.at("tensor_count").get<std::size_t>();
    } catch (const json::exception& e) {
      throw std::runtime_error(
          std::string("shard range manifest: malformed shard: ") + e.what());
    }

    if (shard.filename.empty() ||
        !shard_names.insert(shard.filename).second) {
      throw std::runtime_error(
          "shard range manifest: duplicate or empty shard filename");
    }
    if (shard.header_size == 0U ||
        shard.file_size <= 8U + shard.header_size) {
      throw std::runtime_error(
          "shard range manifest: invalid shard/header size");
    }
    if (shard.fetched_bytes != 8U + shard.header_size) {
      throw std::runtime_error(
          "shard range manifest: fetched_bytes must equal prefix + header");
    }
    if (shard.tensor_count == 0U) {
      throw std::runtime_error(
          "shard range manifest: shard tensor_count must be non-zero");
    }

    if (out.total_fetched_bytes >
        std::numeric_limits<std::uint64_t>::max() - shard.fetched_bytes) {
      throw std::runtime_error(
          "shard range manifest: total fetched byte count overflows");
    }
    out.total_fetched_bytes += shard.fetched_bytes;

    const auto payload_size =
        shard.file_size - (8U + shard.header_size);
    const auto& tensors = shard_json.at("selected_tensor_metadata");
    if (!tensors.is_object()) {
      throw std::runtime_error(
          "shard range manifest: selected_tensor_metadata must be an object");
    }

    for (auto it = tensors.begin(); it != tensors.end(); ++it) {
      SafetensorsRangeTensorProbe tensor;
      tensor.name = it.key();
      tensor.shard = shard.filename;
      try {
        tensor.dtype = it.value().at("dtype").get<std::string>();
        tensor.shape =
            it.value().at("shape").get<std::vector<std::size_t>>();
        const auto offsets =
            it.value().at("data_offsets")
                .get<std::vector<std::uint64_t>>();
        if (offsets.size() != 2U) {
          throw std::runtime_error(
              "shard range manifest: data_offsets must contain two values");
        }
        tensor.data_begin = offsets[0];
        tensor.data_end = offsets[1];
      } catch (const json::exception& e) {
        throw std::runtime_error(
            "shard range manifest: malformed tensor " + tensor.name +
            ": " + e.what());
      }

      if (tensor.data_begin >= tensor.data_end ||
          tensor.data_end > payload_size) {
        throw std::runtime_error(
            "shard range manifest: tensor range exceeds shard payload: " +
            tensor.name);
      }

      const auto elements = checked_elements(tensor.shape);
      const auto bpe = bytes_per_element(tensor.dtype);
      if (elements >
          std::numeric_limits<std::uint64_t>::max() /
              static_cast<std::uint64_t>(bpe)) {
        throw std::runtime_error(
            "shard range manifest: tensor byte size overflows: " +
            tensor.name);
      }
      if (elements * static_cast<std::uint64_t>(bpe) != tensor.byte_size()) {
        throw std::runtime_error(
            "shard range manifest: dtype/shape disagree with byte range: " +
            tensor.name);
      }

      if (!out.tensors.emplace(tensor.name, tensor).second) {
        throw std::runtime_error(
            "shard range manifest: tensor appears in multiple shard probes: " +
            tensor.name);
      }
    }

    out.shards.push_back(std::move(shard));
  }

  for (const auto& name : out.selected_tensors) {
    if (!out.tensors.contains(name)) {
      throw std::runtime_error(
          "shard range manifest: selected tensor was not probed: " + name);
    }
  }

  return out;
}

void validate_qwen_shard_range_pilot(
    const SafetensorsRangeManifest& manifest,
    const QwenShardedCheckpointContract& checkpoint) {
  std::set<std::string> valid_shards(
      checkpoint.shards.begin(),
      checkpoint.shards.end());

  for (const auto& shard : manifest.shards) {
    if (!valid_shards.contains(shard.filename)) {
      throw std::runtime_error(
          "shard range pilot: probed shard is absent from checkpoint index: " +
          shard.filename);
    }
  }

  for (const auto& name : manifest.selected_tensors) {
    const auto mapped = checkpoint.weight_map.find(name);
    if (mapped == checkpoint.weight_map.end()) {
      throw std::runtime_error(
          "shard range pilot: selected tensor absent from checkpoint index: " +
          name);
    }
    const auto& probe = manifest.tensor(name);
    if (probe.shard != mapped->second) {
      throw std::runtime_error(
          "shard range pilot: probed shard disagrees with weight_map for " +
          name);
    }
  }
}

}  // namespace orbi::streammoe
