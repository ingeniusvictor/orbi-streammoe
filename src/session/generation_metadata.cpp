#include "orbi/streammoe/session/generation_metadata.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::optional<json> read_declared_json(
    const QpackReader& qpack,
    std::string_view relative_path,
    bool required) {
  const auto key = std::string(relative_path);
  const auto it = qpack.manifest().files.find(key);
  if (it == qpack.manifest().files.end()) {
    if (required) {
      throw std::runtime_error(
          "generation metadata: qpack manifest does not declare " + key);
    }
    return std::nullopt;
  }

  const auto path = qpack.container_dir() / std::filesystem::path(key);
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error(
        "generation metadata: unable to stat " + key);
  }
  if (actual != it->second) {
    throw std::runtime_error(
        "generation metadata: file size disagrees with qpack manifest: " +
        key);
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "generation metadata: unable to open " + key);
  }

  json value;
  try {
    input >> value;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "generation metadata: invalid JSON in " + key + ": " + e.what());
  }
  if (!value.is_object()) {
    throw std::runtime_error(
        "generation metadata: JSON root must be an object: " + key);
  }
  return value;
}

void append_eos_value(
    const json& value,
    std::vector<std::size_t>& out,
    std::string_view source) {
  const auto append_one =
      [&](const json& item) {
        if (!item.is_number_integer()) {
          throw std::runtime_error(
              "generation metadata: eos_token_id must contain integers in " +
              std::string(source));
        }
        const auto signed_value = item.get<std::int64_t>();
        if (signed_value < 0) {
          throw std::runtime_error(
              "generation metadata: eos_token_id must be non-negative in " +
              std::string(source));
        }
        const auto unsigned_value =
            static_cast<std::uint64_t>(signed_value);
        if (unsigned_value >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
          throw std::runtime_error(
              "generation metadata: eos_token_id exceeds size_t in " +
              std::string(source));
        }
        out.push_back(static_cast<std::size_t>(unsigned_value));
      };

  if (value.is_number_integer()) {
    append_one(value);
    return;
  }
  if (value.is_array()) {
    for (const auto& item : value) {
      append_one(item);
    }
    return;
  }
  if (value.is_null()) {
    return;
  }

  throw std::runtime_error(
      "generation metadata: eos_token_id must be an integer or array in " +
      std::string(source));
}

void collect_eos(
    const std::optional<json>& root,
    std::vector<std::size_t>& out,
    std::string_view source) {
  if (!root.has_value() || !root->contains("eos_token_id")) {
    return;
  }
  append_eos_value(root->at("eos_token_id"), out, source);
}

}  // namespace

QwenGenerationMetadata load_qwen_generation_metadata(
    const QpackReader& qpack) {
  QwenGenerationMetadata metadata;

  const auto generation = read_declared_json(
      qpack,
      "generation_config.json",
      false);
  const auto config = read_declared_json(
      qpack,
      "config.json",
      true);

  collect_eos(
      generation,
      metadata.eos_token_ids,
      "generation_config.json");
  collect_eos(
      config,
      metadata.eos_token_ids,
      "config.json");

  std::sort(
      metadata.eos_token_ids.begin(),
      metadata.eos_token_ids.end());
  metadata.eos_token_ids.erase(
      std::unique(
          metadata.eos_token_ids.begin(),
          metadata.eos_token_ids.end()),
      metadata.eos_token_ids.end());

  return metadata;
}

}  // namespace orbi::streammoe
