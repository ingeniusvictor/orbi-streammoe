#include "orbi/streammoe/container/qpack.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

json read_json_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("qpack: unable to open JSON file: " + path.string());
  }

  json value;
  try {
    input >> value;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "qpack: invalid JSON in " + path.string() + ": " + e.what());
  }
  return value;
}

std::optional<std::uint32_t> optional_u32(const json& object, const char* key) {
  if (!object.contains(key) || object.at(key).is_null()) {
    return std::nullopt;
  }
  return object.at(key).get<std::uint32_t>();
}

QpackManifest parse_manifest(const std::filesystem::path& path) {
  const auto j = read_json_file(path);

  QpackManifest out;
  try {
    out.magic = j.at("magic").get<std::string>();
    out.version = j.at("version").get<std::uint32_t>();
    out.model_name = j.at("modelName").get<std::string>();
    out.source_checkpoint = j.at("sourceCheckpoint").get<std::string>();
    out.quant_bits = optional_u32(j, "quantBits");
    out.quant_group_size = optional_u32(j, "quantGroupSize");

    const auto& files = j.at("files");
    if (!files.is_object()) {
      throw std::runtime_error("qpack: manifest.files must be an object");
    }
    for (auto it = files.begin(); it != files.end(); ++it) {
      out.files.emplace(it.key(), it.value().get<std::uint64_t>());
    }
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "qpack: malformed manifest " + path.string() + ": " + e.what());
  }
  return out;
}

QpackLayout parse_layout(const std::filesystem::path& path) {
  const auto j = read_json_file(path);

  QpackLayout out;
  try {
    out.expert_count = j.at("expertCount").get<std::uint32_t>();
    out.layer_count = j.at("layerCount").get<std::uint32_t>();
    out.expert_stride = j.at("expertStride").get<std::uint64_t>();
    out.linear_layers = j.at("linearLayers").get<std::vector<bool>>();

    const auto& sections = j.at("sections");
    if (!sections.is_array()) {
      throw std::runtime_error("qpack: layout.sections must be an array");
    }

    out.sections.reserve(sections.size());
    for (const auto& item : sections) {
      QpackSection section;
      section.name = item.at("name").get<std::string>();
      section.dtype = item.at("dtype").get<std::string>();
      section.shape = item.at("shape").get<std::vector<std::size_t>>();
      section.offset = item.at("offset").get<std::uint64_t>();
      section.size = item.at("size").get<std::uint64_t>();
      out.sections.push_back(std::move(section));
    }
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "qpack: malformed layout " + path.string() + ": " + e.what());
  }
  return out;
}

std::uint64_t checked_layer_bytes(const QpackLayout& layout) {
  if (layout.expert_count == 0 || layout.expert_stride == 0) {
    throw std::runtime_error("qpack: expertCount and expertStride must be non-zero");
  }

  const auto count = static_cast<std::uint64_t>(layout.expert_count);
  if (layout.expert_stride > std::numeric_limits<std::uint64_t>::max() / count) {
    throw std::runtime_error("qpack: layer byte size overflows uint64");
  }
  return count * layout.expert_stride;
}

void validate_declared_file(
    const std::filesystem::path& root,
    const QpackManifest& manifest,
    std::string_view relative_path,
    bool required) {
  const auto it = manifest.files.find(std::string(relative_path));
  if (it == manifest.files.end()) {
    if (required) {
      throw std::runtime_error(
          "qpack: manifest does not declare required file: " +
          std::string(relative_path));
    }
    return;
  }

  const auto actual = std::filesystem::file_size(root / std::filesystem::path(relative_path));
  if (actual != it->second) {
    throw std::runtime_error(
        "qpack: file size disagrees with manifest: " +
        std::string(relative_path));
  }
}

void validate_expert_quantization(
    const QpackManifest& manifest,
    const QpackLayout& layout) {
  if (!manifest.quant_bits || !manifest.quant_group_size) {
    return;
  }

  const auto bits = *manifest.quant_bits;
  const auto group = *manifest.quant_group_size;
  if (bits == 0 || group == 0 || (32U % bits) != 0U) {
    throw std::runtime_error("qpack: invalid expert quantization metadata");
  }

  for (const std::string prefix : {"gate_proj", "up_proj", "down_proj"}) {
    const auto* weight = layout.find_section(prefix + ".weight");
    const auto* scales = layout.find_section(prefix + ".scales");
    if (weight == nullptr || scales == nullptr) {
      continue;
    }
    if (weight->shape.empty() || scales->shape.empty()) {
      throw std::runtime_error(
          "qpack: quantized expert section has empty shape: " + prefix);
    }

    const auto packed_words = static_cast<std::uint64_t>(weight->shape.back());
    const auto scale_groups = static_cast<std::uint64_t>(scales->shape.back());
    const auto logical_from_weight = packed_words * (32U / bits);
    const auto logical_from_scales = scale_groups * group;

    if (logical_from_weight == 0 || logical_from_weight != logical_from_scales) {
      throw std::runtime_error(
          "qpack: expert quantization disagrees with packed shapes: " + prefix);
    }
  }
}

}  // namespace

const QpackSection* QpackLayout::find_section(std::string_view name) const noexcept {
  const auto it = std::find_if(
      sections.begin(),
      sections.end(),
      [name](const QpackSection& section) { return section.name == name; });
  return it == sections.end() ? nullptr : &*it;
}

QpackReader::QpackReader(std::filesystem::path container_dir)
    : container_dir_(std::move(container_dir)),
      manifest_(parse_manifest(container_dir_ / "manifest.json")),
      layout_(parse_layout(container_dir_ / "packed_experts" / "layout.json")) {
  validate_container();
}

std::filesystem::path QpackReader::layer_path(std::uint32_t layer) const {
  std::ostringstream name;
  name << "layer_" << std::setfill('0') << std::setw(2) << layer << ".bin";
  return container_dir_ / "packed_experts" / name.str();
}

void QpackReader::validate_container() const {
  if (manifest_.magic != kMagic) {
    throw std::runtime_error(
        "qpack: unsupported magic: " + manifest_.magic);
  }
  if (manifest_.version != kManifestVersion) {
    throw std::runtime_error(
        "qpack: unsupported manifest version: " +
        std::to_string(manifest_.version));
  }
  if (layout_.layer_count == 0) {
    throw std::runtime_error("qpack: layerCount must be non-zero");
  }
  if (layout_.sections.empty()) {
    throw std::runtime_error("qpack: sections must not be empty");
  }
  if (layout_.linear_layers.size() != layout_.layer_count) {
    throw std::runtime_error(
        "qpack: linearLayers length must equal layerCount");
  }

  for (const auto& section : layout_.sections) {
    if (section.name.empty() || section.size == 0) {
      throw std::runtime_error("qpack: section name/size is invalid");
    }
    if (section.offset > layout_.expert_stride ||
        section.size > layout_.expert_stride - section.offset) {
      throw std::runtime_error(
          "qpack: section exceeds expertStride: " + section.name);
    }
  }

  if (manifest_.quant_bits.has_value() != manifest_.quant_group_size.has_value()) {
    throw std::runtime_error(
        "qpack: quantBits and quantGroupSize must both be present or both be null");
  }
  validate_expert_quantization(manifest_, layout_);

  const auto expected_layer_bytes = checked_layer_bytes(layout_);

  validate_declared_file(
      container_dir_,
      manifest_,
      "packed_experts/layout.json",
      true);

  for (std::uint32_t layer = 0; layer < layout_.layer_count; ++layer) {
    const auto path = layer_path(layer);
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error(
          "qpack: missing expert layer file: " + path.string());
    }

    const auto actual = std::filesystem::file_size(path);
    if (actual != expected_layer_bytes) {
      throw std::runtime_error(
          "qpack: expert layer has unexpected byte size: " + path.string());
    }

    const auto relative =
        std::filesystem::relative(path, container_dir_).generic_string();
    validate_declared_file(container_dir_, manifest_, relative, true);
  }
}

void QpackReader::read_expert(
    std::uint32_t layer,
    std::uint32_t expert,
    std::span<std::byte> destination) const {
  if (layer >= layout_.layer_count) {
    throw std::out_of_range("qpack: layer index out of range");
  }
  if (expert >= layout_.expert_count) {
    throw std::out_of_range("qpack: expert index out of range");
  }
  if (destination.size() < layout_.expert_stride) {
    throw std::invalid_argument("qpack: destination is smaller than expertStride");
  }

  std::ifstream input(layer_path(layer), std::ios::binary);
  if (!input) {
    throw std::runtime_error("qpack: unable to open expert layer file");
  }

  const auto offset =
      static_cast<std::uint64_t>(expert) * layout_.expert_stride;
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error("qpack: expert offset exceeds streamoff range");
  }

  input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!input) {
    throw std::runtime_error("qpack: seek to expert failed");
  }

  input.read(
      reinterpret_cast<char*>(destination.data()),
      static_cast<std::streamsize>(layout_.expert_stride));
  if (input.gcount() != static_cast<std::streamsize>(layout_.expert_stride)) {
    throw std::runtime_error("qpack: short expert read");
  }
}

std::vector<std::byte> QpackReader::read_expert(
    std::uint32_t layer,
    std::uint32_t expert) const {
  std::vector<std::byte> bytes(
      static_cast<std::size_t>(layout_.expert_stride));
  read_expert(layer, expert, bytes);
  return bytes;
}

}  // namespace orbi::streammoe
