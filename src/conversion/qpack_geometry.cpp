#include "orbi/streammoe/conversion/qpack_geometry.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::uint64_t checked_mul(
    std::uint64_t a,
    std::uint64_t b,
    const char* label) {
  if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
    throw std::runtime_error(
        std::string("qpack geometry: overflow computing ") + label);
  }
  return a * b;
}

std::uint64_t checked_add(
    std::uint64_t a,
    std::uint64_t b,
    const char* label) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    throw std::runtime_error(
        std::string("qpack geometry: overflow computing ") + label);
  }
  return a + b;
}

std::size_t required_size(const json& root, const char* key) {
  try {
    const auto value = root.at(key).get<std::size_t>();
    if (value == 0U) {
      throw std::runtime_error(
          std::string("qpack geometry: zero config field: ") + key);
    }
    return value;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("qpack geometry: malformed config field ") +
        key + ": " + e.what());
  }
}

json read_config(const std::filesystem::path& model_dir) {
  std::ifstream input(model_dir / "config.json", std::ios::binary);
  if (!input) {
    throw std::runtime_error("qpack geometry: unable to open config.json");
  }
  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("qpack geometry: invalid config.json: ") + e.what());
  }
  if (root.value("model_type", std::string{}) != "qwen3_next") {
    throw std::runtime_error(
        "qpack geometry: model_type must be qwen3_next");
  }
  return root;
}

struct ProjectionShape {
  std::string name;
  std::size_t rows{};
  std::size_t cols{};
};

void append_projection(
    QpackExpertGeometry& geometry,
    std::uint64_t& offset,
    const ProjectionShape& projection) {
  const auto bits = geometry.quantization.bits;
  const auto group = geometry.quantization.group_size;
  const auto values_per_word = 32U / bits;

  if ((projection.cols % values_per_word) != 0U) {
    throw std::runtime_error(
        "qpack geometry: projection columns do not fit packed U32 words: " +
        projection.name);
  }
  if ((projection.cols % group) != 0U) {
    throw std::runtime_error(
        "qpack geometry: projection columns do not fit quantization groups: " +
        projection.name);
  }

  const auto rows = static_cast<std::uint64_t>(projection.rows);
  const auto packed_words =
      static_cast<std::uint64_t>(projection.cols / values_per_word);
  const auto groups =
      static_cast<std::uint64_t>(projection.cols / group);

  const auto weight_elements =
      checked_mul(rows, packed_words, "weight element count");
  const auto group_elements =
      checked_mul(rows, groups, "group element count");

  const auto weight_bytes =
      checked_mul(weight_elements, sizeof(std::uint32_t), "weight bytes");
  const auto scales_bytes =
      checked_mul(group_elements, sizeof(float), "scale bytes");
  const auto biases_bytes =
      checked_mul(group_elements, sizeof(float), "bias bytes");

  geometry.sections.push_back(QpackSection{
      .name = projection.name + ".weight",
      .dtype = "U32",
      .shape = {projection.rows, static_cast<std::size_t>(packed_words)},
      .offset = offset,
      .size = weight_bytes,
  });
  offset = checked_add(offset, weight_bytes, "section offset");

  geometry.sections.push_back(QpackSection{
      .name = projection.name + ".scales",
      .dtype = "F32",
      .shape = {projection.rows, static_cast<std::size_t>(groups)},
      .offset = offset,
      .size = scales_bytes,
  });
  offset = checked_add(offset, scales_bytes, "section offset");

  geometry.sections.push_back(QpackSection{
      .name = projection.name + ".biases",
      .dtype = "F32",
      .shape = {projection.rows, static_cast<std::size_t>(groups)},
      .offset = offset,
      .size = biases_bytes,
  });
  offset = checked_add(offset, biases_bytes, "section offset");
}

}  // namespace

QpackExpertGeometry derive_qpack_expert_geometry(
    const std::filesystem::path& model_dir,
    QpackExpertQuantizationSpec quantization) {
  if (quantization.bits == 0U ||
      quantization.bits > 32U ||
      (32U % quantization.bits) != 0U) {
    throw std::runtime_error(
        "qpack geometry: quantization bits must divide 32");
  }
  if (quantization.group_size == 0U) {
    throw std::runtime_error(
        "qpack geometry: quantization group size must be non-zero");
  }

  const auto root = read_config(model_dir);

  QpackExpertGeometry geometry;
  geometry.hidden_size = required_size(root, "hidden_size");
  geometry.moe_intermediate_size =
      required_size(root, "moe_intermediate_size");
  geometry.expert_count = required_size(root, "num_experts");
  geometry.layer_count = required_size(root, "num_hidden_layers");
  geometry.quantization = quantization;

  std::uint64_t offset = 0U;
  append_projection(
      geometry,
      offset,
      ProjectionShape{
          "gate_proj",
          geometry.moe_intermediate_size,
          geometry.hidden_size});
  append_projection(
      geometry,
      offset,
      ProjectionShape{
          "up_proj",
          geometry.moe_intermediate_size,
          geometry.hidden_size});
  append_projection(
      geometry,
      offset,
      ProjectionShape{
          "down_proj",
          geometry.hidden_size,
          geometry.moe_intermediate_size});

  geometry.expert_stride = offset;
  geometry.layer_bytes = checked_mul(
      geometry.expert_stride,
      static_cast<std::uint64_t>(geometry.expert_count),
      "per-layer expert bytes");
  geometry.all_layers_bytes = checked_mul(
      geometry.layer_bytes,
      static_cast<std::uint64_t>(geometry.layer_count),
      "all-layer expert bytes");

  validate_qpack_expert_geometry(geometry);
  return geometry;
}

void validate_qpack_expert_geometry(
    const QpackExpertGeometry& geometry) {
  if (geometry.hidden_size == 0U ||
      geometry.moe_intermediate_size == 0U ||
      geometry.expert_count == 0U ||
      geometry.layer_count == 0U ||
      geometry.expert_stride == 0U ||
      geometry.layer_bytes == 0U ||
      geometry.all_layers_bytes == 0U) {
    throw std::runtime_error(
        "qpack geometry: dimensions and byte counts must be non-zero");
  }
  if (geometry.quantization.bits == 0U ||
      (32U % geometry.quantization.bits) != 0U ||
      geometry.quantization.group_size == 0U) {
    throw std::runtime_error(
        "qpack geometry: invalid quantization contract");
  }
  if (geometry.sections.size() != 9U) {
    throw std::runtime_error(
        "qpack geometry: expected exactly nine expert sections");
  }

  std::uint64_t cursor = 0U;
  for (const auto& section : geometry.sections) {
    if (section.name.empty() ||
        section.dtype.empty() ||
        section.shape.size() != 2U ||
        section.size == 0U) {
      throw std::runtime_error(
          "qpack geometry: malformed section");
    }
    if (section.offset != cursor) {
      throw std::runtime_error(
          "qpack geometry: sections must be contiguous and ordered");
    }
    cursor = checked_add(cursor, section.size, "validation cursor");
  }
  if (cursor != geometry.expert_stride) {
    throw std::runtime_error(
        "qpack geometry: section payload does not equal expert stride");
  }
  if (geometry.layer_bytes !=
      checked_mul(
          geometry.expert_stride,
          static_cast<std::uint64_t>(geometry.expert_count),
          "validated layer bytes")) {
    throw std::runtime_error(
        "qpack geometry: per-layer byte count mismatch");
  }
  if (geometry.all_layers_bytes !=
      checked_mul(
          geometry.layer_bytes,
          static_cast<std::uint64_t>(geometry.layer_count),
          "validated all-layer bytes")) {
    throw std::runtime_error(
        "qpack geometry: all-layer byte count mismatch");
  }
}

}  // namespace orbi::streammoe
