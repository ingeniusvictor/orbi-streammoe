#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

constexpr std::string_view kAffineMode = "affine";

bool suffix_match(
    std::string_view path,
    std::string_view key) noexcept {
  return path.ends_with(key) || key.ends_with(path);
}

MlxAffineQuantSpec parse_spec(
    const json& value,
    const std::string& label) {
  if (!value.is_object()) {
    throw std::runtime_error(
        "mlx checkpoint: quantization entry must be an object: " + label);
  }

  MlxAffineQuantSpec spec;
  try {
    spec.group_size = value.at("group_size").get<std::uint32_t>();
    spec.bits = value.at("bits").get<std::uint32_t>();
    spec.mode =
        value.contains("mode")
            ? value.at("mode").get<std::string>()
            : std::string(kAffineMode);
  } catch (const json::exception& e) {
    throw std::runtime_error(
        "mlx checkpoint: malformed quantization entry " + label +
        ": " + e.what());
  }

  if (spec.group_size == 0U || spec.bits == 0U ||
      (32U % spec.bits) != 0U) {
    throw std::runtime_error(
        "mlx checkpoint: invalid quantization geometry: " + label);
  }
  if (spec.mode != kAffineMode) {
    throw std::runtime_error(
        "mlx checkpoint: unsupported quantization mode '" +
        spec.mode + "' for " + label +
        "; only affine is supported");
  }
  return spec;
}

std::filesystem::path validated_config_path(
    const QpackReader& qpack) {
  constexpr std::string_view kConfig = "config.json";

  const auto it = qpack.manifest().files.find(std::string(kConfig));
  if (it == qpack.manifest().files.end()) {
    throw std::runtime_error(
        "mlx checkpoint: qpack manifest does not declare config.json");
  }

  const auto path = qpack.container_dir() / std::string(kConfig);
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error(
        "mlx checkpoint: unable to stat qpack config.json");
  }
  if (actual != it->second) {
    throw std::runtime_error(
        "mlx checkpoint: config.json size disagrees with qpack manifest");
  }
  return path;
}

std::size_t checked_product(
    const std::vector<std::size_t>& shape,
    std::size_t end_exclusive) {
  std::size_t product = 1U;
  for (std::size_t i = 0; i < end_exclusive; ++i) {
    const auto dim = shape[i];
    if (dim == 0U) {
      throw std::runtime_error(
          "mlx checkpoint: zero tensor dimension");
    }
    if (product >
        std::numeric_limits<std::size_t>::max() / dim) {
      throw std::runtime_error(
          "mlx checkpoint: tensor shape product overflows size_t");
    }
    product *= dim;
  }
  return product;
}

void validate_leading_shapes(
    const std::vector<std::size_t>& weight,
    const std::vector<std::size_t>& scales,
    const std::vector<std::size_t>& biases,
    std::string_view path) {
  if (weight.size() != scales.size() ||
      scales.size() != biases.size() ||
      weight.empty()) {
    throw std::runtime_error(
        "mlx checkpoint: weight/scales/biases rank mismatch: " +
        std::string(path));
  }

  for (std::size_t i = 0; i + 1U < weight.size(); ++i) {
    if (weight[i] != scales[i] || scales[i] != biases[i]) {
      throw std::runtime_error(
          "mlx checkpoint: leading quantized shapes disagree: " +
          std::string(path));
    }
  }
}

}  // namespace

MlxQuantizationConfig parse_mlx_quantization_config(
    const std::filesystem::path& config_path) {
  std::ifstream input(config_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "mlx checkpoint: unable to open config.json: " +
        config_path.string());
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("mlx checkpoint: invalid config JSON: ") + e.what());
  }

  MlxQuantizationConfig out;
  if (!root.contains("quantization")) {
    return out;
  }

  const auto& quant = root.at("quantization");
  if (!quant.is_object()) {
    throw std::runtime_error(
        "mlx checkpoint: quantization must be an object");
  }

  const bool has_default_group = quant.contains("group_size");
  const bool has_default_bits = quant.contains("bits");
  if (has_default_group != has_default_bits) {
    throw std::runtime_error(
        "mlx checkpoint: default quantization requires both group_size and bits");
  }
  if (has_default_group) {
    out.default_quant = parse_spec(quant, "quantization");
  }

  for (auto it = quant.begin(); it != quant.end(); ++it) {
    if (it.key() == "group_size" ||
        it.key() == "bits" ||
        it.key() == "mode") {
      continue;
    }
    if (!it.value().is_object()) {
      continue;
    }

    auto spec = parse_spec(it.value(), it.key());
    out.overrides.emplace_back(it.key(), std::move(spec));
  }

  // Prefer the most specific matching suffix deterministically.
  std::sort(
      out.overrides.begin(),
      out.overrides.end(),
      [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) {
          return a.first.size() > b.first.size();
        }
        return a.first < b.first;
      });

  return out;
}

QpackMlxCheckpoint::QpackMlxCheckpoint(
    const QpackReader& qpack)
    : dense_(qpack),
      quantization_(
          parse_mlx_quantization_config(
              validated_config_path(qpack))) {}

bool QpackMlxCheckpoint::is_quantized(
    std::string_view path) const noexcept {
  try {
    return dense_.contains(std::string(path) + ".scales");
  } catch (...) {
    return false;
  }
}

std::optional<MlxAffineQuantSpec>
QpackMlxCheckpoint::quant_spec_for(
    std::string_view path) const {
  if (!is_quantized(path)) {
    return std::nullopt;
  }

  for (const auto& [key, spec] : quantization_.overrides) {
    if (suffix_match(path, key)) {
      return spec;
    }
  }
  return quantization_.default_quant;
}

MlxAffineModule QpackMlxCheckpoint::read_affine_module(
    std::string_view path) const {
  if (!is_quantized(path)) {
    throw std::runtime_error(
        "mlx checkpoint: module is not stored as affine quantized: " +
        std::string(path));
  }

  const auto spec = quant_spec_for(path);
  if (!spec.has_value()) {
    throw std::runtime_error(
        "mlx checkpoint: quantized module has no quantization spec: " +
        std::string(path));
  }
  if (spec->mode != kAffineMode) {
    throw std::runtime_error(
        "mlx checkpoint: non-affine module reached affine reader");
  }
  if (spec->bits != 4U && spec->bits != 8U) {
    throw std::runtime_error(
        "mlx checkpoint: affine module supports 4-bit or 8-bit only: " +
        std::string(path));
  }

  const std::string base(path);
  const auto weight_name = base + ".weight";
  const auto scales_name = base + ".scales";
  const auto biases_name = base + ".biases";

  if (!dense_.contains(weight_name) ||
      !dense_.contains(scales_name) ||
      !dense_.contains(biases_name)) {
    throw std::runtime_error(
        "mlx checkpoint: affine module tensor triplet is incomplete: " +
        base);
  }

  const auto& weight_info = dense_.info(weight_name);
  const auto& scales_info = dense_.info(scales_name);
  const auto& biases_info = dense_.info(biases_name);

  if (weight_info.dtype != "U32") {
    throw std::runtime_error(
        "mlx checkpoint: affine packed weight dtype must be U32: " +
        base);
  }
  if (scales_info.dtype != biases_info.dtype) {
    throw std::runtime_error(
        "mlx checkpoint: scales/biases dtype mismatch: " + base);
  }
  if (scales_info.dtype != "F32" &&
      scales_info.dtype != "F16" &&
      scales_info.dtype != "BF16") {
    throw std::runtime_error(
        "mlx checkpoint: unsupported scale/bias dtype: " + base);
  }

  validate_leading_shapes(
      weight_info.shape,
      scales_info.shape,
      biases_info.shape,
      path);

  if (scales_info.shape.back() != biases_info.shape.back()) {
    throw std::runtime_error(
        "mlx checkpoint: scales/biases final dimension mismatch: " +
        base);
  }

  const auto per_word = 32U / spec->bits;
  const auto packed_cols = weight_info.shape.back();
  if (packed_cols >
      std::numeric_limits<std::size_t>::max() / per_word) {
    throw std::runtime_error(
        "mlx checkpoint: logical input dimension overflows size_t: " +
        base);
  }
  const auto logical_in_dim = packed_cols * per_word;
  if (logical_in_dim == 0U ||
      logical_in_dim % spec->group_size != 0U) {
    throw std::runtime_error(
        "mlx checkpoint: logical input dimension disagrees with group size: " +
        base);
  }

  const auto groups_per_row =
      logical_in_dim / spec->group_size;
  if (scales_info.shape.back() != groups_per_row) {
    throw std::runtime_error(
        "mlx checkpoint: scale geometry disagrees with packed weight/spec: " +
        base);
  }

  const auto rows =
      checked_product(weight_info.shape, weight_info.shape.size() - 1U);
  const auto scale_rows =
      checked_product(scales_info.shape, scales_info.shape.size() - 1U);
  if (rows != scale_rows) {
    throw std::runtime_error(
        "mlx checkpoint: packed and affine-metadata row counts disagree: " +
        base);
  }

  MlxAffineModule module;
  module.path = base;
  module.quant = *spec;
  module.packed_shape = weight_info.shape;
  module.logical_shape = weight_info.shape;
  module.logical_shape.back() = logical_in_dim;
  module.scale_shape = scales_info.shape;
  module.row_count = rows;
  module.logical_in_dim = logical_in_dim;
  module.packed = dense_.read_u32(weight_name);
  module.scales = dense_.read_floats(scales_name);
  module.biases = dense_.read_floats(biases_name);

  if (module.packed.size() != rows * packed_cols ||
      module.scales.size() != rows * groups_per_row ||
      module.biases.size() != rows * groups_per_row) {
    throw std::runtime_error(
        "mlx checkpoint: typed module payload size mismatch: " + base);
  }

  return module;
}

}  // namespace orbi::streammoe
