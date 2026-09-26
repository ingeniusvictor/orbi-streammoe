#include "orbi/streammoe/conversion/single_expert_pilot.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/conversion/bf16_slice.hpp"

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::uint64_t parse_hex_u64(const std::string& value, const char* field) {
  if (value.size() != 16U) {
    throw std::runtime_error(
        std::string("single expert pilot: ") + field +
        " must be 16 hex digits");
  }
  std::uint64_t out = 0U;
  for (const char ch : value) {
    std::uint64_t digit{};
    if (ch >= '0' && ch <= '9') digit = static_cast<std::uint64_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f') digit = 10U + static_cast<std::uint64_t>(ch - 'a');
    else if (ch >= 'A' && ch <= 'F') digit = 10U + static_cast<std::uint64_t>(ch - 'A');
    else throw std::runtime_error(
        std::string("single expert pilot: invalid hex in ") + field);
    out = (out << 4U) | digit;
  }
  return out;
}

std::vector<std::byte> read_exact(
    const std::filesystem::path& path,
    std::uint64_t expected) {
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec || actual != expected) {
    throw std::runtime_error(
        "single expert pilot: source file size mismatch: " + path.string());
  }
  if (expected > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("single expert pilot: source file too large");
  }
  std::vector<std::byte> out(static_cast<std::size_t>(expected));
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "single expert pilot: unable to open source file: " + path.string());
  }
  input.read(
      reinterpret_cast<char*>(out.data()),
      static_cast<std::streamsize>(out.size()));
  if (input.gcount() != static_cast<std::streamsize>(out.size())) {
    throw std::runtime_error("single expert pilot: short source read");
  }
  return out;
}

const ExpertPilotProjectionManifest& projection_manifest(
    const SingleExpertPilotManifest& manifest,
    const std::string& name) {
  const auto it = std::find_if(
      manifest.projections.begin(),
      manifest.projections.end(),
      [&](const auto& item) { return item.name == name; });
  if (it == manifest.projections.end()) {
    throw std::runtime_error(
        "single expert pilot: missing projection manifest: " + name);
  }
  return *it;
}

const QpackSection& qpack_section(
    const QpackExpertGeometry& geometry,
    const std::string& name) {
  const auto it = std::find_if(
      geometry.sections.begin(),
      geometry.sections.end(),
      [&](const auto& item) { return item.name == name; });
  if (it == geometry.sections.end()) {
    throw std::runtime_error(
        "single expert pilot: missing QPACK section: " + name);
  }
  return *it;
}

void write_u32_le(
    std::span<std::byte> destination,
    std::span<const std::uint32_t> values) {
  if (destination.size() != values.size() * sizeof(std::uint32_t)) {
    throw std::runtime_error("single expert pilot: U32 section size mismatch");
  }
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto value = values[i];
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      destination[i * 4U + lane] = static_cast<std::byte>(
          (value >> (lane * 8U)) & 0xFFU);
    }
  }
}

void write_f32_le_bytes(
    std::span<std::byte> destination,
    std::span<const float> values) {
  if (destination.size() != values.size() * sizeof(float)) {
    throw std::runtime_error("single expert pilot: F32 section size mismatch");
  }
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      destination[i * 4U + lane] = static_cast<std::byte>(
          (bits >> (lane * 8U)) & 0xFFU);
    }
  }
}

void place_projection(
    std::vector<std::byte>& blob,
    const QpackExpertGeometry& geometry,
    const std::string& prefix,
    const AffineQ4Projection& projection) {
  const auto& weight = qpack_section(geometry, prefix + ".weight");
  const auto& scales = qpack_section(geometry, prefix + ".scales");
  const auto& biases = qpack_section(geometry, prefix + ".biases");

  write_u32_le(
      std::span<std::byte>(
          blob.data() + static_cast<std::size_t>(weight.offset),
          static_cast<std::size_t>(weight.size)),
      projection.packed);
  write_f32_le_bytes(
      std::span<std::byte>(
          blob.data() + static_cast<std::size_t>(scales.offset),
          static_cast<std::size_t>(scales.size)),
      projection.scales);
  write_f32_le_bytes(
      std::span<std::byte>(
          blob.data() + static_cast<std::size_t>(biases.offset),
          static_cast<std::size_t>(biases.size)),
      projection.biases);
}

AffineQ4Projection convert_projection(
    const SingleExpertPilotManifest& manifest,
    const std::filesystem::path& manifest_dir,
    const std::string& name,
    std::size_t rows,
    std::size_t cols,
    std::size_t group_size,
    std::uint64_t& source_bytes) {
  const auto& meta = projection_manifest(manifest, name);
  if (meta.shape != std::vector<std::size_t>({rows, cols})) {
    throw std::runtime_error(
        "single expert pilot: source shape mismatch for " + name);
  }
  const auto expected =
      static_cast<std::uint64_t>(rows) *
      static_cast<std::uint64_t>(cols) * 2U;
  if (meta.source_byte_size != expected) {
    throw std::runtime_error(
        "single expert pilot: BF16 byte count mismatch for " + name);
  }

  const auto bytes = read_exact(
      manifest_dir / meta.source_file,
      meta.source_byte_size);
  if (fnv1a64(bytes) != meta.source_fnv1a64) {
    throw std::runtime_error(
        "single expert pilot: source FNV mismatch for " + name);
  }
  source_bytes += meta.source_byte_size;

  const auto values = decode_bf16_le(bytes);
  return quantize_affine_q4_rows(values, rows, cols, group_size);
}

}  // namespace

SingleExpertPilotManifest inspect_single_expert_pilot_manifest(
    const std::filesystem::path& manifest_path) {
  std::ifstream input(manifest_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "single expert pilot: unable to open manifest");
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("single expert pilot: invalid JSON: ") + e.what());
  }
  if (root.value("schema_version", 0U) != 1U) {
    throw std::runtime_error(
        "single expert pilot: unsupported schema_version");
  }

  SingleExpertPilotManifest out;
  try {
    out.model = root.at("model").get<std::string>();
    out.snapshot = root.at("snapshot").get<std::string>();
    out.layer_index = root.at("layer_index").get<std::size_t>();
    out.expert_index = root.at("expert_index").get<std::size_t>();
    out.total_fetched_bytes =
        root.at("total_fetched_bytes").get<std::uint64_t>();

    const auto& projections = root.at("projections");
    if (!projections.is_object()) {
      throw std::runtime_error(
          "single expert pilot: projections must be an object");
    }
    for (auto it = projections.begin(); it != projections.end(); ++it) {
      ExpertPilotProjectionManifest item;
      item.name = it.key();
      item.tensor_name = it.value().at("tensor_name").get<std::string>();
      item.shard = it.value().at("shard").get<std::string>();
      item.shape = it.value().at("shape").get<std::vector<std::size_t>>();
      item.source_byte_size =
          it.value().at("source_byte_size").get<std::uint64_t>();
      item.source_file = it.value().at("source_file").get<std::string>();
      item.source_fnv1a64 = parse_hex_u64(
          it.value().at("source_fnv1a64").get<std::string>(),
          "source_fnv1a64");
      out.projections.push_back(std::move(item));
    }
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("single expert pilot: malformed manifest: ") + e.what());
  }

  if (out.model.empty() || out.snapshot.empty() ||
      out.projections.size() != 3U) {
    throw std::runtime_error(
        "single expert pilot: expected model/snapshot and three projections");
  }

  std::uint64_t total = 0U;
  for (const auto& item : out.projections) {
    if (item.name != "gate_proj" &&
        item.name != "up_proj" &&
        item.name != "down_proj") {
      throw std::runtime_error(
          "single expert pilot: unsupported projection: " + item.name);
    }
    if (item.tensor_name.empty() || item.shard.empty() ||
        item.shape.size() != 2U || item.source_file.empty() ||
        item.source_byte_size == 0U) {
      throw std::runtime_error(
          "single expert pilot: malformed projection manifest");
    }
    if (total > std::numeric_limits<std::uint64_t>::max() -
                    item.source_byte_size) {
      throw std::runtime_error(
          "single expert pilot: fetched byte count overflow");
    }
    total += item.source_byte_size;
  }
  if (total != out.total_fetched_bytes) {
    throw std::runtime_error(
        "single expert pilot: total_fetched_bytes mismatch");
  }
  return out;
}

AffineQ4Projection quantize_affine_q4_rows(
    const std::vector<float>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t group_size) {
  if (rows == 0U || cols == 0U || group_size == 0U ||
      (cols % 8U) != 0U || (cols % group_size) != 0U ||
      values.size() != rows * cols) {
    throw std::invalid_argument(
        "single expert pilot: invalid Q4 matrix geometry");
  }

  AffineQ4Projection out;
  out.rows = rows;
  out.cols = cols;
  out.packed_cols = cols / 8U;
  out.groups_per_row = cols / group_size;
  out.packed.assign(rows * out.packed_cols, 0U);
  out.scales.resize(rows * out.groups_per_row);
  out.biases.resize(rows * out.groups_per_row);

  double error_sum = 0.0;
  std::size_t error_count = 0U;

  for (std::size_t row = 0U; row < rows; ++row) {
    for (std::size_t group = 0U; group < out.groups_per_row; ++group) {
      const auto begin = row * cols + group * group_size;
      const auto end = begin + group_size;

      float minimum = std::numeric_limits<float>::infinity();
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t i = begin; i < end; ++i) {
        const float value = values[i];
        if (!std::isfinite(value)) {
          throw std::runtime_error(
              "single expert pilot: non-finite source weight");
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }

      const float range = maximum - minimum;
      const float scale =
          range > 0.0F ? range / 15.0F : 1.0F;
      const float bias = minimum;
      const auto group_index = row * out.groups_per_row + group;
      out.scales[group_index] = scale;
      out.biases[group_index] = bias;

      for (std::size_t i = begin; i < end; ++i) {
        const auto col = i - row * cols;
        const float normalized = (values[i] - bias) / scale;
        const auto rounded = static_cast<int>(std::lround(normalized));
        const auto q = static_cast<std::uint32_t>(
            std::clamp(rounded, 0, 15));

        const auto word_index =
            row * out.packed_cols + col / 8U;
        const auto lane = static_cast<std::uint32_t>(col % 8U);
        out.packed[word_index] |= q << (lane * 4U);

        const float reconstructed =
            scale * static_cast<float>(q) + bias;
        const float error = std::fabs(reconstructed - values[i]);
        const float allowed = range > 0.0F
            ? scale * 0.5001F + 1e-6F
            : 1e-6F;
        if (error > allowed) {
          throw std::runtime_error(
              "single expert pilot: affine quantization error exceeded group bound");
        }
        out.max_abs_error = std::max(out.max_abs_error, error);
        error_sum += static_cast<double>(error);
        ++error_count;
      }
    }
  }

  out.mean_abs_error =
      error_count == 0U ? 0.0 : error_sum / static_cast<double>(error_count);
  return out;
}

SingleExpertConversionResult convert_qwen_single_expert_pilot(
    const SingleExpertPilotManifest& manifest,
    const std::filesystem::path& manifest_dir,
    const QpackExpertGeometry& geometry,
    const std::filesystem::path& output_blob) {
  if (geometry.quantization.bits != 4U) {
    throw std::runtime_error(
        "single expert pilot: OSM-39C requires 4-bit target geometry");
  }
  if (manifest.layer_index >= geometry.layer_count ||
      manifest.expert_index >= geometry.expert_count) {
    throw std::runtime_error(
        "single expert pilot: layer/expert index out of target range");
  }

  SingleExpertConversionResult out;
  out.gate = convert_projection(
      manifest,
      manifest_dir,
      "gate_proj",
      geometry.moe_intermediate_size,
      geometry.hidden_size,
      geometry.quantization.group_size,
      out.source_bytes);
  out.up = convert_projection(
      manifest,
      manifest_dir,
      "up_proj",
      geometry.moe_intermediate_size,
      geometry.hidden_size,
      geometry.quantization.group_size,
      out.source_bytes);
  out.down = convert_projection(
      manifest,
      manifest_dir,
      "down_proj",
      geometry.hidden_size,
      geometry.moe_intermediate_size,
      geometry.quantization.group_size,
      out.source_bytes);

  if (out.source_bytes != manifest.total_fetched_bytes) {
    throw std::runtime_error(
        "single expert pilot: converted source byte total mismatch");
  }

  if (geometry.expert_stride >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "single expert pilot: expert stride exceeds address space");
  }
  out.qpack_blob.assign(
      static_cast<std::size_t>(geometry.expert_stride),
      std::byte{0});

  place_projection(out.qpack_blob, geometry, "gate_proj", out.gate);
  place_projection(out.qpack_blob, geometry, "up_proj", out.up);
  place_projection(out.qpack_blob, geometry, "down_proj", out.down);

  out.max_abs_error = std::max({
      out.gate.max_abs_error,
      out.up.max_abs_error,
      out.down.max_abs_error});
  const auto gate_values =
      static_cast<double>(out.gate.rows) * static_cast<double>(out.gate.cols);
  const auto up_values =
      static_cast<double>(out.up.rows) * static_cast<double>(out.up.cols);
  const auto down_values =
      static_cast<double>(out.down.rows) * static_cast<double>(out.down.cols);
  const auto total_values = gate_values + up_values + down_values;
  out.mean_abs_error =
      (out.gate.mean_abs_error * gate_values +
       out.up.mean_abs_error * up_values +
       out.down.mean_abs_error * down_values) /
      total_values;

  std::ofstream output(output_blob, std::ios::binary);
  if (!output) {
    throw std::runtime_error(
        "single expert pilot: unable to open QPACK expert output");
  }
  output.write(
      reinterpret_cast<const char*>(out.qpack_blob.data()),
      static_cast<std::streamsize>(out.qpack_blob.size()));
  if (!output) {
    throw std::runtime_error(
        "single expert pilot: unable to write QPACK expert output");
  }

  return out;
}

}  // namespace orbi::streammoe
