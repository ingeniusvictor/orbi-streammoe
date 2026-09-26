#include "orbi/streammoe/conversion/dense_safetensors_conversion.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/container/safetensors.hpp"
#include "orbi/streammoe/conversion/bf16_slice.hpp"
#include "orbi/streammoe/conversion/single_expert_pilot.hpp"

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

struct OutputTensor {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::vector<std::byte> bytes;
};

std::uint64_t parse_hex_u64(const std::string& value) {
  if (value.size() != 16U) {
    throw std::runtime_error(
        "dense conversion: source hash must be 16 hex digits");
  }
  std::uint64_t out = 0U;
  for (const char ch : value) {
    std::uint64_t digit{};
    if (ch >= '0' && ch <= '9') digit = static_cast<std::uint64_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f') digit = 10U + static_cast<std::uint64_t>(ch - 'a');
    else if (ch >= 'A' && ch <= 'F') digit = 10U + static_cast<std::uint64_t>(ch - 'A');
    else throw std::runtime_error("dense conversion: invalid source hash");
    out = (out << 4U) | digit;
  }
  return out;
}

std::uint64_t checked_elements(const std::vector<std::size_t>& shape) {
  if (shape.empty()) {
    throw std::runtime_error("dense conversion: tensor rank must be non-zero");
  }
  std::uint64_t count = 1U;
  for (const auto dim : shape) {
    if (dim == 0U ||
        count > std::numeric_limits<std::uint64_t>::max() /
                    static_cast<std::uint64_t>(dim)) {
      throw std::runtime_error(
          "dense conversion: invalid/overflowing source shape");
    }
    count *= static_cast<std::uint64_t>(dim);
  }
  return count;
}

std::vector<std::byte> read_source(
    const std::filesystem::path& path,
    std::uint64_t expected,
    std::uint64_t expected_hash) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || size != expected ||
      expected > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "dense conversion: source file size mismatch");
  }

  std::vector<std::byte> bytes(static_cast<std::size_t>(expected));
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("dense conversion: unable to open source file");
  }
  input.read(
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("dense conversion: short source read");
  }
  if (fnv1a64(bytes) != expected_hash) {
    throw std::runtime_error("dense conversion: source FNV mismatch");
  }
  return bytes;
}

std::vector<std::byte> f32_bytes(std::span<const float> values) {
  std::vector<std::byte> out(values.size() * sizeof(float));
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      out[i * 4U + lane] = static_cast<std::byte>(
          (bits >> (lane * 8U)) & 0xFFU);
    }
  }
  return out;
}

std::vector<std::byte> u32_bytes(std::span<const std::uint32_t> values) {
  std::vector<std::byte> out(values.size() * sizeof(std::uint32_t));
  for (std::size_t i = 0U; i < values.size(); ++i) {
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      out[i * 4U + lane] = static_cast<std::byte>(
          (values[i] >> (lane * 8U)) & 0xFFU);
    }
  }
  return out;
}

const QpackConversionPlanEntry& plan_entry(
    const QpackConversionPlan& plan,
    const std::string& source_tensor) {
  const auto it = std::lower_bound(
      plan.entries.begin(),
      plan.entries.end(),
      source_tensor,
      [](const auto& entry, const std::string& name) {
        return entry.source_tensor < name;
      });
  if (it == plan.entries.end() || it->source_tensor != source_tensor) {
    throw std::runtime_error(
        "dense conversion: source tensor absent from conversion plan: " +
        source_tensor);
  }
  if (it->tensor_class == QpackConversionClass::routed_expert ||
      it->tensor_class == QpackConversionClass::auxiliary_mtp ||
      it->target_file != "model.safetensors") {
    throw std::runtime_error(
        "dense conversion: manifest selected a non-dense plan entry: " +
        source_tensor);
  }
  return *it;
}

void append_u64_le(std::ofstream& output, std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (std::size_t i = 0U; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (8U * i)) & 0xFFU);
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_safetensors(
    const std::filesystem::path& path,
    std::vector<OutputTensor> tensors) {
  if (tensors.empty()) {
    throw std::runtime_error("dense conversion: no output tensors");
  }
  std::sort(
      tensors.begin(), tensors.end(),
      [](const auto& a, const auto& b) { return a.name < b.name; });

  json header = json::object();
  std::uint64_t offset = 0U;
  std::string previous;
  for (const auto& tensor : tensors) {
    if (tensor.name.empty() || (!previous.empty() && tensor.name <= previous)) {
      throw std::runtime_error(
          "dense conversion: output tensor names must be unique/sorted");
    }
    previous = tensor.name;
    const auto begin = offset;
    if (tensor.bytes.size() >
        std::numeric_limits<std::uint64_t>::max() - offset) {
      throw std::runtime_error("dense conversion: payload offset overflow");
    }
    offset += static_cast<std::uint64_t>(tensor.bytes.size());
    header[tensor.name] = {
        {"dtype", tensor.dtype},
        {"shape", tensor.shape},
        {"data_offsets", {begin, offset}},
    };
  }

  std::string header_text = header.dump();
  while ((header_text.size() % 8U) != 0U) header_text.push_back(' ');

  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const auto temp = std::filesystem::path(path.string() + ".tmp");
  {
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error(
          "dense conversion: unable to create safetensors temp file");
    }
    append_u64_le(output, static_cast<std::uint64_t>(header_text.size()));
    output.write(
        header_text.data(),
        static_cast<std::streamsize>(header_text.size()));
    for (const auto& tensor : tensors) {
      output.write(
          reinterpret_cast<const char*>(tensor.bytes.data()),
          static_cast<std::streamsize>(tensor.bytes.size()));
    }
    output.flush();
    if (!output) {
      throw std::runtime_error(
          "dense conversion: unable to flush safetensors payload");
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
  ec.clear();
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    std::filesystem::remove(temp);
    throw std::runtime_error(
        "dense conversion: unable to commit safetensors output");
  }

  // Validate the exact file using the production reader.
  const SafetensorsReader reader(path);
  const auto names = reader.tensor_names();
  if (names.size() != tensors.size()) {
    throw std::runtime_error(
        "dense conversion: production reader tensor count mismatch");
  }
  for (std::size_t i = 0U; i < names.size(); ++i) {
    if (names[i] != tensors[i].name) {
      throw std::runtime_error(
          "dense conversion: production reader tensor inventory mismatch");
    }
  }
}

}  // namespace

DenseConversionManifest inspect_dense_conversion_manifest(
    const std::filesystem::path& manifest_path) {
  std::ifstream input(manifest_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "dense conversion: unable to open manifest");
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("dense conversion: invalid JSON: ") + e.what());
  }
  if (root.value("schema_version", 0U) != 1U) {
    throw std::runtime_error(
        "dense conversion: unsupported manifest schema");
  }

  DenseConversionManifest out;
  try {
    out.model = root.at("model").get<std::string>();
    out.snapshot = root.at("snapshot").get<std::string>();
    const auto& tensors = root.at("tensors");
    if (!tensors.is_array()) {
      throw std::runtime_error(
          "dense conversion: tensors must be an array");
    }
    for (const auto& value : tensors) {
      DenseSourceTensorManifest item;
      item.source_tensor = value.at("source_tensor").get<std::string>();
      item.source_shape =
          value.at("source_shape").get<std::vector<std::size_t>>();
      item.source_byte_size =
          value.at("source_byte_size").get<std::uint64_t>();
      item.source_file = value.at("source_file").get<std::string>();
      item.source_fnv1a64 =
          parse_hex_u64(value.at("source_fnv1a64").get<std::string>());
      out.tensors.push_back(std::move(item));
    }
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("dense conversion: malformed manifest: ") + e.what());
  }

  if (out.model.empty() || out.snapshot.empty() || out.tensors.empty()) {
    throw std::runtime_error(
        "dense conversion: model/snapshot/tensors must be non-empty");
  }

  std::set<std::string> names;
  for (const auto& tensor : out.tensors) {
    if (tensor.source_tensor.empty() || tensor.source_file.empty() ||
        tensor.source_shape.empty() || tensor.source_byte_size == 0U ||
        tensor.source_fnv1a64 == 0U) {
      throw std::runtime_error(
          "dense conversion: malformed tensor manifest entry");
    }
    const auto elements = checked_elements(tensor.source_shape);
    if (elements > std::numeric_limits<std::uint64_t>::max() / 2U ||
        elements * 2U != tensor.source_byte_size) {
      throw std::runtime_error(
          "dense conversion: BF16 byte size disagrees with shape");
    }
    if (!names.insert(tensor.source_tensor).second) {
      throw std::runtime_error(
          "dense conversion: duplicate source tensor");
    }
  }
  return out;
}

DenseSafetensorsConversionResult convert_dense_plan_slice(
    const QpackConversionPlan& plan,
    const DenseConversionManifest& manifest,
    const std::filesystem::path& manifest_dir,
    const std::filesystem::path& output_safetensors,
    QpackExpertQuantizationSpec quantization) {
  if (quantization.bits != 4U || quantization.group_size == 0U) {
    throw std::invalid_argument(
        "dense conversion: OSM-40A requires 4-bit affine quantization");
  }

  DenseSafetensorsConversionResult result;
  std::vector<OutputTensor> output;

  for (const auto& source : manifest.tensors) {
    const auto& entry = plan_entry(plan, source.source_tensor);
    const auto raw = read_source(
        manifest_dir / source.source_file,
        source.source_byte_size,
        source.source_fnv1a64);
    const auto values = decode_bf16_le(raw);
    result.source_bytes += source.source_byte_size;
    ++result.source_tensor_count;

    if (entry.action == QpackConversionAction::copy_bf16_to_f32) {
      auto bytes = f32_bytes(values);
      result.output_payload_bytes += bytes.size();
      output.push_back({
          entry.target_path,
          "F32",
          source.source_shape,
          std::move(bytes),
      });
      continue;
    }

    if (entry.action != QpackConversionAction::affine_quantize) {
      throw std::runtime_error(
          "dense conversion: unsupported dense conversion action: " +
          std::string(to_string(entry.action)));
    }
    if (source.source_shape.size() != 2U) {
      throw std::runtime_error(
          "dense conversion: affine source must be a matrix");
    }
    const auto rows = source.source_shape[0];
    const auto cols = source.source_shape[1];
    const auto affine = quantize_affine_q4_rows(
        values,
        rows,
        cols,
        quantization.group_size);

    auto packed = u32_bytes(affine.packed);
    auto scales = f32_bytes(affine.scales);
    auto biases = f32_bytes(affine.biases);
    result.output_payload_bytes +=
        packed.size() + scales.size() + biases.size();

    output.push_back({
        entry.target_path + ".weight",
        "U32",
        {rows, affine.packed_cols},
        std::move(packed),
    });
    output.push_back({
        entry.target_path + ".scales",
        "F32",
        {rows, affine.groups_per_row},
        std::move(scales),
    });
    output.push_back({
        entry.target_path + ".biases",
        "F32",
        {rows, affine.groups_per_row},
        std::move(biases),
    });
  }

  write_safetensors(output_safetensors, output);

  const SafetensorsReader reader(output_safetensors);
  result.output_tensor_names = reader.tensor_names();
  result.output_tensor_count = result.output_tensor_names.size();
  return result;
}

}  // namespace orbi::streammoe
