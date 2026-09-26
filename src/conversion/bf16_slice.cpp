#include "orbi/streammoe/conversion/bf16_slice.hpp"

#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::uint64_t parse_hex_u64(const std::string& value, const char* field) {
  if (value.size() != 16U) {
    throw std::runtime_error(
        std::string("BF16 slice: ") + field + " must be 16 hex digits");
  }
  std::uint64_t out = 0U;
  for (const char ch : value) {
    std::uint64_t digit = 0U;
    if (ch >= '0' && ch <= '9') digit = static_cast<std::uint64_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f') digit = 10U + static_cast<std::uint64_t>(ch - 'a');
    else if (ch >= 'A' && ch <= 'F') digit = 10U + static_cast<std::uint64_t>(ch - 'A');
    else {
      throw std::runtime_error(
          std::string("BF16 slice: invalid hex in ") + field);
    }
    out = (out << 4U) | digit;
  }
  return out;
}

std::uint64_t checked_elements(const std::vector<std::size_t>& shape) {
  std::uint64_t count = 1U;
  for (const auto dim : shape) {
    if (dim == 0U) {
      throw std::runtime_error("BF16 slice: source shape has zero dimension");
    }
    if (count >
        std::numeric_limits<std::uint64_t>::max() /
            static_cast<std::uint64_t>(dim)) {
      throw std::runtime_error("BF16 slice: source shape overflows uint64");
    }
    count *= static_cast<std::uint64_t>(dim);
  }
  return count;
}

std::vector<std::byte> read_exact_file(
    const std::filesystem::path& path,
    std::uint64_t expected) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || size != expected) {
    throw std::runtime_error("BF16 slice: source file size mismatch");
  }
  if (expected > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("BF16 slice: source file is too large");
  }

  std::vector<std::byte> bytes(static_cast<std::size_t>(expected));
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("BF16 slice: unable to open source file");
  }
  input.read(
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("BF16 slice: short source read");
  }
  return bytes;
}

std::vector<std::byte> encode_f32_le(std::span<const float> values) {
  std::vector<std::byte> out(values.size() * sizeof(float));
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      out[i * 4U + lane] = static_cast<std::byte>(
          (bits >> (8U * lane)) & 0xFFU);
    }
  }
  return out;
}

}  // namespace

Bf16ConversionSliceManifest inspect_bf16_conversion_slice(
    const std::filesystem::path& manifest_path) {
  std::ifstream input(manifest_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "BF16 slice: unable to open manifest: " + manifest_path.string());
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("BF16 slice: invalid manifest JSON: ") + e.what());
  }

  if (root.value("schema_version", 0U) != 1U) {
    throw std::runtime_error("BF16 slice: unsupported schema_version");
  }
  if (root.value("source_dtype", std::string{}) != "BF16") {
    throw std::runtime_error("BF16 slice: source_dtype must be BF16");
  }
  if (root.value("output_dtype", std::string{}) != "F32") {
    throw std::runtime_error("BF16 slice: output_dtype must be F32");
  }

  Bf16ConversionSliceManifest out;
  try {
    out.model = root.at("model").get<std::string>();
    out.snapshot = root.at("snapshot").get<std::string>();
    out.tensor_name = root.at("tensor_name").get<std::string>();
    out.shard = root.at("shard").get<std::string>();
    out.source_shape =
        root.at("source_shape").get<std::vector<std::size_t>>();
    out.source_absolute_offset =
        root.at("source_absolute_offset").get<std::uint64_t>();
    out.source_byte_size =
        root.at("source_byte_size").get<std::uint64_t>();
    out.fetched_bytes =
        root.at("fetched_bytes").get<std::uint64_t>();
    out.source_file = root.at("source_file").get<std::string>();
    out.source_fnv1a64 =
        parse_hex_u64(root.at("source_fnv1a64").get<std::string>(),
                      "source_fnv1a64");
    out.output_byte_size =
        root.at("output_byte_size").get<std::uint64_t>();
    out.expected_f32_fnv1a64 =
        parse_hex_u64(root.at("expected_f32_fnv1a64").get<std::string>(),
                      "expected_f32_fnv1a64");
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("BF16 slice: malformed manifest: ") + e.what());
  }

  if (out.model.empty() || out.snapshot.empty() ||
      out.tensor_name.empty() || out.shard.empty() ||
      out.source_file.empty()) {
    throw std::runtime_error("BF16 slice: required string field is empty");
  }

  const auto elements = checked_elements(out.source_shape);
  if (elements > std::numeric_limits<std::uint64_t>::max() / 2U ||
      elements * 2U != out.source_byte_size) {
    throw std::runtime_error(
        "BF16 slice: source shape disagrees with BF16 byte size");
  }
  if (out.fetched_bytes != out.source_byte_size) {
    throw std::runtime_error(
        "BF16 slice: fetched_bytes must equal source_byte_size");
  }
  if (elements > std::numeric_limits<std::uint64_t>::max() / 4U ||
      elements * 4U != out.output_byte_size) {
    throw std::runtime_error(
        "BF16 slice: output size disagrees with F32 shape");
  }

  return out;
}

std::uint64_t fnv1a64(std::span<const std::byte> bytes) noexcept {
  std::uint64_t value = 0xCBF29CE484222325ULL;
  for (const auto byte : bytes) {
    value ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(byte));
    value *= 0x100000001B3ULL;
  }
  return value;
}

std::vector<float> decode_bf16_le(
    std::span<const std::byte> bytes) {
  if ((bytes.size() % 2U) != 0U) {
    throw std::runtime_error("BF16 slice: byte count must be even");
  }
  std::vector<float> out(bytes.size() / 2U);
  for (std::size_t i = 0U; i < out.size(); ++i) {
    const auto lo =
        static_cast<std::uint16_t>(
            std::to_integer<std::uint8_t>(bytes[i * 2U]));
    const auto hi =
        static_cast<std::uint16_t>(
            std::to_integer<std::uint8_t>(bytes[i * 2U + 1U]));
    const auto word = static_cast<std::uint16_t>(lo | (hi << 8U));
    const auto bits = static_cast<std::uint32_t>(word) << 16U;
    out[i] = std::bit_cast<float>(bits);
  }
  return out;
}

void write_f32_le(
    const std::filesystem::path& path,
    std::span<const float> values) {
  const auto bytes = encode_f32_le(values);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("BF16 slice: unable to open F32 output");
  }
  out.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (!out) {
    throw std::runtime_error("BF16 slice: unable to write F32 output");
  }
}

Bf16ConversionSliceResult convert_bf16_slice_to_f32(
    const Bf16ConversionSliceManifest& manifest,
    const std::filesystem::path& manifest_dir,
    const std::filesystem::path& output_path) {
  const auto source =
      read_exact_file(manifest_dir / manifest.source_file,
                      manifest.source_byte_size);

  Bf16ConversionSliceResult result;
  result.source_fnv1a64 = fnv1a64(source);
  if (result.source_fnv1a64 != manifest.source_fnv1a64) {
    throw std::runtime_error("BF16 slice: source FNV mismatch");
  }

  result.values = decode_bf16_le(source);
  const auto output_bytes = encode_f32_le(result.values);
  if (output_bytes.size() != manifest.output_byte_size) {
    throw std::runtime_error("BF16 slice: converted output size mismatch");
  }

  result.output_fnv1a64 = fnv1a64(output_bytes);
  if (result.output_fnv1a64 != manifest.expected_f32_fnv1a64) {
    throw std::runtime_error(
        "BF16 slice: converted F32 FNV disagrees with oracle");
  }

  write_f32_le(output_path, result.values);
  return result;
}

}  // namespace orbi::streammoe
