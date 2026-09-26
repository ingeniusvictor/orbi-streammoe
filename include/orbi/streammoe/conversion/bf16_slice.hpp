#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace orbi::streammoe {

struct Bf16ConversionSliceManifest {
  std::string model;
  std::string snapshot;
  std::string tensor_name;
  std::string shard;
  std::vector<std::size_t> source_shape;
  std::uint64_t source_absolute_offset{};
  std::uint64_t source_byte_size{};
  std::uint64_t fetched_bytes{};
  std::string source_file;
  std::uint64_t source_fnv1a64{};
  std::uint64_t output_byte_size{};
  std::uint64_t expected_f32_fnv1a64{};
};

struct Bf16ConversionSliceResult {
  std::vector<float> values;
  std::uint64_t source_fnv1a64{};
  std::uint64_t output_fnv1a64{};
};

[[nodiscard]] Bf16ConversionSliceManifest inspect_bf16_conversion_slice(
    const std::filesystem::path& manifest_path);

[[nodiscard]] std::uint64_t fnv1a64(
    std::span<const std::byte> bytes) noexcept;

[[nodiscard]] std::vector<float> decode_bf16_le(
    std::span<const std::byte> bytes);

void write_f32_le(
    const std::filesystem::path& path,
    std::span<const float> values);

[[nodiscard]] Bf16ConversionSliceResult convert_bf16_slice_to_f32(
    const Bf16ConversionSliceManifest& manifest,
    const std::filesystem::path& manifest_dir,
    const std::filesystem::path& output_path);

}  // namespace orbi::streammoe
