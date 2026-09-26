#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"

namespace orbi::streammoe {

struct ExpertPilotProjectionManifest {
  std::string name;
  std::string tensor_name;
  std::string shard;
  std::vector<std::size_t> shape;
  std::uint64_t source_byte_size{};
  std::string source_file;
  std::uint64_t source_fnv1a64{};
};

struct SingleExpertPilotManifest {
  std::string model;
  std::string snapshot;
  std::size_t layer_index{};
  std::size_t expert_index{};
  std::uint64_t total_fetched_bytes{};
  std::vector<ExpertPilotProjectionManifest> projections;
};

struct AffineQ4Projection {
  std::size_t rows{};
  std::size_t cols{};
  std::size_t packed_cols{};
  std::size_t groups_per_row{};
  std::vector<std::uint32_t> packed;
  std::vector<float> scales;
  std::vector<float> biases;
  float max_abs_error{};
  double mean_abs_error{};
};

struct SingleExpertConversionResult {
  AffineQ4Projection gate;
  AffineQ4Projection up;
  AffineQ4Projection down;
  std::uint64_t source_bytes{};
  std::vector<std::byte> qpack_blob;
  float max_abs_error{};
  double mean_abs_error{};
};

[[nodiscard]] SingleExpertPilotManifest inspect_single_expert_pilot_manifest(
    const std::filesystem::path& manifest_path);

[[nodiscard]] AffineQ4Projection quantize_affine_q4_rows(
    const std::vector<float>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t group_size);

[[nodiscard]] SingleExpertConversionResult convert_qwen_single_expert_pilot(
    const SingleExpertPilotManifest& manifest,
    const std::filesystem::path& manifest_dir,
    const QpackExpertGeometry& geometry,
    const std::filesystem::path& output_blob);

}  // namespace orbi::streammoe
