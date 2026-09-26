#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"

namespace orbi::streammoe {

struct QpackExpertPackagePlan {
  QpackExpertGeometry geometry;
  std::size_t full_attention_interval{};
  std::vector<bool> linear_layers;
  std::vector<std::string> layer_relative_paths;
  std::uint64_t expert_payload_bytes{};
};

struct QpackExpertPackageOptions {
  std::string model_name{"qwen3_next"};
  std::string source_checkpoint;
  std::string source_snapshot;
};

struct QpackExpertPackageResult {
  std::filesystem::path manifest_path;
  std::filesystem::path layout_path;
  std::filesystem::path provenance_path;
  std::size_t layer_count{};
  std::size_t declared_file_count{};
  std::uint64_t expert_payload_bytes{};
};

/// Metadata-only package plan derived from source config.json.
///
/// No expert payload is read.
[[nodiscard]] QpackExpertPackagePlan derive_qpack_expert_package_plan(
    const std::filesystem::path& model_dir,
    QpackExpertQuantizationSpec quantization = {});

/// Finalize an in-place expert package.
///
/// Expects canonical converted files:
///   packed_experts/layer_XX.bin
///   packed_experts/layer_XX.progress.json (or .bak)
///
/// Every journal must be complete and every expert checksum is re-verified by
/// OSM-39D before deterministic layout/manifest/provenance files are written.
[[nodiscard]] QpackExpertPackageResult finalize_qpack_expert_package(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& output_dir,
    const QpackExpertPackageOptions& options,
    QpackExpertQuantizationSpec quantization = {});

}  // namespace orbi::streammoe
