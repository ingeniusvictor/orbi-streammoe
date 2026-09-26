#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"

namespace orbi::streammoe {

struct FullCheckpointPackageResult {
  std::filesystem::path manifest_path;
  std::filesystem::path config_path;
  std::filesystem::path dense_path;
  std::filesystem::path provenance_path;
  std::size_t dense_tensor_count{};
  std::uint64_t dense_file_bytes{};
  std::uint64_t expert_payload_bytes{};
};

/// Promote a verified OSM-39G expert shell plus a complete OSM-40B dense
/// safetensors output into the full runtime checkpoint contract.
///
/// The dense journal is reopened through StreamedSafetensorsBuilder and every
/// committed chunk is verified before any package metadata is changed.
/// The resulting package is finally reopened through QpackReader and
/// QpackMlxCheckpoint.
[[nodiscard]] FullCheckpointPackageResult finalize_full_qpack_checkpoint(
    const std::filesystem::path& output_dir,
    const std::filesystem::path& dense_journal_path,
    QpackExpertQuantizationSpec quantization = {});

}  // namespace orbi::streammoe
