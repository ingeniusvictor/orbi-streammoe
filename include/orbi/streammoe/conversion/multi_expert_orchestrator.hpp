#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "orbi/streammoe/conversion/qpack_layer_writer.hpp"
#include "orbi/streammoe/conversion/single_expert_pilot.hpp"

namespace orbi::streammoe {

struct MultiExpertInput {
  std::filesystem::path manifest_path;
  std::filesystem::path manifest_dir;
};

struct MultiExpertConversionOptions {
  std::size_t first_expert{};
  std::size_t end_expert_exclusive{};
  bool finalize_if_complete{false};
};

struct MultiExpertConversionResult {
  std::size_t requested_experts{};
  std::size_t converted_experts{};
  std::size_t skipped_completed_experts{};
  std::uint64_t source_bytes{};
  float max_abs_error{};
  double weighted_mean_abs_error{};
  std::vector<std::size_t> converted_expert_ids;
  std::vector<std::size_t> skipped_expert_ids;
};

/// Convert a bounded expert range one expert at a time and commit each blob
/// through the resumable OSM-39D fixed-stride writer.
///
/// Peak conversion memory remains bounded by one source expert plus one QPACK
/// expert blob. Experts already certified in the writer journal are skipped
/// before source payload is read.
[[nodiscard]] MultiExpertConversionResult convert_qwen_expert_range(
    const std::vector<MultiExpertInput>& inputs,
    const QpackExpertGeometry& geometry,
    QpackLayerWriter& writer,
    const MultiExpertConversionOptions& options,
    const std::filesystem::path& scratch_dir);

}  // namespace orbi::streammoe
