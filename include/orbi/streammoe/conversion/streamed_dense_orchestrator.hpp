#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"
#include "orbi/streammoe/conversion/qpack_plan.hpp"
#include "orbi/streammoe/conversion/streamed_safetensors_builder.hpp"

namespace orbi::streammoe {

struct StreamedDenseSourceChunk {
  std::size_t first_row{};
  std::size_t row_count{};
  std::string source_file;
  std::uint64_t source_byte_size{};
  std::uint64_t source_fnv1a64{};
};

struct StreamedDenseSourceTensor {
  std::string source_tensor;
  std::vector<std::size_t> source_shape;
  std::vector<StreamedDenseSourceChunk> chunks;
};

struct StreamedDenseManifest {
  std::string model;
  std::string snapshot;
  std::vector<StreamedDenseSourceTensor> tensors;
};

struct StreamedDenseOrchestratorResult {
  std::size_t tensor_count{};
  std::size_t requested_chunks{};
  std::size_t converted_chunks{};
  std::size_t skipped_chunks{};
  std::uint64_t source_bytes{};
  float max_abs_error{};
  double weighted_mean_abs_error{};
};

[[nodiscard]] StreamedDenseManifest inspect_streamed_dense_manifest(
    const std::filesystem::path& manifest_path);

/// Build the exact target safetensors inventory for the supplied dense/global
/// source tensors without materializing payloads.
[[nodiscard]] std::vector<StreamedSafetensorSpec>
plan_streamed_dense_output_specs(
    const QpackConversionPlan& plan,
    const StreamedDenseManifest& manifest,
    QpackExpertQuantizationSpec quantization = {});

/// Execute the listed source chunks against a pre-opened streamed builder.
///
/// Already-certified output rows are skipped before the source file is read.
/// This keeps restart behavior bounded by the missing row chunks only.
[[nodiscard]] StreamedDenseOrchestratorResult execute_streamed_dense_manifest(
    const QpackConversionPlan& plan,
    const StreamedDenseManifest& manifest,
    const std::filesystem::path& manifest_dir,
    StreamedSafetensorsBuilder& builder,
    QpackExpertQuantizationSpec quantization = {});

}  // namespace orbi::streammoe
