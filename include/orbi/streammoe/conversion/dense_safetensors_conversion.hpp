#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"
#include "orbi/streammoe/conversion/qpack_plan.hpp"

namespace orbi::streammoe {

struct DenseSourceTensorManifest {
  std::string source_tensor;
  std::vector<std::size_t> source_shape;
  std::uint64_t source_byte_size{};
  std::string source_file;
  std::uint64_t source_fnv1a64{};
};

struct DenseConversionManifest {
  std::string model;
  std::string snapshot;
  std::vector<DenseSourceTensorManifest> tensors;
};

struct DenseSafetensorsConversionResult {
  std::size_t source_tensor_count{};
  std::size_t output_tensor_count{};
  std::uint64_t source_bytes{};
  std::uint64_t output_payload_bytes{};
  std::vector<std::string> output_tensor_names;
};

[[nodiscard]] DenseConversionManifest inspect_dense_conversion_manifest(
    const std::filesystem::path& manifest_path);

/// Execute a bounded non-expert slice of the deterministic OSM-39A plan.
///
/// Supported actions:
/// - copy_bf16_to_f32
/// - affine_quantize (4-bit affine)
///
/// The output is a deterministic safetensors file accepted by
/// SafetensorsReader / QpackDenseReader. This gate intentionally materializes
/// only the explicitly supplied bounded tensors. Production-scale embedding
/// and LM-head streaming is a subsequent gate.
[[nodiscard]] DenseSafetensorsConversionResult convert_dense_plan_slice(
    const QpackConversionPlan& plan,
    const DenseConversionManifest& manifest,
    const std::filesystem::path& manifest_dir,
    const std::filesystem::path& output_safetensors,
    QpackExpertQuantizationSpec quantization = {});

}  // namespace orbi::streammoe
