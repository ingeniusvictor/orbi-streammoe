#include "orbi/streammoe/conversion/multi_expert_orchestrator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace orbi::streammoe {

MultiExpertConversionResult convert_qwen_expert_range(
    const std::vector<MultiExpertInput>& inputs,
    const QpackExpertGeometry& geometry,
    QpackLayerWriter& writer,
    const MultiExpertConversionOptions& options,
    const std::filesystem::path& scratch_dir) {
  if (options.first_expert >= options.end_expert_exclusive ||
      options.end_expert_exclusive > geometry.expert_count) {
    throw std::invalid_argument(
        "multi expert conversion: invalid expert range");
  }
  if (writer.state().expert_count != geometry.expert_count ||
      writer.state().expert_stride != geometry.expert_stride) {
    throw std::runtime_error(
        "multi expert conversion: writer geometry mismatch");
  }

  std::map<std::size_t, MultiExpertInput> by_expert;
  std::optional<std::size_t> layer_index;

  for (const auto& input : inputs) {
    const auto manifest =
        inspect_single_expert_pilot_manifest(input.manifest_path);

    if (!layer_index.has_value()) {
      layer_index = manifest.layer_index;
    } else if (*layer_index != manifest.layer_index) {
      throw std::runtime_error(
          "multi expert conversion: manifests span multiple layers");
    }

    if (manifest.layer_index != writer.state().layer_index) {
      throw std::runtime_error(
          "multi expert conversion: manifest layer disagrees with writer");
    }
    if (!by_expert.emplace(manifest.expert_index, input).second) {
      throw std::runtime_error(
          "multi expert conversion: duplicate expert manifest");
    }
  }

  MultiExpertConversionResult result;
  result.requested_experts =
      options.end_expert_exclusive - options.first_expert;

  double weighted_error_sum = 0.0;
  double weighted_value_count = 0.0;

  std::filesystem::create_directories(scratch_dir);

  for (std::size_t expert = options.first_expert;
       expert < options.end_expert_exclusive;
       ++expert) {
    if (writer.state().expert_complete(expert)) {
      writer.verify_expert(expert);
      ++result.skipped_completed_experts;
      result.skipped_expert_ids.push_back(expert);
      continue;
    }

    const auto it = by_expert.find(expert);
    if (it == by_expert.end()) {
      throw std::runtime_error(
          "multi expert conversion: missing manifest for requested expert " +
          std::to_string(expert));
    }

    const auto manifest =
        inspect_single_expert_pilot_manifest(it->second.manifest_path);
    const auto temporary =
        scratch_dir /
        ("expert_" + std::to_string(expert) + ".qpack.tmp");

    const auto converted = convert_qwen_single_expert_pilot(
        manifest,
        it->second.manifest_dir,
        geometry,
        temporary);

    if (converted.qpack_blob.size() != geometry.expert_stride) {
      std::filesystem::remove(temporary);
      throw std::runtime_error(
          "multi expert conversion: converted expert stride mismatch");
    }

    const bool committed =
        writer.write_expert(expert, converted.qpack_blob);
    std::filesystem::remove(temporary);

    if (!committed) {
      throw std::runtime_error(
          "multi expert conversion: unexpected completed expert race");
    }

    ++result.converted_experts;
    result.converted_expert_ids.push_back(expert);
    result.source_bytes += converted.source_bytes;
    result.max_abs_error =
        std::max(result.max_abs_error, converted.max_abs_error);

    const double values =
        static_cast<double>(geometry.moe_intermediate_size) *
            static_cast<double>(geometry.hidden_size) * 2.0 +
        static_cast<double>(geometry.hidden_size) *
            static_cast<double>(geometry.moe_intermediate_size);
    weighted_error_sum += converted.mean_abs_error * values;
    weighted_value_count += values;
  }

  if (weighted_value_count > 0.0) {
    result.weighted_mean_abs_error =
        weighted_error_sum / weighted_value_count;
  }
  if (!std::isfinite(result.weighted_mean_abs_error) ||
      !std::isfinite(result.max_abs_error)) {
    throw std::runtime_error(
        "multi expert conversion: non-finite aggregate quantization metrics");
  }

  if (options.finalize_if_complete) {
    writer.finalize();
  }

  return result;
}

}  // namespace orbi::streammoe
