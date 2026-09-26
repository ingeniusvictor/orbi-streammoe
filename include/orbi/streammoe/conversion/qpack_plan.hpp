#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace orbi::streammoe {

enum class QpackConversionClass {
  global_dense,
  layer_dense,
  routed_expert,
  auxiliary_mtp,
};

enum class QpackConversionAction {
  copy_bf16_to_f32,
  affine_quantize,
  split_packed_gate_up_experts,
  split_packed_down_experts,
  direct_expert_quantize,
  exclude_auxiliary_mtp,
};

struct QpackConversionPlanEntry {
  std::string source_tensor;
  std::string source_shard;
  QpackConversionClass tensor_class{QpackConversionClass::layer_dense};
  QpackConversionAction action{QpackConversionAction::affine_quantize};
  std::string target_file;
  std::string target_path;
  std::optional<std::size_t> layer_index;
};

struct QpackConversionPlan {
  std::string source_checkpoint;
  std::size_t layer_count{};
  std::size_t expert_count{};
  std::vector<QpackConversionPlanEntry> entries;

  [[nodiscard]] std::size_t count(QpackConversionClass value) const noexcept;
  [[nodiscard]] std::size_t count(QpackConversionAction value) const noexcept;
};

/// Build a deterministic metadata-only plan from config.json and
/// model.safetensors.index.json.
///
/// No tensor payload is read. Every checkpoint tensor must map to exactly one
/// conversion action and one QPACK destination.
[[nodiscard]] QpackConversionPlan build_qpack_conversion_plan(
    const std::filesystem::path& model_dir);

void validate_qpack_conversion_plan(const QpackConversionPlan& plan);

[[nodiscard]] const char* to_string(QpackConversionClass value) noexcept;
[[nodiscard]] const char* to_string(QpackConversionAction value) noexcept;

}  // namespace orbi::streammoe
