#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {

struct QpackExpertQuantizationSpec {
  std::uint32_t bits{4U};
  std::uint32_t group_size{64U};
};

struct QpackExpertGeometry {
  std::size_t hidden_size{};
  std::size_t moe_intermediate_size{};
  std::size_t expert_count{};
  std::size_t layer_count{};

  QpackExpertQuantizationSpec quantization;
  std::uint64_t expert_stride{};
  std::uint64_t layer_bytes{};
  std::uint64_t all_layers_bytes{};
  std::vector<QpackSection> sections;
};

/// Derive the exact contiguous routed-expert QPACK layout from the source
/// checkpoint config plus the target affine quantization contract.
///
/// The resulting section offsets are deterministic and directly consumable by
/// QPACK layout.json. No tensor payload is read.
[[nodiscard]] QpackExpertGeometry derive_qpack_expert_geometry(
    const std::filesystem::path& model_dir,
    QpackExpertQuantizationSpec quantization = {});

void validate_qpack_expert_geometry(const QpackExpertGeometry& geometry);

}  // namespace orbi::streammoe
