#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "orbi/streammoe/conversion/qpack_geometry.hpp"

namespace orbi::streammoe {

struct QpackLayerResumeState {
  std::size_t layer_index{};
  std::size_t expert_count{};
  std::uint64_t expert_stride{};
  std::size_t completed_experts{};
  std::vector<std::uint64_t> expert_fnv1a64;

  [[nodiscard]] bool expert_complete(std::size_t expert) const noexcept {
    return expert < expert_fnv1a64.size() && expert_fnv1a64[expert] != 0U;
  }

  [[nodiscard]] bool complete() const noexcept {
    return expert_count != 0U && completed_experts == expert_count;
  }
};

class QpackLayerWriter {
 public:
  [[nodiscard]] static QpackLayerWriter open(
      std::filesystem::path layer_path,
      std::filesystem::path journal_path,
      const QpackExpertGeometry& geometry,
      std::size_t layer_index);

  [[nodiscard]] const QpackLayerResumeState& state() const noexcept {
    return state_;
  }

  /// Write one exact expert blob at expert * expertStride.
  ///
  /// Returns true when a new expert was committed. If the expert was already
  /// committed with identical bytes, verifies readback and returns false.
  /// A hash mismatch is rejected instead of silently overwriting certified data.
  [[nodiscard]] bool write_expert(
      std::size_t expert_index,
      std::span<const std::byte> blob);

  /// Re-read one committed expert and verify its journal checksum.
  void verify_expert(std::size_t expert_index) const;

  /// Require every expert slot in the layer to have a verified journal entry.
  void finalize() const;

 private:
  QpackLayerWriter(
      std::filesystem::path layer_path,
      std::filesystem::path journal_path,
      QpackLayerResumeState state);

  std::filesystem::path layer_path_;
  std::filesystem::path journal_path_;
  QpackLayerResumeState state_;
};

}  // namespace orbi::streammoe
