#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace orbi::streammoe {

struct StreamedSafetensorSpec {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
};

struct StreamedSafetensorChunkState {
  std::size_t first_row{};
  std::size_t row_count{};
  std::uint64_t fnv1a64{};
};

struct StreamedSafetensorState {
  StreamedSafetensorSpec spec;
  std::uint64_t payload_offset{};
  std::uint64_t row_bytes{};
  std::size_t completed_rows{};
  std::vector<StreamedSafetensorChunkState> chunks;
};

struct StreamedSafetensorsState {
  std::uint64_t data_start{};
  std::uint64_t payload_bytes{};
  std::vector<StreamedSafetensorState> tensors;

  [[nodiscard]] bool complete() const noexcept;
};

class StreamedSafetensorsBuilder {
 public:
  [[nodiscard]] static StreamedSafetensorsBuilder open(
      std::filesystem::path output_path,
      std::filesystem::path journal_path,
      std::vector<StreamedSafetensorSpec> specs);

  [[nodiscard]] const StreamedSafetensorsState& state() const noexcept {
    return state_;
  }

  /// Commit a contiguous row chunk for one tensor.
  ///
  /// New writes must start exactly at completed_rows. Replaying an already
  /// committed exact chunk is idempotent after readback verification.
  [[nodiscard]] bool write_rows(
      const std::string& tensor_name,
      std::size_t first_row,
      std::size_t row_count,
      std::span<const std::byte> bytes);

  void verify_tensor(const std::string& tensor_name) const;
  void finalize() const;

 private:
  StreamedSafetensorsBuilder(
      std::filesystem::path output_path,
      std::filesystem::path journal_path,
      StreamedSafetensorsState state);

  std::filesystem::path output_path_;
  std::filesystem::path journal_path_;
  StreamedSafetensorsState state_;
};

struct StreamedAffineQ4Chunk {
  std::size_t rows{};
  std::size_t cols{};
  std::size_t packed_cols{};
  std::size_t groups_per_row{};
  std::vector<std::byte> packed_weight_bytes;
  std::vector<std::byte> scale_bytes;
  std::vector<std::byte> bias_bytes;
  float max_abs_error{};
  double mean_abs_error{};
};

/// Convert one bounded BF16 matrix row chunk into the runtime affine-Q4 triplet.
[[nodiscard]] StreamedAffineQ4Chunk convert_bf16_affine_q4_row_chunk(
    std::span<const std::byte> bf16_bytes,
    std::size_t rows,
    std::size_t cols,
    std::size_t group_size);

}  // namespace orbi::streammoe
