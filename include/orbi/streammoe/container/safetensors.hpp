#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace orbi::streammoe {

struct SafetensorInfo {
  std::string dtype;
  std::vector<std::size_t> shape;
  std::uint64_t data_begin{};
  std::uint64_t data_end{};

  [[nodiscard]] std::uint64_t byte_size() const noexcept {
    return data_end - data_begin;
  }
};

class SafetensorsReader {
 public:
  explicit SafetensorsReader(std::filesystem::path path);

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

  [[nodiscard]] std::uint64_t data_start() const noexcept {
    return data_start_;
  }

  [[nodiscard]] bool contains(std::string_view name) const noexcept;
  [[nodiscard]] const SafetensorInfo& info(std::string_view name) const;
  [[nodiscard]] std::vector<std::string> tensor_names() const;

  [[nodiscard]] std::uint64_t absolute_offset(
      std::string_view name) const;

  void read_raw(
      std::string_view name,
      std::uint64_t tensor_byte_offset,
      std::span<std::byte> destination) const;

  [[nodiscard]] std::vector<std::byte> read_raw(
      std::string_view name) const;

  [[nodiscard]] std::vector<float> read_floats(
      std::string_view name) const;

  [[nodiscard]] std::vector<float> read_floats(
      std::string_view name,
      std::size_t element_offset,
      std::size_t element_count) const;

  [[nodiscard]] std::vector<std::uint32_t> read_u32(
      std::string_view name) const;

  [[nodiscard]] std::vector<std::uint32_t> read_u32(
      std::string_view name,
      std::size_t element_offset,
      std::size_t element_count) const;

  [[nodiscard]] static std::size_t bytes_per_element(
      std::string_view dtype);

 private:
  std::filesystem::path path_;
  std::uint64_t file_size_{};
  std::uint64_t data_start_{};
  std::unordered_map<std::string, SafetensorInfo> tensors_;
};

}  // namespace orbi::streammoe
