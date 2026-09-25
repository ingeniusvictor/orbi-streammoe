#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/container/safetensors.hpp"

namespace orbi::streammoe {

class QpackDenseReader {
 public:
  explicit QpackDenseReader(const QpackReader& qpack);

  [[nodiscard]] const SafetensorsReader& safetensors() const noexcept {
    return safetensors_;
  }

  [[nodiscard]] bool contains(std::string_view name) const noexcept;
  [[nodiscard]] const SafetensorInfo& info(std::string_view name) const;
  [[nodiscard]] std::uint64_t absolute_offset(
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

 private:
  SafetensorsReader safetensors_;

  [[nodiscard]] std::string resolve(
      std::string_view name) const;
};

}  // namespace orbi::streammoe
