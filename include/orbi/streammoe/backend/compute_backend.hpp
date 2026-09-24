#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace orbi::streammoe {

enum class BackendKind : std::uint8_t {
  cpu_reference = 0,
  vulkan = 1,
};

class ComputeBackend {
 public:
  virtual ~ComputeBackend() = default;

  [[nodiscard]] virtual BackendKind kind() const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;

  // Intentionally minimal in OSM-00. Tensor/buffer contracts arrive only
  // after qpack and CPU-reference shapes are frozen.
};

}  // namespace orbi::streammoe
