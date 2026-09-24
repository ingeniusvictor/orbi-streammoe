#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace orbi::streammoe {

struct ExpertId {
  std::uint32_t layer{};
  std::uint32_t expert{};
};

struct ExpertReadRequest {
  ExpertId id{};
  std::span<std::byte> destination{};
};

class ExpertStorage {
 public:
  virtual ~ExpertStorage() = default;

  [[nodiscard]] virtual std::size_t expert_stride_bytes() const noexcept = 0;

  // Contract: every destination is fully populated or the call fails.
  // Implementations may service requests concurrently.
  virtual void read_experts(std::span<ExpertReadRequest> requests) = 0;
};

}  // namespace orbi::streammoe
