#include "orbi/streammoe/storage/qpack_expert_storage.hpp"

#include <stdexcept>
#include <utility>

namespace orbi::streammoe {

QpackExpertStorage::QpackExpertStorage(std::filesystem::path container_dir)
    : reader_(std::move(container_dir)) {}

std::size_t QpackExpertStorage::expert_stride_bytes() const noexcept {
  return static_cast<std::size_t>(reader_.layout().expert_stride);
}

void QpackExpertStorage::read_experts(
    std::span<ExpertReadRequest> requests) {
  const auto stride = expert_stride_bytes();

  for (auto& request : requests) {
    if (request.destination.size() < stride) {
      throw std::invalid_argument(
          "qpack expert storage: destination smaller than expert stride");
    }

    reader_.read_expert(
        request.id.layer,
        request.id.expert,
        request.destination);
  }
}

}  // namespace orbi::streammoe
