#pragma once

#include <filesystem>

#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/storage/expert_storage.hpp"

namespace orbi::streammoe {

class QpackExpertStorage final : public ExpertStorage {
 public:
  explicit QpackExpertStorage(std::filesystem::path container_dir);

  [[nodiscard]] std::size_t expert_stride_bytes() const noexcept override;

  void read_experts(std::span<ExpertReadRequest> requests) override;

  [[nodiscard]] const QpackReader& reader() const noexcept { return reader_; }

 private:
  QpackReader reader_;
};

}  // namespace orbi::streammoe
