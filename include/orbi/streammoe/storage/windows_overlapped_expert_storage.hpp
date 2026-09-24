#pragma once

#if defined(_WIN32)

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

#include "orbi/streammoe/storage/expert_storage.hpp"

namespace orbi::streammoe {

class WindowsOverlappedExpertStorage final : public ExpertStorage {
 public:
  explicit WindowsOverlappedExpertStorage(std::filesystem::path container_dir);
  ~WindowsOverlappedExpertStorage() override;

  WindowsOverlappedExpertStorage(const WindowsOverlappedExpertStorage&) = delete;
  WindowsOverlappedExpertStorage& operator=(const WindowsOverlappedExpertStorage&) = delete;
  WindowsOverlappedExpertStorage(WindowsOverlappedExpertStorage&&) noexcept;
  WindowsOverlappedExpertStorage& operator=(WindowsOverlappedExpertStorage&&) noexcept;

  [[nodiscard]] std::size_t expert_stride_bytes() const noexcept override;
  void read_experts(std::span<ExpertReadRequest> requests) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe

#endif  // _WIN32
