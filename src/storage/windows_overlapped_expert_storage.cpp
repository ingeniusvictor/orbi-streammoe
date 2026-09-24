#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "orbi/streammoe/storage/windows_overlapped_expert_storage.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {
namespace {

[[noreturn]] void throw_last_error(const char* context, DWORD error) {
  throw std::system_error(
      static_cast<int>(error),
      std::system_category(),
      context);
}

std::filesystem::path layer_path(
    const std::filesystem::path& container_dir,
    std::uint32_t layer) {
  std::ostringstream name;
  name << "layer_" << std::setfill('0') << std::setw(2) << layer << ".bin";
  return container_dir / "packed_experts" / name.str();
}

class EventHandle {
 public:
  EventHandle() {
    handle_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (handle_ == nullptr) {
      throw_last_error("CreateEventW", GetLastError());
    }
  }

  ~EventHandle() {
    if (handle_ != nullptr) {
      CloseHandle(handle_);
    }
  }

  EventHandle(const EventHandle&) = delete;
  EventHandle& operator=(const EventHandle&) = delete;

  EventHandle(EventHandle&& other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)) {}

  EventHandle& operator=(EventHandle&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (handle_ != nullptr) {
      CloseHandle(handle_);
    }
    handle_ = std::exchange(other.handle_, nullptr);
    return *this;
  }

  [[nodiscard]] HANDLE get() const noexcept { return handle_; }

 private:
  HANDLE handle_{};
};

struct PendingRead {
  OVERLAPPED overlapped{};
  EventHandle event;
  HANDLE file{INVALID_HANDLE_VALUE};
  DWORD expected_bytes{};
  bool launched{};
  DWORD launch_error{ERROR_SUCCESS};
};

}  // namespace

struct WindowsOverlappedExpertStorage::Impl {
  explicit Impl(std::filesystem::path dir)
      : container_dir(std::move(dir)),
        reader(container_dir),
        handles(reader.layout().layer_count, INVALID_HANDLE_VALUE) {}

  ~Impl() {
    for (const auto handle : handles) {
      if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
      }
    }
  }

  HANDLE handle_for_layer(std::uint32_t layer) {
    if (layer >= handles.size()) {
      throw std::out_of_range("windows expert storage: layer index out of range");
    }

    std::scoped_lock lock(handle_mutex);
    if (handles[layer] != INVALID_HANDLE_VALUE) {
      return handles[layer];
    }

    const auto path = layer_path(container_dir, layer);
    const HANDLE handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
        nullptr);

    if (handle == INVALID_HANDLE_VALUE) {
      throw_last_error("CreateFileW expert layer", GetLastError());
    }

    handles[layer] = handle;
    return handle;
  }

  std::filesystem::path container_dir;
  QpackReader reader;
  std::vector<HANDLE> handles;
  std::mutex handle_mutex;
};

WindowsOverlappedExpertStorage::WindowsOverlappedExpertStorage(
    std::filesystem::path container_dir)
    : impl_(std::make_unique<Impl>(std::move(container_dir))) {}

WindowsOverlappedExpertStorage::~WindowsOverlappedExpertStorage() = default;

WindowsOverlappedExpertStorage::WindowsOverlappedExpertStorage(
    WindowsOverlappedExpertStorage&&) noexcept = default;

WindowsOverlappedExpertStorage&
WindowsOverlappedExpertStorage::operator=(
    WindowsOverlappedExpertStorage&&) noexcept = default;

std::size_t WindowsOverlappedExpertStorage::expert_stride_bytes() const noexcept {
  return static_cast<std::size_t>(impl_->reader.layout().expert_stride);
}

void WindowsOverlappedExpertStorage::read_experts(
    std::span<ExpertReadRequest> requests) {
  if (requests.empty()) {
    return;
  }

  const auto stride = expert_stride_bytes();
  if (stride > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
    throw std::runtime_error(
        "windows expert storage: expert stride exceeds Win32 ReadFile size");
  }

  // Build every OVERLAPPED/event/handle before launching any I/O. This keeps
  // all OVERLAPPED storage stable for the entire batch and ensures a setup
  // failure cannot destroy metadata for reads that are already in flight.
  std::vector<PendingRead> pending;
  pending.reserve(requests.size());

  for (auto& request : requests) {
    if (request.id.layer >= impl_->reader.layout().layer_count) {
      throw std::out_of_range("windows expert storage: layer index out of range");
    }
    if (request.id.expert >= impl_->reader.layout().expert_count) {
      throw std::out_of_range("windows expert storage: expert index out of range");
    }
    if (request.destination.size() < stride) {
      throw std::invalid_argument(
          "windows expert storage: destination smaller than expert stride");
    }

    const auto offset =
        static_cast<std::uint64_t>(request.id.expert) *
        static_cast<std::uint64_t>(stride);

    pending.emplace_back();
    auto& item = pending.back();
    item.file = impl_->handle_for_layer(request.id.layer);
    item.expected_bytes = static_cast<DWORD>(stride);
    item.overlapped.hEvent = item.event.get();
    item.overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    item.overlapped.OffsetHigh =
        static_cast<DWORD>((offset >> 32U) & 0xFFFFFFFFULL);
  }

  // Launch the whole batch before waiting. An immediate launch failure is
  // recorded rather than thrown so already-started reads retain live
  // OVERLAPPED/event storage until they are drained below.
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& item = pending[i];
    auto& request = requests[i];

    const BOOL started = ReadFile(
        item.file,
        request.destination.data(),
        item.expected_bytes,
        nullptr,
        &item.overlapped);

    if (started != FALSE) {
      item.launched = true;
      continue;
    }

    const DWORD error = GetLastError();
    if (error == ERROR_IO_PENDING) {
      item.launched = true;
    } else {
      item.launch_error = error;
    }
  }

  std::exception_ptr first_failure;

  for (auto& item : pending) {
    try {
      if (!item.launched) {
        throw_last_error("ReadFile expert", item.launch_error);
      }

      DWORD transferred = 0;
      const BOOL ok = GetOverlappedResult(
          item.file,
          &item.overlapped,
          &transferred,
          TRUE);

      if (ok == FALSE) {
        throw_last_error("GetOverlappedResult expert", GetLastError());
      }
      if (transferred != item.expected_bytes) {
        throw std::runtime_error(
            "windows expert storage: short expert read");
      }
    } catch (...) {
      if (!first_failure) {
        first_failure = std::current_exception();
      }
    }
  }

  if (first_failure) {
    std::rethrow_exception(first_failure);
  }
}

}  // namespace orbi::streammoe

#endif  // _WIN32
