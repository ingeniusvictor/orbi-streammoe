#if !defined(_WIN32)

#include "orbi/streammoe/storage/posix_pread_expert_storage.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {
namespace {

std::filesystem::path layer_path(
    const std::filesystem::path& container_dir,
    std::uint32_t layer) {
  std::ostringstream name;
  name << "layer_" << std::setfill('0') << std::setw(2) << layer << ".bin";
  return container_dir / "packed_experts" / name.str();
}

[[noreturn]] void throw_errno(const char* context, int error) {
  throw std::system_error(error, std::generic_category(), context);
}

}  // namespace

struct PosixPreadExpertStorage::Impl {
  explicit Impl(std::filesystem::path dir)
      : container_dir(std::move(dir)),
        reader(container_dir),
        fds(reader.layout().layer_count, -1) {}

  ~Impl() {
    for (const int fd : fds) {
      if (fd >= 0) {
        ::close(fd);
      }
    }
  }

  int fd_for_layer(std::uint32_t layer) {
    if (layer >= fds.size()) {
      throw std::out_of_range("posix expert storage: layer index out of range");
    }

    std::scoped_lock lock(fd_mutex);
    if (fds[layer] >= 0) {
      return fds[layer];
    }

    const auto path = layer_path(container_dir, layer);
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw_errno("open expert layer", errno);
    }

    fds[layer] = fd;
    return fd;
  }

  std::filesystem::path container_dir;
  QpackReader reader;
  std::vector<int> fds;
  std::mutex fd_mutex;
};

PosixPreadExpertStorage::PosixPreadExpertStorage(
    std::filesystem::path container_dir)
    : impl_(std::make_unique<Impl>(std::move(container_dir))) {}

PosixPreadExpertStorage::~PosixPreadExpertStorage() = default;

PosixPreadExpertStorage::PosixPreadExpertStorage(
    PosixPreadExpertStorage&&) noexcept = default;

PosixPreadExpertStorage&
PosixPreadExpertStorage::operator=(
    PosixPreadExpertStorage&&) noexcept = default;

std::size_t PosixPreadExpertStorage::expert_stride_bytes() const noexcept {
  return static_cast<std::size_t>(impl_->reader.layout().expert_stride);
}

void PosixPreadExpertStorage::read_experts(
    std::span<ExpertReadRequest> requests) {
  if (requests.empty()) {
    return;
  }

  const auto stride = expert_stride_bytes();
  if (stride > static_cast<std::size_t>(
                   std::numeric_limits<ssize_t>::max())) {
    throw std::runtime_error(
        "posix expert storage: expert stride exceeds pread size range");
  }

  struct PreparedRead {
    int fd{};
    off_t offset{};
    std::byte* destination{};
  };

  // Resolve/validate everything before creating worker threads so a setup
  // failure cannot leave partially launched reads.
  std::vector<PreparedRead> prepared;
  prepared.reserve(requests.size());

  for (auto& request : requests) {
    if (request.id.layer >= impl_->reader.layout().layer_count) {
      throw std::out_of_range("posix expert storage: layer index out of range");
    }
    if (request.id.expert >= impl_->reader.layout().expert_count) {
      throw std::out_of_range("posix expert storage: expert index out of range");
    }
    if (request.destination.size() < stride) {
      throw std::invalid_argument(
          "posix expert storage: destination smaller than expert stride");
    }

    const auto offset_u64 =
        static_cast<std::uint64_t>(request.id.expert) *
        static_cast<std::uint64_t>(stride);

    if (offset_u64 >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      throw std::runtime_error(
          "posix expert storage: expert offset exceeds off_t range");
    }

    prepared.push_back({
        .fd = impl_->fd_for_layer(request.id.layer),
        .offset = static_cast<off_t>(offset_u64),
        .destination = request.destination.data(),
    });
  }

  std::vector<std::exception_ptr> failures(requests.size());
  std::vector<std::jthread> workers;
  workers.reserve(requests.size());

  for (std::size_t i = 0; i < prepared.size(); ++i) {
    workers.emplace_back([&, i] {
      try {
        const auto& item = prepared[i];
        const ssize_t result = ::pread(
            item.fd,
            item.destination,
            stride,
            item.offset);

        if (result < 0) {
          throw_errno("pread expert", errno);
        }
        if (static_cast<std::size_t>(result) != stride) {
          throw std::runtime_error(
              "posix expert storage: short expert read");
        }
      } catch (...) {
        failures[i] = std::current_exception();
      }
    });
  }

  // Destroying jthreads joins them. Clear explicitly before examining the
  // failure slots so all worker writes are complete.
  workers.clear();

  for (const auto& failure : failures) {
    if (failure) {
      std::rethrow_exception(failure);
    }
  }
}

}  // namespace orbi::streammoe

#endif  // !_WIN32
