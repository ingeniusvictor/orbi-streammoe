#include "orbi/streammoe/container/qpack_dense.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace orbi::streammoe {
namespace {

std::filesystem::path validated_dense_path(
    const QpackReader& qpack) {
  constexpr const char* kDense = "model.safetensors";

  const auto it = qpack.manifest().files.find(kDense);
  if (it == qpack.manifest().files.end()) {
    throw std::runtime_error(
        "qpack dense: manifest does not declare model.safetensors");
  }

  const auto path = qpack.container_dir() / kDense;
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error(
        "qpack dense: unable to stat model.safetensors");
  }
  if (actual != it->second) {
    throw std::runtime_error(
        "qpack dense: model.safetensors size disagrees with manifest");
  }
  return path;
}

}  // namespace

QpackDenseReader::QpackDenseReader(const QpackReader& qpack)
    : safetensors_(validated_dense_path(qpack)) {}

std::string QpackDenseReader::resolve(
    std::string_view name) const {
  const std::string direct(name);
  if (safetensors_.contains(direct)) return direct;

  const std::string prefixed =
      "language_model." + direct;
  if (safetensors_.contains(prefixed)) return prefixed;

  return direct;
}

bool QpackDenseReader::contains(
    std::string_view name) const noexcept {
  try {
    return safetensors_.contains(resolve(name));
  } catch (...) {
    return false;
  }
}

const SafetensorInfo& QpackDenseReader::info(
    std::string_view name) const {
  return safetensors_.info(resolve(name));
}

std::uint64_t QpackDenseReader::absolute_offset(
    std::string_view name) const {
  return safetensors_.absolute_offset(resolve(name));
}

std::vector<float> QpackDenseReader::read_floats(
    std::string_view name) const {
  return safetensors_.read_floats(resolve(name));
}

std::vector<std::uint32_t> QpackDenseReader::read_u32(
    std::string_view name) const {
  return safetensors_.read_u32(resolve(name));
}

}  // namespace orbi::streammoe
