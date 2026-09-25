#include "orbi/streammoe/container/safetensors.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

constexpr std::uint64_t kMaxHeaderBytes = 64ULL * 1024ULL * 1024ULL;

std::uint64_t read_u64_le(const std::array<std::byte, 8>& bytes) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<std::uint64_t>(
                 std::to_integer<std::uint8_t>(bytes[i]))
             << (8U * i);
  }
  return value;
}

std::uint16_t read_u16_le(const std::byte* p) {
  return static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(p[0])) |
      (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(p[1])) << 8U));
}

std::uint32_t read_u32_le(const std::byte* p) {
  return
      static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[0])) |
      (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[1])) << 8U) |
      (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[2])) << 16U) |
      (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(p[3])) << 24U);
}

float half_to_float(std::uint16_t h) noexcept {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(h & 0x8000U) << 16U;
  std::uint32_t exponent = (h >> 10U) & 0x1FU;
  std::uint32_t mantissa = h & 0x03FFU;

  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03FFU;
      const std::uint32_t fexp =
          static_cast<std::uint32_t>(127 - 15 - shift + 1);
      bits = sign | (fexp << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    exponent = exponent + (127U - 15U);
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::uint64_t checked_product(
    const std::vector<std::size_t>& shape) {
  std::uint64_t product = 1;
  for (const auto dim : shape) {
    if (dim == 0U) return 0U;
    if (product >
        std::numeric_limits<std::uint64_t>::max() /
            static_cast<std::uint64_t>(dim)) {
      throw std::runtime_error(
          "safetensors: tensor shape element count overflows uint64");
    }
    product *= static_cast<std::uint64_t>(dim);
  }
  return product;
}

void read_exact(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::span<std::byte> destination) {
  if (destination.empty()) return;

  if (offset >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(
        "safetensors: file offset exceeds streamoff range");
  }
  if (destination.size() >
      static_cast<std::size_t>(
          std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error(
        "safetensors: read size exceeds streamsize range");
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "safetensors: unable to open file: " + path.string());
  }

  input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!input) {
    throw std::runtime_error("safetensors: seek failed");
  }

  input.read(
      reinterpret_cast<char*>(destination.data()),
      static_cast<std::streamsize>(destination.size()));
  if (input.gcount() !=
      static_cast<std::streamsize>(destination.size())) {
    throw std::runtime_error("safetensors: short read");
  }
}

}  // namespace

std::size_t SafetensorsReader::bytes_per_element(
    std::string_view dtype) {
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4U;
  if (dtype == "F16" || dtype == "BF16" ||
      dtype == "I16" || dtype == "U16") return 2U;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8U;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") return 1U;
  throw std::runtime_error(
      "safetensors: unsupported dtype: " + std::string(dtype));
}

SafetensorsReader::SafetensorsReader(std::filesystem::path path)
    : path_(std::move(path)) {
  std::error_code ec;
  file_size_ = std::filesystem::file_size(path_, ec);
  if (ec) {
    throw std::runtime_error(
        "safetensors: unable to stat file: " + path_.string());
  }
  if (file_size_ < 8U) {
    throw std::runtime_error(
        "safetensors: file is shorter than header-length prefix");
  }

  std::array<std::byte, 8> prefix{};
  read_exact(path_, 0U, prefix);
  const auto header_size = read_u64_le(prefix);
  if (header_size == 0U || header_size > kMaxHeaderBytes) {
    throw std::runtime_error(
        "safetensors: header size is zero or exceeds safety limit");
  }
  if (header_size > file_size_ - 8U) {
    throw std::runtime_error(
        "safetensors: header exceeds file size");
  }

  std::vector<std::byte> header_bytes(
      static_cast<std::size_t>(header_size));
  read_exact(path_, 8U, header_bytes);

  std::string header_text(
      reinterpret_cast<const char*>(header_bytes.data()),
      header_bytes.size());

  json header;
  try {
    header = json::parse(header_text);
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("safetensors: malformed JSON header: ") + e.what());
  }
  if (!header.is_object()) {
    throw std::runtime_error(
        "safetensors: header root must be an object");
  }

  data_start_ = 8U + header_size;
  const auto payload_size = file_size_ - data_start_;

  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& entry = it.value();
    if (!entry.is_object()) {
      throw std::runtime_error(
          "safetensors: tensor entry must be an object: " + it.key());
    }

    SafetensorInfo info;
    try {
      info.dtype = entry.at("dtype").get<std::string>();
      info.shape = entry.at("shape").get<std::vector<std::size_t>>();
      const auto offsets =
          entry.at("data_offsets").get<std::vector<std::uint64_t>>();
      if (offsets.size() != 2U) {
        throw std::runtime_error(
            "safetensors: data_offsets must have two elements");
      }
      info.data_begin = offsets[0];
      info.data_end = offsets[1];
    } catch (const json::exception& e) {
      throw std::runtime_error(
          "safetensors: malformed tensor entry " + it.key() +
          ": " + e.what());
    }

    if (info.data_begin > info.data_end ||
        info.data_end > payload_size) {
      throw std::runtime_error(
          "safetensors: tensor data range exceeds payload: " + it.key());
    }

    const auto bytes = bytes_per_element(info.dtype);
    const auto elements = checked_product(info.shape);
    if (elements >
        std::numeric_limits<std::uint64_t>::max() /
            static_cast<std::uint64_t>(bytes)) {
      throw std::runtime_error(
          "safetensors: tensor byte count overflows uint64: " + it.key());
    }

    const auto expected =
        elements * static_cast<std::uint64_t>(bytes);
    if (expected != info.byte_size()) {
      throw std::runtime_error(
          "safetensors: tensor byte range disagrees with dtype/shape: " +
          it.key());
    }

    tensors_.emplace(it.key(), std::move(info));
  }

  if (tensors_.empty()) {
    throw std::runtime_error(
        "safetensors: file contains no tensors");
  }
}

bool SafetensorsReader::contains(
    std::string_view name) const noexcept {
  return tensors_.contains(std::string(name));
}

const SafetensorInfo& SafetensorsReader::info(
    std::string_view name) const {
  const auto it = tensors_.find(std::string(name));
  if (it == tensors_.end()) {
    throw std::runtime_error(
        "safetensors: missing tensor: " + std::string(name));
  }
  return it->second;
}

std::vector<std::string> SafetensorsReader::tensor_names() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto& [name, info] : tensors_) {
    (void)info;
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

std::uint64_t SafetensorsReader::absolute_offset(
    std::string_view name) const {
  const auto& tensor = info(name);
  if (tensor.data_begin >
      std::numeric_limits<std::uint64_t>::max() - data_start_) {
    throw std::runtime_error(
        "safetensors: absolute offset overflows uint64");
  }
  return data_start_ + tensor.data_begin;
}

void SafetensorsReader::read_raw(
    std::string_view name,
    std::uint64_t tensor_byte_offset,
    std::span<std::byte> destination) const {
  const auto& tensor = info(name);

  if (tensor_byte_offset > tensor.byte_size() ||
      destination.size() >
          tensor.byte_size() - tensor_byte_offset) {
    throw std::out_of_range(
        "safetensors: requested tensor byte range is out of bounds");
  }

  read_exact(
      path_,
      absolute_offset(name) + tensor_byte_offset,
      destination);
}

std::vector<std::byte> SafetensorsReader::read_raw(
    std::string_view name) const {
  const auto& tensor = info(name);
  if (tensor.byte_size() >
      static_cast<std::uint64_t>(
          std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "safetensors: tensor is too large for address space");
  }

  std::vector<std::byte> bytes(
      static_cast<std::size_t>(tensor.byte_size()));
  read_raw(name, 0U, bytes);
  return bytes;
}

std::vector<float> SafetensorsReader::read_floats(
    std::string_view name) const {
  const auto& tensor = info(name);
  const auto raw = read_raw(name);

  std::vector<float> values;
  if (tensor.dtype == "F32") {
    values.resize(raw.size() / 4U);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] =
          std::bit_cast<float>(read_u32_le(raw.data() + i * 4U));
    }
    return values;
  }

  if (tensor.dtype == "F16") {
    values.resize(raw.size() / 2U);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = half_to_float(
          read_u16_le(raw.data() + i * 2U));
    }
    return values;
  }

  if (tensor.dtype == "BF16") {
    values.resize(raw.size() / 2U);
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto bits =
          static_cast<std::uint32_t>(
              read_u16_le(raw.data() + i * 2U))
          << 16U;
      values[i] = std::bit_cast<float>(bits);
    }
    return values;
  }

  throw std::runtime_error(
      "safetensors: tensor is not F32/F16/BF16: " +
      std::string(name));
}

std::vector<std::uint32_t> SafetensorsReader::read_u32(
    std::string_view name) const {
  const auto& tensor = info(name);
  if (tensor.dtype != "U32") {
    throw std::runtime_error(
        "safetensors: tensor is not U32: " + std::string(name));
  }

  const auto raw = read_raw(name);
  std::vector<std::uint32_t> values(raw.size() / 4U);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = read_u32_le(raw.data() + i * 4U);
  }
  return values;
}

}  // namespace orbi::streammoe
