#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace orbi::streammoe {
namespace {

template <typename T>
std::span<const T> typed_section(
    std::span<const std::byte> bytes,
    const QpackSection& section,
    const char* label) {
  if ((section.offset % alignof(T)) != 0U) {
    throw std::runtime_error(
        std::string("qpack expert: misaligned ") + label + " section");
  }
  if ((section.size % sizeof(T)) != 0U) {
    throw std::runtime_error(
        std::string("qpack expert: invalid ") + label + " section size");
  }
  if (section.offset > bytes.size() ||
      section.size > bytes.size() - section.offset) {
    throw std::runtime_error(
        std::string("qpack expert: ") + label + " section exceeds cache slot");
  }

  const auto* ptr = reinterpret_cast<const T*>(
      bytes.data() + static_cast<std::size_t>(section.offset));
  return {
      ptr,
      static_cast<std::size_t>(section.size / sizeof(T)),
  };
}

std::size_t product(std::span<const std::size_t> shape) {
  std::size_t value = 1;
  for (const auto dim : shape) {
    if (dim == 0 || value > std::numeric_limits<std::size_t>::max() / dim) {
      throw std::runtime_error("qpack expert: invalid/overflowing section shape");
    }
    value *= dim;
  }
  return value;
}

float bf16_to_float(std::uint16_t value) noexcept {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

float f16_to_float(std::uint16_t value) noexcept {
  const std::uint32_t sign =
      (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t mantissa = value & 0x03FFU;

  std::uint32_t bits = 0;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      int exponent_unbiased = -14;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        --exponent_unbiased;
      }
      mantissa &= 0x03FFU;
      const auto exponent32 =
          static_cast<std::uint32_t>(exponent_unbiased + 127) << 23U;
      bits = sign | exponent32 | (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    const auto exponent32 = (exponent + (127U - 15U)) << 23U;
    bits = sign | exponent32 | (mantissa << 13U);
  }

  return std::bit_cast<float>(bits);
}

std::vector<float> float_section(
    std::span<const std::byte> bytes,
    const QpackSection& section,
    const char* label) {
  const auto count = product(section.shape);

  if (section.offset > bytes.size() ||
      section.size > bytes.size() - section.offset) {
    throw std::runtime_error(
        std::string("qpack expert: ") + label + " section exceeds cache slot");
  }

  const auto* raw =
      bytes.data() + static_cast<std::size_t>(section.offset);

  std::vector<float> out;
  out.resize(count);

  if (section.dtype == "F32") {
    const auto expected = count * sizeof(float);
    if (section.size != expected) {
      throw std::runtime_error(
          std::string("qpack expert: F32 ") + label +
          " byte size disagrees with shape");
    }

    for (std::size_t i = 0; i < count; ++i) {
      std::uint32_t bits{};
      std::memcpy(&bits, raw + i * sizeof(bits), sizeof(bits));
      out[i] = std::bit_cast<float>(bits);
    }
    return out;
  }

  if (section.dtype == "F16" || section.dtype == "BF16") {
    const auto expected = count * sizeof(std::uint16_t);
    if (section.size != expected) {
      throw std::runtime_error(
          std::string("qpack expert: ") + section.dtype + " " + label +
          " byte size disagrees with shape");
    }

    for (std::size_t i = 0; i < count; ++i) {
      std::uint16_t bits{};
      std::memcpy(&bits, raw + i * sizeof(bits), sizeof(bits));
      out[i] =
          section.dtype == "F16"
              ? f16_to_float(bits)
              : bf16_to_float(bits);
    }
    return out;
  }

  throw std::runtime_error(
      std::string("qpack expert: unsupported ") + label +
      " dtype " + section.dtype);
}

}  // namespace

QpackExpertQ4View bind_qpack_q4_projection(
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::string_view projection) {
  const auto& manifest = reader.manifest();
  const auto& layout = reader.layout();

  if (!manifest.quant_bits || !manifest.quant_group_size ||
      *manifest.quant_bits != 4U || *manifest.quant_group_size == 0U) {
    throw std::runtime_error(
        "qpack expert: projection requires affine Q4 manifest metadata");
  }

  const std::string prefix(projection);
  const auto* weight = layout.find_section(prefix + ".weight");
  const auto* scales = layout.find_section(prefix + ".scales");
  const auto* biases = layout.find_section(prefix + ".biases");

  if (weight == nullptr || scales == nullptr || biases == nullptr) {
    throw std::runtime_error(
        "qpack expert: weight/scales/biases section set is incomplete");
  }

  if (weight->dtype != "U32") {
    throw std::runtime_error("qpack expert: Q4 weight dtype must be U32");
  }

  if (scales->dtype != biases->dtype) {
    throw std::runtime_error(
        "qpack expert: scale/bias dtypes must match");
  }
  if (scales->dtype != "F32" &&
      scales->dtype != "F16" &&
      scales->dtype != "BF16") {
    throw std::runtime_error(
        "qpack expert: scale/bias dtype must be F32, F16, or BF16");
  }

  if (weight->shape.size() != 2U || scales->shape.size() != 2U ||
      biases->shape.size() != 2U) {
    throw std::runtime_error(
        "qpack expert: projection sections must be rank-2");
  }

  const auto out_dim = weight->shape[0];
  const auto packed_cols = weight->shape[1];
  if (out_dim == 0U || packed_cols == 0U) {
    throw std::runtime_error(
        "qpack expert: projection dimensions must be non-zero");
  }

  if (packed_cols > std::numeric_limits<std::size_t>::max() / 8U) {
    throw std::runtime_error(
        "qpack expert: packed projection dimension overflows");
  }
  const auto logical_cols = packed_cols * 8U;
  const auto group_size =
      static_cast<std::size_t>(*manifest.quant_group_size);
  if ((logical_cols % group_size) != 0U) {
    throw std::runtime_error(
        "qpack expert: logical input dimension not divisible by group size");
  }

  const auto groups_per_row = logical_cols / group_size;
  if (scales->shape[0] != out_dim || biases->shape[0] != out_dim ||
      scales->shape[1] != groups_per_row ||
      biases->shape[1] != groups_per_row) {
    throw std::runtime_error(
        "qpack expert: scale/bias shapes disagree with Q4 projection geometry");
  }

  const auto packed = typed_section<std::uint32_t>(
      entry.bytes, *weight, "weight");
  if (packed.size() != product(weight->shape)) {
    throw std::runtime_error(
        "qpack expert: weight byte size disagrees with declared shape");
  }

  auto scale_values = float_section(entry.bytes, *scales, "scales");
  auto bias_values = float_section(entry.bytes, *biases, "biases");

  return {
      .packed = packed,
      .scales = std::move(scale_values),
      .biases = std::move(bias_values),
      .out_dim = out_dim,
      .packed_cols = packed_cols,
      .group_size = group_size,
      .metadata_dtype = scales->dtype,
  };
}

VulkanQ4GemvResult run_vulkan_qpack_q4_projection(
    VulkanComputeContext& context,
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::string_view projection,
    std::span<const float> x) noexcept {
  try {
    const auto view =
        bind_qpack_q4_projection(reader, entry, projection);
    return run_vulkan_q4_gemv(
        context,
        view.packed,
        view.out_dim,
        view.packed_cols,
        view.scales,
        view.biases,
        view.group_size,
        x);
  } catch (const std::exception& e) {
    VulkanQ4GemvResult result;
    result.diagnostic =
        std::string("qpack expert Vulkan projection failed: ") + e.what();
    return result;
  } catch (...) {
    VulkanQ4GemvResult result;
    result.diagnostic =
        "qpack expert Vulkan projection encountered an unknown exception.";
    return result;
  }
}

}  // namespace orbi::streammoe
