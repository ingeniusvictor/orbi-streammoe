#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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
  // OSM-13 uses float32 metadata fixtures. Native BF16/F16 loading is the
  // next compatibility step before a real production qpack pilot.
  if (scales->dtype != "F32" || biases->dtype != "F32") {
    throw std::runtime_error(
        "qpack expert: OSM-13 requires F32 scales/biases");
  }

  if (weight->shape.size() != 2U || scales->shape.size() != 2U ||
      biases->shape.size() != 2U) {
    throw std::runtime_error(
        "qpack expert: projection sections must be rank-2");
  }

  const auto out_dim = weight->shape[0];
  const auto packed_cols = weight->shape[1];
  if (out_dim == 0U || packed_cols == 0U) {
    throw std::runtime_error("qpack expert: projection dimensions must be non-zero");
  }

  const auto logical_cols = packed_cols * 8U;
  const auto group_size = static_cast<std::size_t>(*manifest.quant_group_size);
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
  const auto scale_values = typed_section<float>(
      entry.bytes, *scales, "scales");
  const auto bias_values = typed_section<float>(
      entry.bytes, *biases, "biases");

  if (packed.size() != product(weight->shape) ||
      scale_values.size() != product(scales->shape) ||
      bias_values.size() != product(biases->shape)) {
    throw std::runtime_error(
        "qpack expert: section byte sizes disagree with declared shapes");
  }

  return {
      .packed = packed,
      .scales = scale_values,
      .biases = bias_values,
      .out_dim = out_dim,
      .packed_cols = packed_cols,
      .group_size = group_size,
  };
}

VulkanQ4GemvResult run_vulkan_qpack_q4_projection(
    VulkanComputeContext& context,
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::string_view projection,
    std::span<const float> x) noexcept {
  try {
    const auto view = bind_qpack_q4_projection(reader, entry, projection);
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
