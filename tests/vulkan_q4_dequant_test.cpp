#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_q4_dequant.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool require_vulkan_compute() {
  const char* value = std::getenv("ORBI_REQUIRE_VULKAN_COMPUTE");
  return value != nullptr && std::string(value) != "0";
}

std::uint32_t pack_q4_word(
    std::uint32_t base,
    std::uint32_t step) {
  std::uint32_t word = 0;
  for (std::uint32_t lane = 0; lane < 8; ++lane) {
    const std::uint32_t q = (base + lane * step) & 0xFU;
    word |= q << (lane * 4U);
  }
  return word;
}

void invalid_shape_gate(VulkanComputeContext& context) {
  const std::vector<std::uint32_t> packed{0U};
  const std::vector<float> scales{1.0F};
  const std::vector<float> biases{0.0F};

  const auto result = run_vulkan_q4_dequant(
      context,
      packed,
      1,
      1,
      scales,
      biases,
      3);

  require(!result.executed, "non-divisible Q4 group size must be rejected");
  require(!result.diagnostic.empty(), "rejected Q4 shape needs a diagnostic");
}

}  // namespace

int main() {
  try {
    std::string context_diagnostic;
    auto context = VulkanComputeContext::create(&context_diagnostic);

    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no Vulkan compute context was created: " +
            context_diagnostic);
      }

      std::cout
          << "OSM-11 Vulkan Q4 dequant: PASS (no compute device on host)\n"
          << context_diagnostic << "\n";
      return 0;
    }

    invalid_shape_gate(*context);

    constexpr std::size_t rows = 2;
    constexpr std::size_t packed_cols = 16;
    constexpr std::size_t logical_cols = packed_cols * 8;
    constexpr std::size_t group_size = 64;

    std::vector<std::uint32_t> packed;
    packed.reserve(rows * packed_cols);

    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t word = 0; word < packed_cols; ++word) {
        packed.push_back(pack_q4_word(
            static_cast<std::uint32_t>(row * 3 + word),
            static_cast<std::uint32_t>(word % 5 + 1)));
      }
    }

    const std::vector<float> scales{
        0.25F, 1.5F,
        0.75F, 2.0F,
    };

    const std::vector<float> biases{
        -1.25F, 0.5F,
        3.0F, -4.0F,
    };

    const auto expected = cpu::dequantize_affine_rows(
        packed,
        rows,
        packed_cols,
        scales,
        biases,
        cpu::AffineQuantSpec{
            .bits = 4,
            .group_size = static_cast<std::uint32_t>(group_size),
        });

    const auto result = run_vulkan_q4_dequant(
        *context,
        packed,
        rows,
        packed_cols,
        scales,
        biases,
        group_size);

    require(result.executed, result.diagnostic);
    require(
        result.values.size() == rows * logical_cols,
        "Vulkan Q4 output size mismatch");
    require(
        result.values.size() == expected.size(),
        "Vulkan Q4 and CPU oracle sizes differ");

    float max_abs_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float error = std::fabs(result.values[i] - expected[i]);
      max_abs_error = std::max(max_abs_error, error);

      if (error > 1e-6F) {
        throw std::runtime_error(
            "Q4 dequant mismatch at index " + std::to_string(i) +
            ": gpu=" + std::to_string(result.values[i]) +
            " cpu=" + std::to_string(expected[i]) +
            " abs_error=" + std::to_string(error));
      }
    }

    require(
        expected.front() != expected.back(),
        "Q4 fixture must be non-trivial");

    std::cout
        << "OSM-11 Vulkan Q4 dequant: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  rows=" << rows
        << " logical_cols=" << logical_cols
        << " group_size=" << group_size << "\n"
        << "  max_abs_error=" << max_abs_error << "\n"
        << "  " << result.diagnostic << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-11 Vulkan Q4 dequant: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
