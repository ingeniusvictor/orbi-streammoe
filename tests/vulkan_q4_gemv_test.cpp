#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_q4_gemv.hpp"
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

std::uint32_t pack_word(
    std::size_t row,
    std::size_t word) {
  std::uint32_t packed = 0;
  for (std::uint32_t lane = 0; lane < 8; ++lane) {
    const std::uint32_t q = static_cast<std::uint32_t>(
        (row * 7 + word * 3 + lane * (word % 5 + 1)) & 0xFU);
    packed |= q << (lane * 4U);
  }
  return packed;
}

void invalid_shape_gate(VulkanComputeContext& context) {
  const std::vector<std::uint32_t> packed{0U};
  const std::vector<float> scales{1.0F};
  const std::vector<float> biases{0.0F};
  const std::vector<float> x(8, 1.0F);

  const auto result = run_vulkan_q4_gemv(
      context,
      packed,
      1,
      1,
      scales,
      biases,
      3,
      x);

  require(!result.executed, "non-divisible Q4 GEMV group size must be rejected");
  require(
      !result.diagnostic.empty(),
      "rejected Q4 GEMV shape needs a diagnostic");
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
          << "OSM-12 Vulkan Q4 GEMV: PASS (no compute device on host)\n"
          << context_diagnostic << "\n";
      return 0;
    }

    invalid_shape_gate(*context);

    constexpr std::size_t out_dim = 7;
    constexpr std::size_t packed_cols = 16;
    constexpr std::size_t in_dim = packed_cols * 8;
    constexpr std::size_t group_size = 64;
    constexpr std::size_t groups_per_row = in_dim / group_size;

    std::vector<std::uint32_t> packed;
    packed.reserve(out_dim * packed_cols);
    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t word = 0; word < packed_cols; ++word) {
        packed.push_back(pack_word(row, word));
      }
    }

    std::vector<float> scales(out_dim * groups_per_row);
    std::vector<float> biases(out_dim * groups_per_row);
    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t g = 0; g < groups_per_row; ++g) {
        const auto i = row * groups_per_row + g;
        scales[i] = 0.125F + 0.075F * static_cast<float>(row + 2 * g + 1);
        biases[i] =
            (g == 0 ? -0.45F : 0.30F) +
            0.035F * static_cast<float>(row);
      }
    }

    std::vector<float> x(in_dim);
    for (std::size_t i = 0; i < x.size(); ++i) {
      const int signed_bucket = static_cast<int>(i % 17) - 8;
      x[i] =
          static_cast<float>(signed_bucket) * 0.0625F +
          static_cast<float>((i * 7) % 5) * 0.0175F;
    }

    const auto dequantized = cpu::dequantize_affine_rows(
        packed,
        out_dim,
        packed_cols,
        scales,
        biases,
        cpu::AffineQuantSpec{
            .bits = 4,
            .group_size = static_cast<std::uint32_t>(group_size),
        });

    const auto expected = cpu::matvec_row_major(
        dequantized,
        out_dim,
        in_dim,
        x);

    const auto result = run_vulkan_q4_gemv(
        *context,
        packed,
        out_dim,
        packed_cols,
        scales,
        biases,
        group_size,
        x);

    require(result.executed, result.diagnostic);
    require(
        result.values.size() == expected.size(),
        "Vulkan Q4 GEMV output size mismatch");

    float max_abs_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float error = std::fabs(result.values[i] - expected[i]);
      max_abs_error = std::max(max_abs_error, error);

      if (error > 5e-4F) {
        throw std::runtime_error(
            "Q4 GEMV mismatch at row " + std::to_string(i) +
            ": gpu=" + std::to_string(result.values[i]) +
            " cpu=" + std::to_string(expected[i]) +
            " abs_error=" + std::to_string(error));
      }
    }

    require(
        std::fabs(expected.front() - expected.back()) > 1e-3F,
        "Q4 GEMV fixture must produce non-trivial rows");

    std::cout
        << "OSM-12 Vulkan Q4 GEMV: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  out_dim=" << out_dim
        << " in_dim=" << in_dim
        << " group_size=" << group_size << "\n"
        << "  max_abs_error=" << max_abs_error << "\n"
        << "  " << result.diagnostic << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-12 Vulkan Q4 GEMV: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
