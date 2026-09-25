#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"
#include "orbi/streammoe/backend/vulkan_q4_gemv.hpp"
#include "orbi/streammoe/backend/vulkan_q4_weights.hpp"
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

std::uint32_t pack_word(std::size_t row, std::size_t word) {
  std::uint32_t packed = 0;
  for (std::uint32_t lane = 0; lane < 8; ++lane) {
    const auto q = static_cast<std::uint32_t>(
        (row * 11U + word * 5U + lane * 3U) & 0xFU);
    packed |= q << (lane * 4U);
  }
  return packed;
}

float max_abs_error(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  require(actual.size() == expected.size(), "vector size mismatch");
  float result = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    result = std::max(result, std::fabs(actual[i] - expected[i]));
  }
  return result;
}

}  // namespace

int main() {
  try {
    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);

    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no context: " + diagnostic);
      }
      std::cout
          << "OSM-19 preloaded Q4 weights: PASS (no compute device on host)\n";
      return 0;
    }

    constexpr std::size_t out_dim = 13;
    constexpr std::size_t packed_cols = 16;
    constexpr std::size_t in_dim = packed_cols * 8;
    constexpr std::size_t group_size = 64;
    constexpr std::size_t groups_per_row = in_dim / group_size;

    std::vector<std::uint32_t> packed(out_dim * packed_cols);
    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t word = 0; word < packed_cols; ++word) {
        packed[row * packed_cols + word] = pack_word(row, word);
      }
    }

    std::vector<float> scales(out_dim * groups_per_row);
    std::vector<float> biases(out_dim * groups_per_row);
    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t g = 0; g < groups_per_row; ++g) {
        const auto i = row * groups_per_row + g;
        scales[i] = 0.0625F * static_cast<float>((row + g) % 7U + 1U);
        biases[i] =
            0.03125F * static_cast<float>(
                static_cast<int>((row * 3U + g) % 9U) - 4);
      }
    }

    const auto dequantized = cpu::dequantize_affine_rows(
        packed,
        out_dim,
        packed_cols,
        scales,
        biases,
        cpu::AffineQuantSpec{.bits = 4, .group_size = group_size});

    auto weights = VulkanQ4ProjectionWeights::create(
        *context,
        packed,
        out_dim,
        packed_cols,
        scales,
        biases,
        group_size,
        &diagnostic);

    require(weights.has_value(), diagnostic);
    require(weights->valid(), "preloaded Q4 weight object must be valid");
    require(weights->out_dim() == out_dim, "out_dim mismatch");
    require(weights->in_dim() == in_dim, "in_dim mismatch");
    require(weights->packed_cols() == packed_cols, "packed_cols mismatch");
    require(weights->group_size() == group_size, "group_size mismatch");
    require(weights->native_device() == context->native_device(), "device mismatch");
    require(weights->native_packed_buffer() != 0U, "packed GPU buffer missing");

    auto input = VulkanFloatBuffer::create(*context, in_dim, &diagnostic);
    auto output = VulkanFloatBuffer::create(*context, out_dim, &diagnostic);
    require(input.has_value() && output.has_value(), diagnostic);

    float worst_error = 0.0F;

    // Run two different tokens through the same uploaded projection. This is
    // the OSM-19 contract: weight residency survives token execution.
    for (std::size_t token = 0; token < 2; ++token) {
      std::vector<float> x(in_dim);
      for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] =
            static_cast<float>(
                static_cast<int>((i * (token + 3U)) % 23U) - 11) *
            0.03125F;
      }

      const auto expected =
          cpu::matvec_row_major(dequantized, out_dim, in_dim, x);

      require(input->upload(x, &diagnostic), diagnostic);

      const auto dispatch = run_vulkan_q4_gemv_preloaded(
          *context, *weights, *input, *output);
      require(dispatch.executed, dispatch.diagnostic);

      const auto actual = output->download(&diagnostic);
      require(actual.has_value(), diagnostic);

      const auto error = max_abs_error(*actual, expected);
      worst_error = std::max(worst_error, error);
      require(error <= 5e-4F, "preloaded Q4 GEMV CPU parity failed");
    }

    std::cout
        << "OSM-19 preloaded Q4 weights: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  packed_bytes=" << weights->packed_size_bytes() << "\n"
        << "  reuse_count=2\n"
        << "  worst_max_abs_error=" << worst_error << "\n"
        << "  projection_uploaded_once=PASS\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-19 preloaded Q4 weights: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
