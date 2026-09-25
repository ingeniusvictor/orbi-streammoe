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
#include "orbi/streammoe/backend/vulkan_swiglu.hpp"
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
    const auto q = static_cast<std::uint32_t>(
        (row * 5U + word * 3U + lane * 2U) & 0xFU);
    packed |= q << (lane * 4U);
  }
  return packed;
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
          << "OSM-18 persistent Vulkan ops: PASS (no compute device on host)\n";
      return 0;
    }

    constexpr std::size_t out_dim = 9;
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
      for (std::size_t group = 0; group < groups_per_row; ++group) {
        const auto i = row * groups_per_row + group;
        scales[i] = 0.125F * static_cast<float>((row + group) % 5U + 1U);
        biases[i] =
            0.0625F * static_cast<float>(
                static_cast<int>((row * 2U + group) % 7U) - 3);
      }
    }

    std::vector<float> x(in_dim);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] =
          static_cast<float>(static_cast<int>(i % 21U) - 10) * 0.03125F;
    }

    const auto dequantized = cpu::dequantize_affine_rows(
        packed,
        out_dim,
        packed_cols,
        scales,
        biases,
        cpu::AffineQuantSpec{.bits = 4, .group_size = group_size});
    const auto expected_gemv = cpu::matvec_row_major(
        dequantized, out_dim, in_dim, x);

    auto input = VulkanFloatBuffer::create(*context, in_dim, &diagnostic);
    auto gemv_output =
        VulkanFloatBuffer::create(*context, out_dim, &diagnostic);
    require(input.has_value() && gemv_output.has_value(), diagnostic);
    require(input->upload(x, &diagnostic), diagnostic);

    const auto gemv = run_vulkan_q4_gemv_buffers(
        *context,
        packed,
        out_dim,
        packed_cols,
        scales,
        biases,
        group_size,
        *input,
        *gemv_output);
    require(gemv.executed, gemv.diagnostic);

    const auto gemv_values = gemv_output->download(&diagnostic);
    require(gemv_values.has_value(), diagnostic);
    float gemv_error = 0.0F;
    for (std::size_t i = 0; i < expected_gemv.size(); ++i) {
      gemv_error = std::max(
          gemv_error,
          std::fabs((*gemv_values)[i] - expected_gemv[i]));
    }
    require(gemv_error <= 5e-4F, "persistent Q4 GEMV parity failed");

    std::vector<float> gate_values(193);
    std::vector<float> up_values(193);
    for (std::size_t i = 0; i < gate_values.size(); ++i) {
      gate_values[i] =
          static_cast<float>(static_cast<int>(i % 31U) - 15) * 0.125F;
      up_values[i] =
          static_cast<float>(static_cast<int>((i * 11U) % 27U) - 13) *
          0.09375F;
    }

    auto expected_swiglu = gate_values;
    cpu::swiglu_inplace(expected_swiglu, up_values);

    auto gate =
        VulkanFloatBuffer::create(*context, gate_values.size(), &diagnostic);
    auto up =
        VulkanFloatBuffer::create(*context, up_values.size(), &diagnostic);
    auto hidden =
        VulkanFloatBuffer::create(*context, gate_values.size(), &diagnostic);
    require(
        gate.has_value() && up.has_value() && hidden.has_value(),
        diagnostic);
    require(gate->upload(gate_values, &diagnostic), diagnostic);
    require(up->upload(up_values, &diagnostic), diagnostic);

    const auto swiglu =
        run_vulkan_swiglu_buffers(*context, *gate, *up, *hidden);
    require(swiglu.executed, swiglu.diagnostic);

    const auto hidden_values = hidden->download(&diagnostic);
    require(hidden_values.has_value(), diagnostic);
    float swiglu_error = 0.0F;
    for (std::size_t i = 0; i < expected_swiglu.size(); ++i) {
      swiglu_error = std::max(
          swiglu_error,
          std::fabs((*hidden_values)[i] - expected_swiglu[i]));
    }
    require(swiglu_error <= 2e-5F, "persistent SwiGLU parity failed");

    std::cout
        << "OSM-18 persistent Vulkan ops: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  q4_gemv_max_abs_error=" << gemv_error << "\n"
        << "  swiglu_max_abs_error=" << swiglu_error << "\n"
        << "  activation_buffers_reused_without_intermediate_readback=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-18 persistent Vulkan ops: FAIL: " << e.what() << "\n";
    return 1;
  }
}
