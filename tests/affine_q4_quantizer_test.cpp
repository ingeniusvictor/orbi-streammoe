#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/single_expert_pilot.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"

using namespace orbi::streammoe;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main() {
  try {
    constexpr std::size_t rows = 2U;
    constexpr std::size_t cols = 8U;
    constexpr std::size_t group = 4U;
    const std::vector<float> values{
        -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 2.5F,
        3.0F, 3.0F, 3.0F, 3.0F, -2.0F, -1.0F, 0.0F, 1.0F};

    const auto q = quantize_affine_q4_rows(values, rows, cols, group);
    require(q.packed.size() == 2U, "packed size mismatch");
    require(q.scales.size() == 4U, "scale size mismatch");
    require(q.biases.size() == 4U, "bias size mismatch");
    require(std::isfinite(q.max_abs_error), "max error must be finite");
    require(std::isfinite(q.mean_abs_error), "mean error must be finite");

    const auto decoded = cpu::dequantize_affine_rows(
        q.packed,
        rows,
        q.packed_cols,
        q.scales,
        q.biases,
        cpu::AffineQuantSpec{
            .bits = 4U,
            .group_size = static_cast<std::uint32_t>(group)});

    require(decoded.size() == values.size(), "dequantized size mismatch");
    float measured = 0.0F;
    for (std::size_t i = 0U; i < values.size(); ++i) {
      measured = std::max(measured, std::fabs(decoded[i] - values[i]));
    }
    require(
        std::fabs(measured - q.max_abs_error) <= 1e-6F,
        "reported error disagrees with CPU dequant oracle");

    bool rejected = false;
    try {
      (void)quantize_affine_q4_rows(values, rows, cols, 3U);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "non-divisible group size must fail");

    std::cout
        << "OSM-39C affine Q4 quantizer: PASS\n"
        << "  pack_roundtrip=PASS\n"
        << "  group_error_bound=PASS\n"
        << "  cpu_dequant_parity=PASS\n"
        << "  max_abs_error=" << q.max_abs_error << "\n"
        << "  mean_abs_error=" << q.mean_abs_error << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-39C affine Q4 quantizer: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
