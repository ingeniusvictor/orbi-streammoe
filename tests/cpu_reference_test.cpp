#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/cpu/reference_ops.hpp"

using namespace orbi::streammoe::cpu;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(float actual, float expected, float tolerance, const std::string& message) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(
        message + ": actual=" + std::to_string(actual) +
        " expected=" + std::to_string(expected));
  }
}

std::uint32_t pack_q4(const std::vector<std::uint32_t>& values) {
  require(values.size() == 8, "q4 pack requires 8 values");
  std::uint32_t word = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    require(values[i] < 16, "q4 lane out of range");
    word |= values[i] << (4U * i);
  }
  return word;
}

std::uint32_t pack_q8(const std::vector<std::uint32_t>& values) {
  require(values.size() == 4, "q8 pack requires 4 values");
  std::uint32_t word = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    require(values[i] < 256, "q8 lane out of range");
    word |= values[i] << (8U * i);
  }
  return word;
}

void test_q4_affine() {
  const std::vector<std::uint32_t> packed{
      pack_q4({0, 1, 2, 3, 4, 5, 6, 7}),
  };
  const std::vector<float> scales{2.0F};
  const std::vector<float> biases{-1.0F};

  const auto out = dequantize_affine_rows(
      packed,
      1,
      1,
      scales,
      biases,
      {.bits = 4, .group_size = 8});

  require(out.size() == 8, "q4 output width");
  for (std::size_t i = 0; i < out.size(); ++i) {
    require_close(
        out[i],
        2.0F * static_cast<float>(i) - 1.0F,
        1e-6F,
        "q4 affine value");
  }
}

void test_q8_affine() {
  const std::vector<std::uint32_t> packed{
      pack_q8({2, 4, 6, 8}),
  };
  const std::vector<float> scales{0.5F};
  const std::vector<float> biases{1.0F};

  const auto out = dequantize_affine_rows(
      packed,
      1,
      1,
      scales,
      biases,
      {.bits = 8, .group_size = 4});

  const std::vector<float> expected{2.0F, 3.0F, 4.0F, 5.0F};
  require(out.size() == expected.size(), "q8 output width");
  for (std::size_t i = 0; i < out.size(); ++i) {
    require_close(out[i], expected[i], 1e-6F, "q8 affine value");
  }
}

void test_matvec_and_rmsnorm() {
  const std::vector<float> weight{
      1.0F, 2.0F,
      -1.0F, 3.0F,
  };
  const std::vector<float> x{4.0F, 5.0F};
  const auto y = matvec_row_major(weight, 2, 2, x);

  require_close(y[0], 14.0F, 1e-6F, "matvec row 0");
  require_close(y[1], 11.0F, 1e-6F, "matvec row 1");

  std::vector<float> values{3.0F, 4.0F};
  const std::vector<float> norm_weight{2.0F, 0.5F};
  rms_norm_inplace(values, 1, 2, norm_weight, 0.0F);

  const float base = std::sqrt((9.0F + 16.0F) / 2.0F);
  require_close(values[0], (3.0F / base) * 2.0F, 1e-6F, "rmsnorm col 0");
  require_close(values[1], (4.0F / base) * 0.5F, 1e-6F, "rmsnorm col 1");
}

void test_router_topk() {
  const std::vector<float> logits{0.0F, 1.0F, 2.0F, 3.0F};

  const auto raw = route_top_k(logits, 2, false);
  require(raw.size() == 2, "router raw size");
  require(raw[0].expert == 3, "router best expert");
  require(raw[1].expert == 2, "router second expert");

  std::vector<float> probs = logits;
  softmax_inplace(probs);
  require_close(raw[0].probability, probs[3], 1e-6F, "router raw prob 0");
  require_close(raw[1].probability, probs[2], 1e-6F, "router raw prob 1");

  const auto normalized = route_top_k(logits, 2, true);
  require(normalized[0].expert == 3, "router normalized best expert");
  require(normalized[1].expert == 2, "router normalized second expert");
  require_close(
      normalized[0].probability + normalized[1].probability,
      1.0F,
      1e-6F,
      "router selected normalization");
}

void test_swiglu() {
  std::vector<float> gate{-1.0F, 0.0F, 1.0F};
  const std::vector<float> up{2.0F, 3.0F, 4.0F};

  swiglu_inplace(gate, up);

  require_close(gate[0], silu(-1.0F) * 2.0F, 1e-6F, "swiglu 0");
  require_close(gate[1], 0.0F, 1e-6F, "swiglu 1");
  require_close(gate[2], silu(1.0F) * 4.0F, 1e-6F, "swiglu 2");
}

void test_invalid_quant_rejected() {
  bool rejected = false;
  try {
    const std::vector<std::uint32_t> packed{0};
    const std::vector<float> scales{1.0F};
    const std::vector<float> biases{0.0F};
    (void)dequantize_affine_rows(
        packed,
        1,
        1,
        scales,
        biases,
        {.bits = 3, .group_size = 8});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "unsupported affine bit width was accepted");
}

}  // namespace

int main() {
  try {
    test_q4_affine();
    test_q8_affine();
    test_matvec_and_rmsnorm();
    test_router_topk();
    test_swiglu();
    test_invalid_quant_rejected();

    std::cout << "OSM-03 CPU reference primitives: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "OSM-03 CPU reference primitives: FAIL: " << e.what() << "\n";
    return 1;
  }
}
