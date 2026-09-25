#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_rmsnorm.hpp"
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

void test_invalid_shape(VulkanComputeContext& context) {
  const std::vector<float> input{1.0F, 2.0F};
  const std::vector<float> bad_weight{1.0F};

  const auto result = run_vulkan_rms_norm(
      context,
      input,
      1,
      2,
      bad_weight,
      1e-6F);

  require(!result.executed, "invalid RMSNorm shape must be rejected");
  require(!result.diagnostic.empty(), "invalid RMSNorm must return diagnostic");
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
          << "OSM-10 Vulkan RMSNorm: PASS (no compute device on host)\n"
          << context_diagnostic << "\n";
      return 0;
    }

    test_invalid_shape(*context);

    constexpr std::size_t rows = 3;
    constexpr std::size_t dim = 8;
    constexpr float eps = 1e-6F;

    const std::vector<float> input{
         3.0F,  4.0F,  0.0F, -2.0F,  1.0F, -1.0F,  2.0F, -2.0F,
         0.5F, -0.25F, 1.5F,  2.0F, -3.0F,  0.75F, 4.0F, -1.25F,
        -2.5F,  1.0F,  0.25F, 3.5F,  2.25F, -0.5F,  1.75F, 0.125F,
    };

    const std::vector<float> weight{
        2.0F, 0.5F, 1.5F, 0.75F, 1.0F, 1.25F, 0.625F, 1.75F,
    };

    auto expected = input;
    cpu::rms_norm_inplace(expected, rows, dim, weight, eps);

    const auto result = run_vulkan_rms_norm(
        *context,
        input,
        rows,
        dim,
        weight,
        eps);

    require(result.executed, result.diagnostic);
    require(
        result.values.size() == expected.size(),
        "Vulkan RMSNorm output size mismatch");

    float max_abs_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float error = std::fabs(result.values[i] - expected[i]);
      max_abs_error = std::max(max_abs_error, error);
      if (error > 3e-5F) {
        throw std::runtime_error(
            "RMSNorm mismatch at index " + std::to_string(i) +
            ": gpu=" + std::to_string(result.values[i]) +
            " cpu=" + std::to_string(expected[i]) +
            " abs_error=" + std::to_string(error));
      }
    }

    std::cout
        << "OSM-10 Vulkan RMSNorm: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  rows=" << rows << " dim=" << dim << "\n"
        << "  max_abs_error=" << max_abs_error << "\n"
        << "  " << result.diagnostic << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-10 Vulkan RMSNorm: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
