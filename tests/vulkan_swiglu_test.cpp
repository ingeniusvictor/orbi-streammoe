#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
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
          << "OSM-16 Vulkan SwiGLU: PASS (no compute device on host)\n"
          << diagnostic << "\n";
      return 0;
    }

    std::vector<float> gate(257);
    std::vector<float> up(257);
    for (std::size_t i = 0; i < gate.size(); ++i) {
      gate[i] =
          static_cast<float>(static_cast<int>(i % 29U) - 14) * 0.1875F;
      up[i] =
          static_cast<float>(static_cast<int>((i * 7U) % 23U) - 11) * 0.125F;
    }

    auto expected = gate;
    cpu::swiglu_inplace(expected, up);

    const auto result = run_vulkan_swiglu(*context, gate, up);
    require(result.executed, result.diagnostic);
    require(result.values.size() == expected.size(), "SwiGLU output size mismatch");

    float max_abs_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float error = std::fabs(result.values[i] - expected[i]);
      max_abs_error = std::max(max_abs_error, error);
      if (error > 2e-5F) {
        throw std::runtime_error(
            "SwiGLU mismatch at index " + std::to_string(i) +
            ": gpu=" + std::to_string(result.values[i]) +
            " cpu=" + std::to_string(expected[i]) +
            " error=" + std::to_string(error));
      }
    }

    const std::vector<float> bad_up{1.0F};
    const auto rejected = run_vulkan_swiglu(*context, gate, bad_up);
    require(!rejected.executed, "mismatched SwiGLU inputs must be rejected");

    std::cout
        << "OSM-16 Vulkan SwiGLU: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  count=" << gate.size() << "\n"
        << "  max_abs_error=" << max_abs_error << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-16 Vulkan SwiGLU: FAIL: " << e.what() << "\n";
    return 1;
  }
}
