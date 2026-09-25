#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"
#include "orbi/streammoe/backend/vulkan_weighted_accumulate.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool require_vulkan_compute() {
  const char* value = std::getenv("ORBI_REQUIRE_VULKAN_COMPUTE");
  return value != nullptr && std::string(value) != "0";
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
          << "OSM-24B Vulkan weighted accumulate: PASS (no compute device on host)\n";
      return 0;
    }

    constexpr std::size_t count = 257;

    auto source = VulkanFloatBuffer::create(*context, count, &diagnostic);
    auto accumulator = VulkanFloatBuffer::create(*context, count, &diagnostic);
    require(source.has_value(), diagnostic);
    require(accumulator.has_value(), diagnostic);

    std::vector<float> zeros(count, 0.0F);
    require(
        accumulator->upload(zeros, &diagnostic),
        "accumulator zero upload failed: " + diagnostic);

    std::vector<float> source_a(count);
    std::vector<float> source_b(count);
    std::vector<float> expected(count, 0.0F);

    for (std::size_t i = 0; i < count; ++i) {
      source_a[i] =
          static_cast<float>(static_cast<int>(i % 19U) - 9) * 0.125F;
      source_b[i] =
          static_cast<float>(static_cast<int>((i * 7U) % 23U) - 11) * 0.0625F;
    }

    constexpr float weight_a = 0.25F;
    constexpr float weight_b = -0.5F;

    require(source->upload(source_a, &diagnostic), diagnostic);
    auto dispatch = run_vulkan_weighted_accumulate(
        *context, *source, weight_a, *accumulator);
    require(dispatch.executed, dispatch.diagnostic);

    for (std::size_t i = 0; i < count; ++i) {
      expected[i] += weight_a * source_a[i];
    }

    require(source->upload(source_b, &diagnostic), diagnostic);
    dispatch = run_vulkan_weighted_accumulate(
        *context, *source, weight_b, *accumulator);
    require(dispatch.executed, dispatch.diagnostic);

    for (std::size_t i = 0; i < count; ++i) {
      expected[i] += weight_b * source_b[i];
    }

    const auto actual = accumulator->download(&diagnostic);
    require(actual.has_value(), diagnostic);

    const float error = max_abs_error(*actual, expected);
    require(error <= 1e-6F, "weighted accumulate CPU parity failed");

    const auto source_handle = source->native_buffer();
    const auto accumulator_handle = accumulator->native_buffer();

    require(source_handle != 0U, "source native buffer missing");
    require(accumulator_handle != 0U, "accumulator native buffer missing");
    require(
        source_handle != accumulator_handle,
        "source and accumulator must be distinct for OSM-24B certification");

    std::cout
        << "OSM-24B Vulkan weighted accumulate: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  element_count=" << count << "\n"
        << "  dispatches=2\n"
        << "  final_readbacks=1\n"
        << "  max_abs_error=" << error << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-24B Vulkan weighted accumulate: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
