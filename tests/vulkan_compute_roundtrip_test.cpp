#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_compute_roundtrip.hpp"

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
    std::string context_diagnostic;
    auto context = VulkanComputeContext::create(&context_diagnostic);

    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no Vulkan compute context was created: " +
            context_diagnostic);
      }

      std::cout
          << "OSM-09 Vulkan compute round-trip: PASS (no compute device on host)\n"
          << context_diagnostic << "\n";
      return 0;
    }

    constexpr std::uint32_t initial = 0xa5a5a5a5U;
    auto result = run_vulkan_compute_roundtrip(*context, initial);

    require(result.executed, result.diagnostic);
    require(
        result.initial_value == initial,
        "round-trip result must preserve the requested initial value");
    require(
        result.output_value == kVulkanRoundTripSentinel,
        "GPU readback did not match OSM-09 sentinel");
    require(
        result.output_value != initial,
        "shader must visibly modify the storage buffer");

    std::cout
        << "OSM-09 Vulkan compute round-trip: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  initial=0x" << std::hex << result.initial_value << "\n"
        << "  output=0x" << result.output_value << std::dec << "\n"
        << "  " << result.diagnostic << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-09 Vulkan compute round-trip: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
