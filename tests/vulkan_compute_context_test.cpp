#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_runtime.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    const auto probe = probe_vulkan_runtime();
    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);

    require(!diagnostic.empty(), "context creation must return a diagnostic");

    const bool has_compute_device = [&probe] {
      for (const auto& device : probe.devices) {
        if (device.compute_queue_family.has_value()) return true;
      }
      return false;
    }();

    if (!probe.loader_available ||
        !probe.instance_created ||
        !has_compute_device) {
      require(
          !context.has_value(),
          "context should not be created when the runtime has no compute device");
      std::cout
          << "OSM-08 Vulkan compute context: PASS (no compute device on host)\n"
          << diagnostic << "\n";
      return 0;
    }

    require(context.has_value(), "compute-capable Vulkan host must create context");
    require(context->valid(), "created Vulkan context must be valid");
    require(!context->info().device_name.empty(), "device name must be recorded");
    require(context->native_instance() != 0U, "instance handle must be non-zero");
    require(context->native_physical_device() != 0U, "physical device handle must be non-zero");
    require(context->native_device() != 0U, "logical device handle must be non-zero");
    require(context->native_queue() != 0U, "queue handle must be non-zero");
    require(context->native_command_pool() != 0U, "command pool handle must be non-zero");

    auto moved = std::move(*context);
    require(moved.valid(), "moved Vulkan context must remain valid");

    std::cout
        << "OSM-08 Vulkan compute context: PASS\n"
        << diagnostic << "\n"
        << "  device=" << moved.info().device_name
        << " queue_family=" << moved.info().queue_family_index
        << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-08 Vulkan compute context: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
