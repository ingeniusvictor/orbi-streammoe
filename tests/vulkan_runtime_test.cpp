#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/backend/vulkan_runtime.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  try {
    const auto probe = probe_vulkan_runtime();

    require(
        !probe.diagnostic.empty(),
        "Vulkan probe must always return a diagnostic");

    if (!probe.loader_available) {
      require(
          !probe.instance_created,
          "instance cannot exist when Vulkan loader is unavailable");
      require(
          probe.devices.empty(),
          "devices cannot exist when Vulkan loader is unavailable");

      std::cout
          << "OSM-07 Vulkan bootstrap: PASS (loader unavailable on CI host)\n"
          << probe.diagnostic
          << "\n";
      return 0;
    }

    require(
        probe.loader_api_version != 0,
        "available Vulkan loader must report an API version");

    if (!probe.instance_created) {
      require(
          probe.devices.empty(),
          "devices cannot be reported when instance creation failed");

      std::cout
          << "OSM-07 Vulkan bootstrap: PASS (loader present; instance unavailable on CI host)\n"
          << probe.diagnostic
          << "\n";
      return 0;
    }

    for (const auto& device : probe.devices) {
      require(!device.name.empty(), "enumerated Vulkan device must have a name");
      require(device.api_version != 0, "enumerated Vulkan device must report API version");
      require(
          device.max_compute_workgroup_invocations > 0,
          "enumerated Vulkan device must report compute workgroup limits");
    }

    std::cout
        << "OSM-07 Vulkan bootstrap: PASS\n"
        << probe.diagnostic
        << "\n";

    for (const auto& device : probe.devices) {
      std::cout
          << "  device=" << device.name
          << " vendor=" << device.vendor_id
          << " device=" << device.device_id
          << " api=" << VK_VERSION_MAJOR(device.api_version)
          << "." << VK_VERSION_MINOR(device.api_version)
          << "." << VK_VERSION_PATCH(device.api_version)
          << " compute="
          << (device.compute_queue_family.has_value() ? "yes" : "no")
          << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-07 Vulkan bootstrap: FAIL: "
        << e.what()
        << "\n";
    return 1;
  }
}
