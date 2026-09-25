#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"

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
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no context: " +
            context_diagnostic);
      }
      std::cout
          << "OSM-17 Vulkan float buffer: PASS (no compute device on host)\n";
      return 0;
    }

    constexpr std::size_t count = 513;
    std::string diagnostic;
    auto buffer = VulkanFloatBuffer::create(*context, count, &diagnostic);
    require(buffer.has_value(), diagnostic);
    require(buffer->valid(), "created float buffer must be valid");
    require(buffer->size() == count, "element count mismatch");
    require(
        buffer->size_bytes() == count * sizeof(float),
        "byte count mismatch");
    require(buffer->native_buffer() != 0U, "native buffer must be non-zero");

    std::vector<float> input(count);
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] =
          static_cast<float>(static_cast<int>(i % 37U) - 18) * 0.03125F;
    }

    require(buffer->upload(input, &diagnostic), diagnostic);

    const auto first = buffer->download(&diagnostic);
    require(first.has_value(), diagnostic);
    require(first->size() == input.size(), "download size mismatch");

    for (std::size_t i = 0; i < input.size(); ++i) {
      require(
          std::fabs((*first)[i] - input[i]) <= 0.0F,
          "upload/download round-trip changed float bits");
    }

    auto moved = std::move(*buffer);
    require(moved.valid(), "moved buffer must remain valid");
    require(moved.native_buffer() != 0U, "moved native handle must survive");

    for (auto& value : input) value = value * -1.5F + 0.25F;
    require(moved.upload(input, &diagnostic), diagnostic);

    const auto second = moved.download(&diagnostic);
    require(second.has_value(), diagnostic);
    for (std::size_t i = 0; i < input.size(); ++i) {
      require(
          (*second)[i] == input[i],
          "reused persistent buffer returned incorrect value");
    }

    const std::vector<float> wrong_size(1, 0.0F);
    require(
        !moved.upload(wrong_size, &diagnostic),
        "wrong-sized upload must be rejected");

    std::cout
        << "OSM-17 Vulkan float buffer: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  elements=" << moved.size() << "\n"
        << "  bytes=" << moved.size_bytes() << "\n"
        << "  reusable_upload_download=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-17 Vulkan float buffer: FAIL: " << e.what() << "\n";
    return 1;
  }
}
