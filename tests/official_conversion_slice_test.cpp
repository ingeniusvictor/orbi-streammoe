#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/bf16_slice.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_conversion_slice_tests <slice-dir>");
    }

    const auto root = std::filesystem::path(argv[1]);
    const auto manifest =
        inspect_bf16_conversion_slice(root / "conversion-slice.json");

    require(
        manifest.model == "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "official slice model mismatch");
    require(
        manifest.snapshot == "f5e99a3698d364cf77584543481b778afee26177",
        "official slice snapshot mismatch");
    require(
        manifest.tensor_name == "model.norm.weight",
        "official slice tensor mismatch");
    require(
        manifest.source_shape == std::vector<std::size_t>({2048U}),
        "official slice shape mismatch");
    require(manifest.source_byte_size == 4096U, "official BF16 size mismatch");
    require(manifest.fetched_bytes == 4096U, "official fetched byte count mismatch");
    require(manifest.output_byte_size == 8192U, "official F32 size mismatch");

    const auto output_path = root / "model.norm.weight.f32.bin";
    const auto result =
        convert_bf16_slice_to_f32(manifest, root, output_path);

    require(result.values.size() == 2048U, "official output element count mismatch");
    require(std::filesystem::file_size(output_path) == 8192U,
            "official output artifact size mismatch");

    double sum = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (const auto value : result.values) {
      require(std::isfinite(value), "official final norm contains non-finite value");
      sum += static_cast<double>(value);
      min_value = std::min(min_value, value);
      max_value = std::max(max_value, value);
    }
    require(max_value >= min_value, "official final norm range invalid");

    std::cout
        << "OSM-38D official BF16 conversion slice: PASS\n"
        << "  tensor=" << manifest.tensor_name << "\n"
        << "  source_bytes=" << manifest.source_byte_size << "\n"
        << "  output_bytes=" << manifest.output_byte_size << "\n"
        << "  elements=" << result.values.size() << "\n"
        << "  min=" << min_value << "\n"
        << "  max=" << max_value << "\n"
        << "  sum=" << sum << "\n"
        << "  source_fnv1a64=" << std::hex << std::setw(16)
        << std::setfill('0') << result.source_fnv1a64 << "\n"
        << "  output_fnv1a64=" << std::hex << std::setw(16)
        << std::setfill('0') << result.output_fnv1a64 << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-38D official BF16 conversion slice: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
