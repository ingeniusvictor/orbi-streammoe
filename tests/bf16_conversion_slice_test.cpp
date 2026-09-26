#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/bf16_slice.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_bytes(const fs::path& path, const std::vector<unsigned char>& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write source fixture");
  out.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
}

void write_text(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write manifest fixture");
  out << text;
}

std::string manifest_text() {
  return R"JSON({
    "schema_version": 1,
    "model": "fixture/qwen",
    "snapshot": "fixture-snapshot",
    "tensor_name": "model.norm.weight",
    "shard": "model-00001-of-00001.safetensors",
    "source_dtype": "BF16",
    "source_shape": [3],
    "source_absolute_offset": 128,
    "source_byte_size": 6,
    "fetched_bytes": 6,
    "source_file": "slice.bf16.bin",
    "source_fnv1a64": "5a24b89cd9cfa12d",
    "output_dtype": "F32",
    "output_byte_size": 12,
    "expected_f32_fnv1a64": "c598e74ad8b1c9b5"
  })JSON";
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm38d-bf16-slice";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    write_bytes(
        root / "slice.bf16.bin",
        {0x80U, 0x3fU, 0x00U, 0xc0U, 0x00U, 0x3fU});
    write_text(root / "conversion-slice.json", manifest_text());

    const auto manifest =
        inspect_bf16_conversion_slice(root / "conversion-slice.json");
    require(manifest.source_shape == std::vector<std::size_t>({3U}),
            "source shape mismatch");
    require(manifest.source_byte_size == 6U, "source size mismatch");
    require(manifest.output_byte_size == 12U, "output size mismatch");

    const auto result = convert_bf16_slice_to_f32(
        manifest,
        root,
        root / "slice.f32.bin");
    require(result.values.size() == 3U, "converted element count mismatch");
    require(result.values[0] == 1.0F, "BF16 1.0 decode mismatch");
    require(result.values[1] == -2.0F, "BF16 -2.0 decode mismatch");
    require(result.values[2] == 0.5F, "BF16 0.5 decode mismatch");
    require(
        result.source_fnv1a64 == 0x5a24b89cd9cfa12dULL,
        "source FNV mismatch");
    require(
        result.output_fnv1a64 == 0xc598e74ad8b1c9b5ULL,
        "output FNV mismatch");
    require(fs::file_size(root / "slice.f32.bin") == 12U,
            "F32 artifact size mismatch");

    write_bytes(
        root / "slice.bf16.bin",
        {0x80U, 0x3fU, 0x00U, 0xc0U, 0x01U, 0x3fU});
    bool corruption_rejected = false;
    try {
      (void)convert_bf16_slice_to_f32(
          manifest,
          root,
          root / "corrupt.f32.bin");
    } catch (const std::exception&) {
      corruption_rejected = true;
    }
    require(corruption_rejected, "source corruption must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-38D BF16 conversion slice: PASS\n"
        << "  known_bf16_decode=PASS\n"
        << "  deterministic_f32_bytes=PASS\n"
        << "  oracle_fnv=PASS\n"
        << "  corruption_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-38D BF16 conversion slice: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
