#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/storage/qpack_expert_storage.hpp"

namespace fs = std::filesystem;
using orbi::streammoe::ExpertReadRequest;
using orbi::streammoe::QpackExpertStorage;
using orbi::streammoe::QpackReader;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create " + path.string());
  out << text;
}

void write_layer(const fs::path& path, std::uint8_t layer_seed) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create " + path.string());

  constexpr std::size_t kExperts = 3;
  constexpr std::size_t kStride = 16;
  for (std::size_t expert = 0; expert < kExperts; ++expert) {
    for (std::size_t i = 0; i < kStride; ++i) {
      const auto value = static_cast<std::uint8_t>(
          layer_seed + expert * kStride + i);
      out.put(static_cast<char>(value));
    }
  }
}

fs::path make_container(
    const fs::path& root,
    std::string magic = "QPACK",
    std::uint32_t quant_group_size = 8) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  // For the synthetic quantization geometry below:
  // weight last dim 2 packed uint32 words at int4 => logical inner dim 16.
  // scales last dim 2 groups at group size 8 => logical inner dim 16.
  const std::string layout =
      "{\n"
      "  \"expertCount\": 3,\n"
      "  \"layerCount\": 2,\n"
      "  \"expertStride\": 16,\n"
      "  \"sections\": [\n"
      "    {\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[2,2],\"offset\":0,\"size\":8},\n"
      "    {\"name\":\"gate_proj.scales\",\"dtype\":\"BF16\",\"shape\":[2],\"offset\":8,\"size\":4}\n"
      "  ],\n"
      "  \"linearLayers\": [true, false]\n"
      "}\n";

  write_text(root / "packed_experts" / "layout.json", layout);
  write_layer(root / "packed_experts" / "layer_00.bin", 0);
  write_layer(root / "packed_experts" / "layer_01.bin", 64);

  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");

  const std::string manifest =
      "{\n"
      "  \"magic\": \"" + magic + "\",\n"
      "  \"version\": 1,\n"
      "  \"modelName\": \"qwen3_next\",\n"
      "  \"sourceCheckpoint\": \"synthetic-fixture\",\n"
      "  \"quantBits\": 4,\n"
      "  \"quantGroupSize\": " + std::to_string(quant_group_size) + ",\n"
      "  \"files\": {\n"
      "    \"packed_experts/layout.json\": " + std::to_string(layout_size) + ",\n"
      "    \"packed_experts/layer_00.bin\": 48,\n"
      "    \"packed_experts/layer_01.bin\": 48\n"
      "  }\n"
      "}\n";

  write_text(root / "manifest.json", manifest);
  return root;
}

void test_parse_and_read(const fs::path& root) {
  const QpackReader reader(make_container(root));

  require(reader.manifest().magic == "QPACK", "magic mismatch");
  require(reader.manifest().version == 1, "version mismatch");
  require(reader.manifest().quant_bits == 4, "quant bits mismatch");
  require(reader.manifest().quant_group_size == 8, "quant group mismatch");

  require(reader.layout().expert_count == 3, "expert count mismatch");
  require(reader.layout().layer_count == 2, "layer count mismatch");
  require(reader.layout().expert_stride == 16, "stride mismatch");
  require(reader.layout().linear_layers.size() == 2, "linear layer count mismatch");
  require(reader.layout().linear_layers[0], "layer 0 should be linear");
  require(!reader.layout().linear_layers[1], "layer 1 should be full attention");

  const auto* section = reader.find_section("gate_proj.weight");
  require(section != nullptr, "expected section missing");
  require(section->offset == 0 && section->size == 8, "section metadata mismatch");

  const auto bytes = reader.read_expert(1, 2);
  require(bytes.size() == 16, "expert byte count mismatch");
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto expected = static_cast<std::uint8_t>(64 + 2 * 16 + i);
    const auto actual = std::to_integer<std::uint8_t>(bytes[i]);
    require(actual == expected, "expert bytes differ from fixed-stride source");
  }
}

void test_rejects_bad_magic(const fs::path& root) {
  make_container(root, "NOT_QPACK");
  bool rejected = false;
  try {
    const QpackReader ignored(root);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "bad qpack magic was accepted");
}

void test_rejects_short_layer(const fs::path& root) {
  make_container(root);
  fs::resize_file(root / "packed_experts" / "layer_01.bin", 47);

  bool rejected = false;
  try {
    const QpackReader ignored(root);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "short expert layer was accepted");
}

void test_rejects_quant_shape_mismatch(const fs::path& root) {
  make_container(root, "QPACK", 64);

  bool rejected = false;
  try {
    const QpackReader ignored(root);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "quantization/packed-shape mismatch was accepted");
}


void test_storage_adapter(const fs::path& root) {
  make_container(root);
  QpackExpertStorage storage(root);

  std::vector<std::byte> first(storage.expert_stride_bytes());
  std::vector<std::byte> second(storage.expert_stride_bytes());
  std::vector<ExpertReadRequest> requests{
      {
          .id = {.layer = 0, .expert = 1},
          .destination = std::span<std::byte>(first.data(), first.size()),
      },
      {
          .id = {.layer = 1, .expert = 2},
          .destination = std::span<std::byte>(second.data(), second.size()),
      },
  };

  storage.read_experts(requests);

  for (std::size_t i = 0; i < first.size(); ++i) {
    const auto expected = static_cast<std::uint8_t>(16 + i);
    require(
        std::to_integer<std::uint8_t>(first[i]) == expected,
        "qpack storage adapter first expert mismatch");
  }
  for (std::size_t i = 0; i < second.size(); ++i) {
    const auto expected = static_cast<std::uint8_t>(64 + 2 * 16 + i);
    require(
        std::to_integer<std::uint8_t>(second[i]) == expected,
        "qpack storage adapter second expert mismatch");
  }
}

void test_bounds(const fs::path& root) {
  const QpackReader reader(make_container(root));

  bool bad_layer = false;
  try {
    (void)reader.read_expert(2, 0);
  } catch (const std::out_of_range&) {
    bad_layer = true;
  }
  require(bad_layer, "out-of-range layer was accepted");

  bool bad_expert = false;
  try {
    (void)reader.read_expert(0, 3);
  } catch (const std::out_of_range&) {
    bad_expert = true;
  }
  require(bad_expert, "out-of-range expert was accepted");

  std::array<std::byte, 8> too_small{};
  bool bad_buffer = false;
  try {
    reader.read_expert(0, 0, too_small);
  } catch (const std::invalid_argument&) {
    bad_buffer = true;
  }
  require(bad_buffer, "undersized destination was accepted");
}

}  // namespace

int main() {
  const auto base =
      fs::temp_directory_path() / "orbi-streammoe-osm02-qpack-tests";

  try {
    test_parse_and_read(base / "parse");
    test_rejects_bad_magic(base / "bad-magic");
    test_rejects_short_layer(base / "short-layer");
    test_rejects_quant_shape_mismatch(base / "bad-quant");
    test_storage_adapter(base / "storage-adapter");
    test_bounds(base / "bounds");
    fs::remove_all(base);
    std::cout << "OSM-02 qpack compatibility: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(base);
    std::cerr << "OSM-02 qpack compatibility: FAIL: " << e.what() << "\n";
    return 1;
  }
}
