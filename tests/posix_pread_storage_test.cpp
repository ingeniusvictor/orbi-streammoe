#if !defined(_WIN32)

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/storage/posix_pread_expert_storage.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("unable to create " + path.string());
  }
  out << text;
}

void write_layer(
    const fs::path& path,
    std::uint8_t layer_seed,
    std::size_t expert_count,
    std::size_t stride) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("unable to create " + path.string());
  }

  for (std::size_t expert = 0; expert < expert_count; ++expert) {
    for (std::size_t i = 0; i < stride; ++i) {
      const auto value = static_cast<std::uint8_t>(
          layer_seed + expert * stride + i);
      out.put(static_cast<char>(value));
    }
  }
}

fs::path make_container(const fs::path& root) {
  constexpr std::size_t kExperts = 4;
  constexpr std::size_t kLayers = 2;
  constexpr std::size_t kStride = 16;

  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  const std::string layout =
      "{\n"
      "  \"expertCount\": 4,\n"
      "  \"layerCount\": 2,\n"
      "  \"expertStride\": 16,\n"
      "  \"sections\": [\n"
      "    {\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[2,2],\"offset\":0,\"size\":8},\n"
      "    {\"name\":\"gate_proj.scales\",\"dtype\":\"BF16\",\"shape\":[2],\"offset\":8,\"size\":4}\n"
      "  ],\n"
      "  \"linearLayers\": [true, false]\n"
      "}\n";

  write_text(root / "packed_experts" / "layout.json", layout);
  write_layer(root / "packed_experts" / "layer_00.bin", 0, kExperts, kStride);
  write_layer(root / "packed_experts" / "layer_01.bin", 64, kExperts, kStride);

  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");

  const std::string manifest =
      "{\n"
      "  \"magic\": \"QPACK\",\n"
      "  \"version\": 1,\n"
      "  \"modelName\": \"qwen3_next\",\n"
      "  \"sourceCheckpoint\": \"posix-pread-fixture\",\n"
      "  \"quantBits\": 4,\n"
      "  \"quantGroupSize\": 8,\n"
      "  \"files\": {\n"
      "    \"packed_experts/layout.json\": " + std::to_string(layout_size) + ",\n"
      "    \"packed_experts/layer_00.bin\": 64,\n"
      "    \"packed_experts/layer_01.bin\": 64\n"
      "  }\n"
      "}\n";

  write_text(root / "manifest.json", manifest);
  return root;
}

void check_bytes(
    std::span<const std::byte> bytes,
    std::uint8_t layer_seed,
    std::uint32_t expert) {
  require(bytes.size() == 16, "unexpected expert stride");

  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const auto expected = static_cast<std::uint8_t>(
        layer_seed + expert * 16U + static_cast<std::uint32_t>(i));
    const auto actual = std::to_integer<std::uint8_t>(bytes[i]);
    require(actual == expected, "pread expert bytes mismatch");
  }
}

void test_concurrent_preads(const fs::path& root) {
  PosixPreadExpertStorage storage(make_container(root));

  std::vector<std::byte> a(16);
  std::vector<std::byte> b(16);
  std::vector<std::byte> c(16);

  std::vector<ExpertReadRequest> requests{
      {
          .id = {.layer = 0, .expert = 1},
          .destination = std::span<std::byte>(a.data(), a.size()),
      },
      {
          .id = {.layer = 0, .expert = 3},
          .destination = std::span<std::byte>(b.data(), b.size()),
      },
      {
          .id = {.layer = 1, .expert = 2},
          .destination = std::span<std::byte>(c.data(), c.size()),
      },
  };

  storage.read_experts(requests);

  check_bytes(a, 0, 1);
  check_bytes(b, 0, 3);
  check_bytes(c, 64, 2);
}

void test_cache_integration(const fs::path& root) {
  PosixPreadExpertStorage storage(root);
  ExpertCache cache(storage, 16 * 3);

  const std::vector<std::uint32_t> experts{0, 2, 3};
  const auto first = cache.fetch(1, experts);

  require(first.size() == 3, "cache integration result size");
  check_bytes(first[0].bytes, 64, 0);
  check_bytes(first[1].bytes, 64, 2);
  check_bytes(first[2].bytes, 64, 3);

  auto stats = cache.stats();
  require(stats.misses == 3 && stats.hits == 0, "first cache integration stats");

  const std::vector<std::uint32_t> hits{3, 0};
  const auto second = cache.fetch(1, hits);
  check_bytes(second[0].bytes, 64, 3);
  check_bytes(second[1].bytes, 64, 0);

  stats = cache.stats();
  require(stats.hits == 2 && stats.misses == 3, "repeat cache integration stats");
}

void test_short_read_fails(const fs::path& root) {
  PosixPreadExpertStorage storage(root);
  fs::resize_file(root / "packed_experts" / "layer_00.bin", 16 * 2 + 8);

  std::vector<std::byte> bytes(16);
  std::vector<ExpertReadRequest> requests{
      {
          .id = {.layer = 0, .expert = 2},
          .destination = std::span<std::byte>(bytes.data(), bytes.size()),
      },
  };

  bool failed = false;
  try {
    storage.read_experts(requests);
  } catch (const std::exception&) {
    failed = true;
  }

  require(failed, "short pread was accepted");
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm06-posix-pread";

  try {
    test_concurrent_preads(root);
    test_cache_integration(root);
    test_short_read_fails(root);
    fs::remove_all(root);

    std::cout << "OSM-06 POSIX pread expert I/O: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-06 POSIX pread expert I/O: FAIL: "
        << e.what()
        << "\n";
    return 1;
  }
}

#endif  // !_WIN32
