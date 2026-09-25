#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"
#include "orbi/streammoe/storage/qpack_expert_storage.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool require_vulkan_compute() {
  const char* value = std::getenv("ORBI_REQUIRE_VULKAN_COMPUTE");
  return value != nullptr && std::string(value) != "0";
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create " + path.string());
  out << text;
}

std::uint16_t bf16(float value) {
  return static_cast<std::uint16_t>(
      std::bit_cast<std::uint32_t>(value) >> 16U);
}

std::uint32_t pack_word(
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t row,
    std::size_t word) {
  std::uint32_t packed = 0;
  for (std::uint32_t lane = 0; lane < 8; ++lane) {
    const auto q = static_cast<std::uint32_t>(
        (projection_seed * 3U + expert * 5U + row * 7U +
         word * 2U + lane) & 0xFU);
    packed |= q << (lane * 4U);
  }
  return packed;
}

float scale_value(
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t row,
    std::size_t group) {
  const int bucket = static_cast<int>(
      (projection_seed + expert + row + group) % 6U) + 1;
  return static_cast<float>(bucket) * 0.125F;
}

float bias_value(
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t row,
    std::size_t group) {
  const int bucket = static_cast<int>(
      (projection_seed * 2U + expert + row + group) % 7U) - 3;
  return static_cast<float>(bucket) * 0.125F;
}

void write_u32(
    std::vector<std::byte>& blob,
    std::size_t byte_offset,
    std::uint32_t value) {
  std::memcpy(blob.data() + byte_offset, &value, sizeof(value));
}

void write_bf16(
    std::vector<std::byte>& blob,
    std::size_t byte_offset,
    float value) {
  const auto bits = bf16(value);
  std::memcpy(blob.data() + byte_offset, &bits, sizeof(bits));
}

void fill_projection(
    std::vector<std::byte>& blob,
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t out_dim,
    std::size_t packed_cols,
    std::size_t groups_per_row,
    std::size_t weight_offset,
    std::size_t scales_offset,
    std::size_t biases_offset) {
  for (std::size_t row = 0; row < out_dim; ++row) {
    for (std::size_t word = 0; word < packed_cols; ++word) {
      write_u32(
          blob,
          weight_offset +
              (row * packed_cols + word) * sizeof(std::uint32_t),
          pack_word(projection_seed, expert, row, word));
    }

    for (std::size_t group = 0; group < groups_per_row; ++group) {
      const auto index = row * groups_per_row + group;
      write_bf16(
          blob,
          scales_offset + index * sizeof(std::uint16_t),
          scale_value(projection_seed, expert, row, group));
      write_bf16(
          blob,
          biases_offset + index * sizeof(std::uint16_t),
          bias_value(projection_seed, expert, row, group));
    }
  }
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  constexpr std::size_t gate_weight = 0;
  constexpr std::size_t gate_scales = 4096;
  constexpr std::size_t gate_biases = 4352;
  constexpr std::size_t up_weight = 4608;
  constexpr std::size_t up_scales = 8704;
  constexpr std::size_t up_biases = 8960;
  constexpr std::size_t down_weight = 9216;
  constexpr std::size_t down_scales = 13312;
  constexpr std::size_t down_biases = 13568;
  constexpr std::size_t stride = 16384;

  const std::string layout =
      "{\n"
      "  \"expertCount\": 2,\n"
      "  \"layerCount\": 1,\n"
      "  \"expertStride\": 16384,\n"
      "  \"sections\": [\n"
      "    {\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[64,16],\"offset\":0,\"size\":4096},\n"
      "    {\"name\":\"gate_proj.scales\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":4096,\"size\":256},\n"
      "    {\"name\":\"gate_proj.biases\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":4352,\"size\":256},\n"
      "    {\"name\":\"up_proj.weight\",\"dtype\":\"U32\",\"shape\":[64,16],\"offset\":4608,\"size\":4096},\n"
      "    {\"name\":\"up_proj.scales\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":8704,\"size\":256},\n"
      "    {\"name\":\"up_proj.biases\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":8960,\"size\":256},\n"
      "    {\"name\":\"down_proj.weight\",\"dtype\":\"U32\",\"shape\":[128,8],\"offset\":9216,\"size\":4096},\n"
      "    {\"name\":\"down_proj.scales\",\"dtype\":\"BF16\",\"shape\":[128,1],\"offset\":13312,\"size\":256},\n"
      "    {\"name\":\"down_proj.biases\",\"dtype\":\"BF16\",\"shape\":[128,1],\"offset\":13568,\"size\":256}\n"
      "  ],\n"
      "  \"linearLayers\": [true]\n"
      "}\n";

  write_text(root / "packed_experts" / "layout.json", layout);

  std::ofstream layer(
      root / "packed_experts" / "layer_00.bin",
      std::ios::binary);
  if (!layer) throw std::runtime_error("unable to create expert layer");

  for (std::size_t expert = 0; expert < 2; ++expert) {
    std::vector<std::byte> blob(stride);

    fill_projection(
        blob, 1, expert, 64, 16, 2,
        gate_weight, gate_scales, gate_biases);
    fill_projection(
        blob, 2, expert, 64, 16, 2,
        up_weight, up_scales, up_biases);
    fill_projection(
        blob, 3, expert, 128, 8, 1,
        down_weight, down_scales, down_biases);

    layer.write(
        reinterpret_cast<const char*>(blob.data()),
        static_cast<std::streamsize>(blob.size()));
  }
  layer.close();

  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");
  const auto layer_size =
      fs::file_size(root / "packed_experts" / "layer_00.bin");

  const std::string manifest =
      "{\n"
      "  \"magic\": \"QPACK\",\n"
      "  \"version\": 1,\n"
      "  \"modelName\": \"qwen3_next\",\n"
      "  \"sourceCheckpoint\": \"osm20-resident-expert-fixture\",\n"
      "  \"quantBits\": 4,\n"
      "  \"quantGroupSize\": 64,\n"
      "  \"files\": {\n"
      "    \"packed_experts/layout.json\": " + std::to_string(layout_size) + ",\n"
      "    \"packed_experts/layer_00.bin\": " + std::to_string(layer_size) + "\n"
      "  }\n"
      "}\n";

  write_text(root / "manifest.json", manifest);
  return root;
}

std::vector<float> cpu_projection(
    const QpackExpertQ4View& view,
    std::span<const float> x) {
  const auto dequantized = cpu::dequantize_affine_rows(
      view.packed,
      view.out_dim,
      view.packed_cols,
      view.scales,
      view.biases,
      cpu::AffineQuantSpec{
          .bits = 4,
          .group_size = static_cast<std::uint32_t>(view.group_size),
      });

  return cpu::matvec_row_major(
      dequantized,
      view.out_dim,
      view.packed_cols * 8U,
      x);
}

std::vector<float> cpu_expert(
    const QpackExpertQ4View& gate,
    const QpackExpertQ4View& up,
    const QpackExpertQ4View& down,
    std::span<const float> x) {
  auto hidden = cpu_projection(gate, x);
  const auto up_values = cpu_projection(up, x);
  cpu::swiglu_inplace(hidden, up_values);
  return cpu_projection(down, hidden);
}

float max_abs_error(
    std::span<const float> actual,
    std::span<const float> expected) {
  require(actual.size() == expected.size(), "result size mismatch");
  float result = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    result = std::max(result, std::fabs(actual[i] - expected[i]));
  }
  return result;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm20-resident-expert";

  try {
    QpackExpertStorage storage(make_qpack(root));

    // One host slot lets us prove that the Vulkan-resident expert survives
    // eviction of its source ExpertCacheEntry.
    ExpertCache cache(storage, storage.expert_stride_bytes(), 1U);
    const std::vector<std::uint32_t> first_pick{1};
    const auto first = cache.fetch(0, first_pick);
    require(first.size() == 1U, "expected source expert");

    const auto gate =
        bind_qpack_q4_projection(storage.reader(), first.front(), "gate_proj");
    const auto up =
        bind_qpack_q4_projection(storage.reader(), first.front(), "up_proj");
    const auto down =
        bind_qpack_q4_projection(storage.reader(), first.front(), "down_proj");

    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no context: " + diagnostic);
      }
      fs::remove_all(root);
      std::cout
          << "OSM-20 Vulkan resident expert: PASS (no compute device on host)\n";
      return 0;
    }

    auto resident = VulkanResidentExpert::create(
        *context,
        storage.reader(),
        first.front(),
        &diagnostic);
    require(resident.has_value(), diagnostic);
    require(resident->valid(), "resident expert must be valid");
    require(resident->input_dim() == 128U, "resident input dim mismatch");
    require(
        resident->intermediate_dim() == 64U,
        "resident intermediate dim mismatch");
    require(resident->output_dim() == 128U, "resident output dim mismatch");
    require(
        resident->packed_weight_bytes() == 12288U,
        "resident packed Q4 byte accounting mismatch");

    // Evict expert 1 from the single host cache slot after the Vulkan object
    // has been created. Its GPU-resident copy must remain executable.
    const std::vector<std::uint32_t> second_pick{0};
    const auto replacement = cache.fetch(0, second_pick);
    require(replacement.size() == 1U, "expected replacement expert");
    require(
        !cache.resident_slot(0, 1).has_value(),
        "source expert should have been evicted from host cache");
    require(
        cache.resident_slot(0, 0).has_value(),
        "replacement expert should occupy host cache");

    float worst_error = 0.0F;
    for (std::size_t token = 0; token < 2; ++token) {
      std::vector<float> x(128);
      for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] =
            static_cast<float>(
                static_cast<int>((i * (token + 5U)) % 29U) - 14) *
            0.03125F;
      }

      const auto expected = cpu_expert(gate, up, down, x);
      const auto actual = resident->run(*context, x);
      require(actual.executed, actual.diagnostic);

      const auto error = max_abs_error(actual.values, expected);
      worst_error = std::max(worst_error, error);
      require(error <= 0.01F, "resident expert CPU parity failed");
    }

    fs::remove_all(root);
    std::cout
        << "OSM-20 Vulkan resident expert: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  packed_weight_bytes=" << resident->packed_weight_bytes() << "\n"
        << "  token_reuse_count=2\n"
        << "  host_source_evicted_after_upload=PASS\n"
        << "  worst_max_abs_error=" << worst_error << "\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-20 Vulkan resident expert: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
