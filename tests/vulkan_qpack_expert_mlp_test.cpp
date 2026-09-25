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
#include "orbi/streammoe/backend/vulkan_qpack_expert_mlp.hpp"
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
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
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
      const auto value = pack_word(projection_seed, expert, row, word);
      write_u32(
          blob,
          weight_offset +
              (row * packed_cols + word) * sizeof(std::uint32_t),
          value);
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

  constexpr std::size_t payload_bytes = 13824;
  constexpr std::size_t stride = 16384;
  static_assert(payload_bytes <= stride);

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
      "  \"sourceCheckpoint\": \"osm15-upstream-shaped-fixture\",\n"
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

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm15-expert-mlp";

  try {
    QpackExpertStorage storage(make_qpack(root));
    require(
        storage.expert_stride_bytes() == 16384U,
        "fixture must preserve upstream 16 KiB expert stride");

    ExpertCache cache(storage, storage.expert_stride_bytes() * 2U);
    const std::vector<std::uint32_t> picks{1};
    const auto entries = cache.fetch(0, picks);
    require(entries.size() == 1U, "expected one cached expert");

    const auto gate =
        bind_qpack_q4_projection(storage.reader(), entries.front(), "gate_proj");
    const auto up =
        bind_qpack_q4_projection(storage.reader(), entries.front(), "up_proj");
    const auto down =
        bind_qpack_q4_projection(storage.reader(), entries.front(), "down_proj");

    require(gate.out_dim == 64U, "gate intermediate dimension mismatch");
    require(up.out_dim == 64U, "up intermediate dimension mismatch");
    require(down.out_dim == 128U, "down hidden dimension mismatch");
    require(gate.metadata_dtype == "BF16", "expected BF16 metadata");

    std::vector<float> x(128);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] =
          static_cast<float>(static_cast<int>(i % 19U) - 9) * 0.03125F;
    }

    auto expected_gate = cpu_projection(gate, x);
    const auto expected_up = cpu_projection(up, x);
    cpu::swiglu_inplace(expected_gate, expected_up);
    const auto expected = cpu_projection(down, expected_gate);

    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no context: " + diagnostic);
      }

      fs::remove_all(root);
      std::cout
          << "OSM-15 streamed expert MLP: PASS (no compute device on host)\n";
      return 0;
    }

    const auto result = run_vulkan_qpack_q4_expert_mlp(
        *context,
        storage.reader(),
        entries.front(),
        x);

    require(result.executed, result.diagnostic);
    require(result.values.size() == expected.size(), "expert output size mismatch");

    float max_abs_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const auto error = std::fabs(result.values[i] - expected[i]);
      max_abs_error = std::max(max_abs_error, error);
      if (error > 0.01F) {
        throw std::runtime_error(
            "expert MLP mismatch at index " + std::to_string(i) +
            ", error=" + std::to_string(error));
      }
    }

    require(cache.stats().misses == 1U, "expert must be loaded once");

    fs::remove_all(root);
    std::cout
        << "OSM-15 streamed expert MLP: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  expert_stride=16384\n"
        << "  topology=gate + up + SwiGLU + down\n"
        << "  max_abs_error=" << max_abs_error << "\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-15 streamed expert MLP: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
