#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"
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

std::uint32_t pack_word(
    std::size_t expert,
    std::size_t row,
    std::size_t word) {
  std::uint32_t packed = 0;
  for (std::uint32_t lane = 0; lane < 8; ++lane) {
    const auto q = static_cast<std::uint32_t>(
        (expert * 5 + row * 3 + word * 7 + lane * 2) & 0xFU);
    packed |= q << (lane * 4U);
  }
  return packed;
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  constexpr std::size_t experts = 2;
  constexpr std::size_t out_dim = 4;
  constexpr std::size_t packed_cols = 16;
  constexpr std::size_t groups_per_row = 2;
  constexpr std::size_t weight_words = out_dim * packed_cols;
  constexpr std::size_t scale_values = out_dim * groups_per_row;

  constexpr std::size_t weight_offset = 0;
  constexpr std::size_t weight_bytes = weight_words * sizeof(std::uint32_t);
  constexpr std::size_t scale_offset = weight_offset + weight_bytes;
  constexpr std::size_t scale_bytes = scale_values * sizeof(float);
  constexpr std::size_t bias_offset = scale_offset + scale_bytes;
  constexpr std::size_t bias_bytes = scale_values * sizeof(float);
  constexpr std::size_t stride = bias_offset + bias_bytes;

  const std::string layout =
      "{\n"
      "  \"expertCount\": 2,\n"
      "  \"layerCount\": 1,\n"
      "  \"expertStride\": " + std::to_string(stride) + ",\n"
      "  \"sections\": [\n"
      "    {\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[4,16],\"offset\":0,\"size\":" + std::to_string(weight_bytes) + "},\n"
      "    {\"name\":\"gate_proj.scales\",\"dtype\":\"F32\",\"shape\":[4,2],\"offset\":" + std::to_string(scale_offset) + ",\"size\":" + std::to_string(scale_bytes) + "},\n"
      "    {\"name\":\"gate_proj.biases\",\"dtype\":\"F32\",\"shape\":[4,2],\"offset\":" + std::to_string(bias_offset) + ",\"size\":" + std::to_string(bias_bytes) + "}\n"
      "  ],\n"
      "  \"linearLayers\": [true]\n"
      "}\n";

  write_text(root / "packed_experts" / "layout.json", layout);

  std::ofstream layer(root / "packed_experts" / "layer_00.bin", std::ios::binary);
  if (!layer) throw std::runtime_error("unable to create synthetic layer file");

  for (std::size_t expert = 0; expert < experts; ++expert) {
    std::vector<std::byte> blob(stride);

    auto* words = reinterpret_cast<std::uint32_t*>(
        blob.data() + weight_offset);
    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t word = 0; word < packed_cols; ++word) {
        words[row * packed_cols + word] = pack_word(expert, row, word);
      }
    }

    auto* scales = reinterpret_cast<float*>(blob.data() + scale_offset);
    auto* biases = reinterpret_cast<float*>(blob.data() + bias_offset);
    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t g = 0; g < groups_per_row; ++g) {
        const auto i = row * groups_per_row + g;
        scales[i] =
            0.10F + 0.025F * static_cast<float>(expert + row + 2 * g + 1);
        biases[i] =
            -0.35F + 0.05F * static_cast<float>(expert * 2 + row + g);
      }
    }

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
      "  \"modelName\": \"osm13-synthetic\",\n"
      "  \"sourceCheckpoint\": \"synthetic\",\n"
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

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm13-qpack-expert";

  try {
    QpackExpertStorage storage(make_qpack(root));
    ExpertCache cache(storage, storage.expert_stride_bytes() * 2U);

    const std::vector<std::uint32_t> picks{1};
    const auto entries = cache.fetch(0, picks);
    require(entries.size() == 1U, "expected one cached expert");

    const auto view = bind_qpack_q4_projection(
        storage.reader(), entries.front(), "gate_proj");
    require(view.out_dim == 4U, "qpack projection out_dim mismatch");
    require(view.packed_cols == 16U, "qpack projection packed_cols mismatch");
    require(view.group_size == 64U, "qpack projection group_size mismatch");

    std::vector<float> x(view.packed_cols * 8U);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] =
          static_cast<float>(static_cast<int>(i % 13U) - 6) * 0.0625F;
    }

    const auto dequantized = cpu::dequantize_affine_rows(
        view.packed,
        view.out_dim,
        view.packed_cols,
        view.scales,
        view.biases,
        cpu::AffineQuantSpec{.bits = 4, .group_size = 64});

    const auto expected = cpu::matvec_row_major(
        dequantized,
        view.out_dim,
        view.packed_cols * 8U,
        x);

    std::string context_diagnostic;
    auto context = VulkanComputeContext::create(&context_diagnostic);
    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no Vulkan compute context was created: " +
            context_diagnostic);
      }

      std::cout
          << "OSM-13 qpack expert Vulkan bridge: PASS (no compute device on host)\n"
          << context_diagnostic << "\n";
      fs::remove_all(root);
      return 0;
    }

    const auto result = run_vulkan_qpack_q4_projection(
        *context,
        storage.reader(),
        entries.front(),
        "gate_proj",
        x);

    require(result.executed, result.diagnostic);
    require(result.values.size() == expected.size(), "result size mismatch");

    float max_abs_error = 0.0F;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float error = std::fabs(result.values[i] - expected[i]);
      max_abs_error = std::max(max_abs_error, error);
      if (error > 5e-4F) {
        throw std::runtime_error(
            "qpack expert GEMV mismatch at row " + std::to_string(i));
      }
    }

    const auto first_stats = cache.stats();
    require(first_stats.misses == 1U, "first fetch should miss");

    const auto again = cache.fetch(0, picks);
    require(again.size() == 1U, "repeat fetch size mismatch");
    const auto second_stats = cache.stats();
    require(second_stats.hits == 1U, "repeat fetch should hit");
    require(
        again.front().slot == entries.front().slot,
        "repeat fetch should reuse resident cache slot");

    std::cout
        << "OSM-13 qpack expert Vulkan bridge: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  slot=" << entries.front().slot
        << " stride=" << cache.stride_bytes() << "\n"
        << "  max_abs_error=" << max_abs_error << "\n";

    fs::remove_all(root);
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-13 qpack expert Vulkan bridge: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
