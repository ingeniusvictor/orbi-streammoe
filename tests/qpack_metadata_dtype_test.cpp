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

std::uint16_t f32_to_bf16(float value) {
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
}

std::uint16_t f32_to_f16(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign =
      static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent == 0xFFU) {
    if (mantissa == 0U) return static_cast<std::uint16_t>(sign | 0x7C00U);
    return static_cast<std::uint16_t>(
        sign | 0x7C00U | static_cast<std::uint16_t>(mantissa >> 13U) | 1U);
  }

  int half_exp = static_cast<int>(exponent) - 127 + 15;
  if (half_exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }

  if (half_exp <= 0) {
    if (half_exp < -10) return sign;

    std::uint32_t normalized = mantissa | 0x800000U;
    const int shift = 14 - half_exp;
    std::uint32_t half_mantissa = normalized >> shift;
    const std::uint32_t remainder_mask = (1U << shift) - 1U;
    const std::uint32_t remainder = normalized & remainder_mask;
    const std::uint32_t halfway = 1U << (shift - 1);

    if (remainder > halfway ||
        (remainder == halfway && (half_mantissa & 1U) != 0U)) {
      ++half_mantissa;
    }

    return static_cast<std::uint16_t>(sign | half_mantissa);
  }

  std::uint32_t half_mantissa = mantissa >> 13U;
  const std::uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (half_mantissa & 1U) != 0U)) {
    ++half_mantissa;
    if (half_mantissa == 0x400U) {
      half_mantissa = 0;
      ++half_exp;
      if (half_exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
      }
    }
  }

  return static_cast<std::uint16_t>(
      sign |
      (static_cast<std::uint16_t>(half_exp) << 10U) |
      static_cast<std::uint16_t>(half_mantissa));
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

std::size_t dtype_bytes(const std::string& dtype) {
  if (dtype == "F32") return 4U;
  if (dtype == "F16" || dtype == "BF16") return 2U;
  throw std::invalid_argument("unsupported test dtype");
}

void store_float(
    std::vector<std::byte>& blob,
    std::size_t offset,
    std::size_t index,
    const std::string& dtype,
    float value) {
  if (dtype == "F32") {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    std::memcpy(
        blob.data() + offset + index * sizeof(bits),
        &bits,
        sizeof(bits));
    return;
  }

  const std::uint16_t bits =
      dtype == "F16" ? f32_to_f16(value) : f32_to_bf16(value);
  std::memcpy(
      blob.data() + offset + index * sizeof(bits),
      &bits,
      sizeof(bits));
}

float expected_scale(std::size_t expert, std::size_t row, std::size_t group) {
  const int bucket = static_cast<int>((expert + row + group) % 6U) + 1;
  return static_cast<float>(bucket) * 0.125F;
}

float expected_bias(std::size_t expert, std::size_t row, std::size_t group) {
  const int bucket =
      static_cast<int>((expert * 2U + row + group) % 7U) - 3;
  return static_cast<float>(bucket) * 0.125F;
}

fs::path make_qpack(const fs::path& root, const std::string& dtype) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  constexpr std::size_t experts = 2;
  constexpr std::size_t out_dim = 4;
  constexpr std::size_t packed_cols = 16;
  constexpr std::size_t groups_per_row = 2;
  constexpr std::size_t weight_words = out_dim * packed_cols;
  constexpr std::size_t metadata_values = out_dim * groups_per_row;

  const std::size_t element_bytes = dtype_bytes(dtype);
  constexpr std::size_t weight_offset = 0;
  constexpr std::size_t weight_bytes = weight_words * sizeof(std::uint32_t);
  const std::size_t scale_offset = weight_offset + weight_bytes;
  const std::size_t scale_bytes = metadata_values * element_bytes;
  const std::size_t bias_offset = scale_offset + scale_bytes;
  const std::size_t bias_bytes = metadata_values * element_bytes;
  const std::size_t stride = bias_offset + bias_bytes;

  const std::string layout =
      "{\n"
      "  \"expertCount\": 2,\n"
      "  \"layerCount\": 1,\n"
      "  \"expertStride\": " + std::to_string(stride) + ",\n"
      "  \"sections\": [\n"
      "    {\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[4,16],\"offset\":0,\"size\":" + std::to_string(weight_bytes) + "},\n"
      "    {\"name\":\"gate_proj.scales\",\"dtype\":\"" + dtype + "\",\"shape\":[4,2],\"offset\":" + std::to_string(scale_offset) + ",\"size\":" + std::to_string(scale_bytes) + "},\n"
      "    {\"name\":\"gate_proj.biases\",\"dtype\":\"" + dtype + "\",\"shape\":[4,2],\"offset\":" + std::to_string(bias_offset) + ",\"size\":" + std::to_string(bias_bytes) + "}\n"
      "  ],\n"
      "  \"linearLayers\": [true]\n"
      "}\n";

  write_text(root / "packed_experts" / "layout.json", layout);

  std::ofstream layer(
      root / "packed_experts" / "layer_00.bin",
      std::ios::binary);
  if (!layer) throw std::runtime_error("unable to create synthetic layer file");

  for (std::size_t expert = 0; expert < experts; ++expert) {
    std::vector<std::byte> blob(stride);

    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t word = 0; word < packed_cols; ++word) {
        const auto packed = pack_word(expert, row, word);
        const auto offset =
            weight_offset +
            (row * packed_cols + word) * sizeof(std::uint32_t);
        std::memcpy(blob.data() + offset, &packed, sizeof(packed));
      }
    }

    for (std::size_t row = 0; row < out_dim; ++row) {
      for (std::size_t group = 0; group < groups_per_row; ++group) {
        const auto index = row * groups_per_row + group;
        store_float(
            blob,
            scale_offset,
            index,
            dtype,
            expected_scale(expert, row, group));
        store_float(
            blob,
            bias_offset,
            index,
            dtype,
            expected_bias(expert, row, group));
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
      "  \"modelName\": \"osm14-" + dtype + "\",\n"
      "  \"sourceCheckpoint\": \"synthetic-upstream-dtype-fixture\",\n"
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

void certify_dtype(
    const fs::path& root,
    const std::string& dtype,
    VulkanComputeContext* context) {
  QpackExpertStorage storage(make_qpack(root, dtype));
  ExpertCache cache(storage, storage.expert_stride_bytes() * 2U);

  const std::vector<std::uint32_t> picks{1};
  const auto entries = cache.fetch(0, picks);
  require(entries.size() == 1U, dtype + ": expected one cached expert");

  const auto view = bind_qpack_q4_projection(
      storage.reader(),
      entries.front(),
      "gate_proj");

  require(view.metadata_dtype == dtype, dtype + ": decoded dtype mismatch");
  require(view.scales.size() == 8U, dtype + ": scale count mismatch");
  require(view.biases.size() == 8U, dtype + ": bias count mismatch");

  for (std::size_t row = 0; row < 4U; ++row) {
    for (std::size_t group = 0; group < 2U; ++group) {
      const auto index = row * 2U + group;
      require(
          std::fabs(view.scales[index] - expected_scale(1U, row, group)) <
              1e-6F,
          dtype + ": scale decode mismatch");
      require(
          std::fabs(view.biases[index] - expected_bias(1U, row, group)) <
              1e-6F,
          dtype + ": bias decode mismatch");
    }
  }

  std::vector<float> x(view.packed_cols * 8U);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] =
        static_cast<float>(static_cast<int>(i % 15U) - 7) * 0.0625F;
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

  if (context == nullptr) return;

  const auto result = run_vulkan_qpack_q4_projection(
      *context,
      storage.reader(),
      entries.front(),
      "gate_proj",
      x);

  require(result.executed, dtype + ": " + result.diagnostic);
  require(
      result.values.size() == expected.size(),
      dtype + ": Vulkan output size mismatch");

  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto error = std::fabs(result.values[i] - expected[i]);
    if (error > 5e-4F) {
      throw std::runtime_error(
          dtype + ": Vulkan parity mismatch at row " + std::to_string(i) +
          ", error=" + std::to_string(error));
    }
  }
}

}  // namespace

int main() {
  const auto base =
      fs::temp_directory_path() / "orbi-streammoe-osm14-qpack-metadata";

  try {
    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);

    if (!context.has_value() && require_vulkan_compute()) {
      throw std::runtime_error(
          "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no Vulkan compute context was created: " +
          diagnostic);
    }

    for (const std::string dtype : {"F32", "F16", "BF16"}) {
      certify_dtype(
          base / dtype,
          dtype,
          context.has_value() ? &*context : nullptr);
    }

    fs::remove_all(base);
    std::cout
        << "OSM-14 qpack metadata dtypes: PASS\n"
        << "  F32/F16/BF16 decode certified\n"
        << (context.has_value()
                ? "  Vulkan parity executed\n"
                : "  Vulkan parity skipped: " + diagnostic + "\n");
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(base);
    std::cerr
        << "OSM-14 qpack metadata dtypes: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
