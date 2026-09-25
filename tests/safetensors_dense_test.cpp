#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/container/qpack_dense.hpp"
#include "orbi/streammoe/container/safetensors.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void append_u16_le(std::vector<std::byte>& out, std::uint16_t value) {
  out.push_back(static_cast<std::byte>(value & 0xFFU));
  out.push_back(static_cast<std::byte>((value >> 8U) & 0xFFU));
}

void append_u32_le(std::vector<std::byte>& out, std::uint32_t value) {
  for (std::uint32_t shift = 0; shift < 32U; shift += 8U) {
    out.push_back(static_cast<std::byte>((value >> shift) & 0xFFU));
  }
}

void append_u64_le(std::ofstream& out, std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (8U * i)) & 0xFFU);
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void append_f32_le(std::vector<std::byte>& out, float value) {
  append_u32_le(out, std::bit_cast<std::uint32_t>(value));
}

std::uint16_t bf16(float value) {
  return static_cast<std::uint16_t>(
      std::bit_cast<std::uint32_t>(value) >> 16U);
}

fs::path write_safetensors(const fs::path& path) {
  std::vector<std::byte> payload;

  append_f32_le(payload, 1.25F);
  append_f32_le(payload, -2.5F);

  append_u16_le(payload, 0x3C00U);
  append_u16_le(payload, 0xC000U);
  append_u16_le(payload, 0x3800U);

  append_u16_le(payload, bf16(1.5F));
  append_u16_le(payload, bf16(-0.75F));

  append_u32_le(payload, 0x12345678U);
  append_u32_le(payload, 0xDEADBEEFU);

  append_f32_le(payload, 3.0F);
  append_f32_le(payload, -4.0F);

  const std::string header =
      "{"
      "\"plain_f32\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
      "\"plain_f16\":{\"dtype\":\"F16\",\"shape\":[3],\"data_offsets\":[8,14]},"
      "\"plain_bf16\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[14,18]},"
      "\"packed_u32\":{\"dtype\":\"U32\",\"shape\":[2],\"data_offsets\":[18,26]},"
      "\"language_model.router.weight\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[26,34]}"
      "}";

  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create safetensors fixture");
  append_u64_le(out, header.size());
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(
      reinterpret_cast<const char*>(payload.data()),
      static_cast<std::streamsize>(payload.size()));
  return path;
}

void write_text(const fs::path& path, const std::string& value) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write fixture file");
  out << value;
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  write_safetensors(root / "model.safetensors");

  const std::string layout =
      "{"
      "\"expertCount\":1,"
      "\"layerCount\":1,"
      "\"expertStride\":16,"
      "\"sections\":["
      "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[1],\"offset\":0,\"size\":4}"
      "],"
      "\"linearLayers\":[true]"
      "}";
  write_text(root / "packed_experts" / "layout.json", layout);

  std::vector<char> expert_bytes(16, 0);
  std::ofstream layer(
      root / "packed_experts" / "layer_00.bin",
      std::ios::binary);
  layer.write(expert_bytes.data(), expert_bytes.size());
  layer.close();

  const auto dense_size = fs::file_size(root / "model.safetensors");
  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");
  const auto layer_size =
      fs::file_size(root / "packed_experts" / "layer_00.bin");

  const std::string manifest =
      "{"
      "\"magic\":\"QPACK\","
      "\"version\":1,"
      "\"modelName\":\"qwen3_next\","
      "\"sourceCheckpoint\":\"osm26a\","
      "\"files\":{"
      "\"model.safetensors\":" + std::to_string(dense_size) + ","
      "\"packed_experts/layout.json\":" + std::to_string(layout_size) + ","
      "\"packed_experts/layer_00.bin\":" + std::to_string(layer_size) +
      "}"
      "}";
  write_text(root / "manifest.json", manifest);
  return root;
}

fs::path write_bad_safetensors(const fs::path& path) {
  const std::string header =
      "{"
      "\"broken\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}"
      "}";
  std::ofstream out(path, std::ios::binary);
  append_u64_le(out, header.size());
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  std::uint32_t value = 0;
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  return path;
}

bool close(float a, float b, float eps = 1e-6F) {
  return std::fabs(a - b) <= eps;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm26a-safetensors";

  try {
    const auto qpack_path = make_qpack(root);
    const auto dense_path = root / "model.safetensors";

    SafetensorsReader reader(dense_path);
    require(reader.contains("plain_f32"), "missing F32 tensor");
    require(reader.contains("plain_f16"), "missing F16 tensor");
    require(reader.contains("plain_bf16"), "missing BF16 tensor");
    require(reader.contains("packed_u32"), "missing U32 tensor");

    const auto f32 = reader.read_floats("plain_f32");
    require(f32.size() == 2U, "F32 size mismatch");
    require(close(f32[0], 1.25F), "F32 value 0 mismatch");
    require(close(f32[1], -2.5F), "F32 value 1 mismatch");

    const auto f16 = reader.read_floats("plain_f16");
    require(f16.size() == 3U, "F16 size mismatch");
    require(close(f16[0], 1.0F), "F16 value 0 mismatch");
    require(close(f16[1], -2.0F), "F16 value 1 mismatch");
    require(close(f16[2], 0.5F), "F16 value 2 mismatch");

    const auto bf = reader.read_floats("plain_bf16");
    require(bf.size() == 2U, "BF16 size mismatch");
    require(close(bf[0], 1.5F), "BF16 value 0 mismatch");
    require(close(bf[1], -0.75F), "BF16 value 1 mismatch");

    const auto u32 = reader.read_u32("packed_u32");
    require(u32.size() == 2U, "U32 size mismatch");
    require(u32[0] == 0x12345678U, "U32 value 0 mismatch");
    require(u32[1] == 0xDEADBEEFU, "U32 value 1 mismatch");

    const auto first_offset = reader.absolute_offset("plain_f32");
    require(
        first_offset == reader.data_start(),
        "first tensor absolute offset mismatch");
    require(
        reader.absolute_offset("packed_u32") == reader.data_start() + 18U,
        "packed tensor absolute offset mismatch");

    QpackReader qpack(qpack_path);
    QpackDenseReader dense(qpack);
    require(
        dense.contains("router.weight"),
        "language_model prefix fallback failed");

    const auto router = dense.read_floats("router.weight");
    require(router.size() == 2U, "router tensor size mismatch");
    require(close(router[0], 3.0F), "router value 0 mismatch");
    require(close(router[1], -4.0F), "router value 1 mismatch");

    bool malformed_rejected = false;
    try {
      SafetensorsReader bad(
          write_bad_safetensors(root / "bad.safetensors"));
      (void)bad;
    } catch (const std::exception&) {
      malformed_rejected = true;
    }
    require(malformed_rejected, "malformed tensor geometry must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-26A safetensors dense reader: PASS\n"
        << "  tensors=" << reader.tensor_names().size() << "\n"
        << "  F32/F16/BF16/U32 decode=PASS\n"
        << "  absolute_offsets=PASS\n"
        << "  language_model_prefix=PASS\n"
        << "  malformed_geometry_rejected=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-26A safetensors dense reader: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
