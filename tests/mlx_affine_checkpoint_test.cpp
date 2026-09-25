#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

struct TensorFixture {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::vector<std::byte> bytes;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void append_u16_le(
    std::vector<std::byte>& out,
    std::uint16_t value) {
  out.push_back(static_cast<std::byte>(value & 0xFFU));
  out.push_back(static_cast<std::byte>((value >> 8U) & 0xFFU));
}

void append_u32_le(
    std::vector<std::byte>& out,
    std::uint32_t value) {
  for (std::uint32_t shift = 0; shift < 32U; shift += 8U) {
    out.push_back(
        static_cast<std::byte>((value >> shift) & 0xFFU));
  }
}

void append_u64_le(
    std::ofstream& out,
    std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] =
        static_cast<char>((value >> (8U * i)) & 0xFFU);
  }
  out.write(
      bytes.data(),
      static_cast<std::streamsize>(bytes.size()));
}

std::uint16_t bf16(float value) {
  return static_cast<std::uint16_t>(
      std::bit_cast<std::uint32_t>(value) >> 16U);
}

std::vector<std::byte> u32_bytes(
    std::initializer_list<std::uint32_t> values) {
  std::vector<std::byte> out;
  for (const auto value : values) append_u32_le(out, value);
  return out;
}

std::vector<std::byte> bf16_bytes(
    std::initializer_list<float> values) {
  std::vector<std::byte> out;
  for (const auto value : values) append_u16_le(out, bf16(value));
  return out;
}

std::string shape_json(
    const std::vector<std::size_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0U) out << ",";
    out << shape[i];
  }
  out << "]";
  return out.str();
}

void write_safetensors(
    const fs::path& path,
    const std::vector<TensorFixture>& tensors) {
  std::ostringstream header;
  header << "{";

  std::uint64_t offset = 0U;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (i != 0U) header << ",";
    const auto& tensor = tensors[i];

    header
        << "\"" << tensor.name << "\":{"
        << "\"dtype\":\"" << tensor.dtype << "\","
        << "\"shape\":" << shape_json(tensor.shape) << ","
        << "\"data_offsets\":["
        << offset << ","
        << (offset + tensor.bytes.size()) << "]}";

    offset += tensor.bytes.size();
  }
  header << "}";

  const auto header_text = header.str();
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error(
        "unable to create safetensors fixture");
  }

  append_u64_le(out, header_text.size());
  out.write(
      header_text.data(),
      static_cast<std::streamsize>(header_text.size()));

  for (const auto& tensor : tensors) {
    out.write(
        reinterpret_cast<const char*>(tensor.bytes.data()),
        static_cast<std::streamsize>(tensor.bytes.size()));
  }
}

void write_text(
    const fs::path& path,
    const std::string& value) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("unable to write fixture file");
  }
  out << value;
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  const std::string prefix = "language_model.";

  std::vector<TensorFixture> tensors{
      {
          prefix + "model.layers.0.mlp.gate.weight",
          "U32",
          {2, 1},
          u32_bytes({0x03020100U, 0x07060504U}),
      },
      {
          prefix + "model.layers.0.mlp.gate.scales",
          "BF16",
          {2, 1},
          bf16_bytes({0.5F, 0.25F}),
      },
      {
          prefix + "model.layers.0.mlp.gate.biases",
          "BF16",
          {2, 1},
          bf16_bytes({-1.0F, 1.5F}),
      },
      {
          prefix + "model.layers.0.mlp.shared_expert.gate_proj.weight",
          "U32",
          {2, 1},
          u32_bytes({0x76543210U, 0xFEDCBA98U}),
      },
      {
          prefix + "model.layers.0.mlp.shared_expert.gate_proj.scales",
          "BF16",
          {2, 2},
          bf16_bytes({0.5F, 0.25F, 0.125F, 1.0F}),
      },
      {
          prefix + "model.layers.0.mlp.shared_expert.gate_proj.biases",
          "BF16",
          {2, 2},
          bf16_bytes({0.0F, -0.5F, 0.75F, -1.0F}),
      },
      {
          prefix + "model.layers.0.bad.weight",
          "U32",
          {1, 1},
          u32_bytes({0x01020304U}),
      },
      {
          prefix + "model.layers.0.bad.scales",
          "BF16",
          {1, 2},
          bf16_bytes({1.0F, 2.0F}),
      },
      {
          prefix + "model.layers.0.bad.biases",
          "BF16",
          {1, 2},
          bf16_bytes({0.0F, 0.0F}),
      },
  };

  write_safetensors(root / "model.safetensors", tensors);

  const std::string config =
      "{"
      "\"model_type\":\"qwen3_next\","
      "\"quantization\":{"
      "\"group_size\":4,"
      "\"bits\":8,"
      "\"mode\":\"affine\","
      "\"model.layers.0.mlp.shared_expert.gate_proj\":{"
      "\"group_size\":4,"
      "\"bits\":4"
      "}"
      "}"
      "}";
  write_text(root / "config.json", config);

  const std::string layout =
      "{"
      "\"expertCount\":1,"
      "\"layerCount\":1,"
      "\"expertStride\":16,"
      "\"sections\":["
      "{\"name\":\"gate_proj.weight\",\"dtype\":\"U32\","
      "\"shape\":[1],\"offset\":0,\"size\":4}"
      "],"
      "\"linearLayers\":[true]"
      "}";
  write_text(root / "packed_experts" / "layout.json", layout);

  std::vector<char> expert_bytes(16, 0);
  std::ofstream layer(
      root / "packed_experts" / "layer_00.bin",
      std::ios::binary);
  layer.write(
      expert_bytes.data(),
      static_cast<std::streamsize>(expert_bytes.size()));
  layer.close();

  const auto dense_size = fs::file_size(root / "model.safetensors");
  const auto config_size = fs::file_size(root / "config.json");
  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");
  const auto layer_size =
      fs::file_size(root / "packed_experts" / "layer_00.bin");

  const std::string manifest =
      "{"
      "\"magic\":\"QPACK\","
      "\"version\":1,"
      "\"modelName\":\"qwen3_next\","
      "\"sourceCheckpoint\":\"osm26b\","
      "\"files\":{"
      "\"model.safetensors\":" + std::to_string(dense_size) + ","
      "\"config.json\":" + std::to_string(config_size) + ","
      "\"packed_experts/layout.json\":" + std::to_string(layout_size) + ","
      "\"packed_experts/layer_00.bin\":" + std::to_string(layer_size) +
      "}"
      "}";

  write_text(root / "manifest.json", manifest);
  return root;
}

bool close(float a, float b, float eps = 1e-6F) {
  return std::fabs(a - b) <= eps;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm26b-affine";

  try {
    QpackReader qpack(make_qpack(root));
    QpackMlxCheckpoint checkpoint(qpack);

    require(
        checkpoint.quantization().default_quant.has_value(),
        "default quantization missing");
    require(
        checkpoint.quantization().default_quant->bits == 8U,
        "default bits mismatch");
    require(
        checkpoint.quantization().default_quant->group_size == 4U,
        "default group size mismatch");
    require(
        checkpoint.quantization().default_quant->mode == "affine",
        "default mode mismatch");

    const auto default_spec =
        checkpoint.quant_spec_for("model.layers.0.mlp.gate");
    require(default_spec.has_value(), "default module spec missing");
    require(default_spec->bits == 8U, "default module must be 8-bit");

    const auto override_spec = checkpoint.quant_spec_for(
        "model.layers.0.mlp.shared_expert.gate_proj");
    require(override_spec.has_value(), "override module spec missing");
    require(override_spec->bits == 4U, "override must be 4-bit");
    require(
        override_spec->mode == "affine",
        "missing override mode must default to affine");

    const auto router =
        checkpoint.read_affine_module("model.layers.0.mlp.gate");
    require(router.quant.bits == 8U, "router quant bits mismatch");
    require(router.row_count == 2U, "router rows mismatch");
    require(router.logical_in_dim == 4U, "router logical input mismatch");
    require(
        router.logical_shape == std::vector<std::size_t>({2, 4}),
        "router logical shape mismatch");
    require(router.packed.size() == 2U, "router packed size mismatch");
    require(
        router.packed[0] == 0x03020100U &&
        router.packed[1] == 0x07060504U,
        "router packed bytes changed");
    require(router.scales.size() == 2U, "router scales size mismatch");
    require(close(router.scales[0], 0.5F), "router scale mismatch");
    require(close(router.biases[1], 1.5F), "router bias mismatch");

    const auto shared = checkpoint.read_affine_module(
        "model.layers.0.mlp.shared_expert.gate_proj");
    require(shared.quant.bits == 4U, "shared override bits mismatch");
    require(shared.row_count == 2U, "shared rows mismatch");
    require(shared.logical_in_dim == 8U, "shared logical input mismatch");
    require(
        shared.logical_shape == std::vector<std::size_t>({2, 8}),
        "shared logical shape mismatch");
    require(shared.scales.size() == 4U, "shared scales size mismatch");
    require(close(shared.scales[2], 0.125F), "shared scale value mismatch");
    require(close(shared.biases[2], 0.75F), "shared bias value mismatch");

    require(
        !checkpoint.quant_spec_for("model.layers.0.not_quantized").has_value(),
        "unquantized module must not receive a quantization spec");

    bool bad_geometry_rejected = false;
    try {
      (void)checkpoint.read_affine_module("model.layers.0.bad");
    } catch (const std::exception&) {
      bad_geometry_rejected = true;
    }
    require(
        bad_geometry_rejected,
        "bad affine geometry must be rejected");

    const auto bad_config = root / "non_affine.json";
    write_text(
        bad_config,
        "{\"quantization\":{"
        "\"group_size\":64,"
        "\"bits\":4,"
        "\"mode\":\"mxfp4\""
        "}}");

    bool non_affine_rejected = false;
    try {
      (void)parse_mlx_quantization_config(bad_config);
    } catch (const std::exception&) {
      non_affine_rejected = true;
    }
    require(
        non_affine_rejected,
        "non-affine quantization mode must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-26B MLX affine checkpoint binding: PASS\n"
        << "  default_quant=8-bit/g4 affine\n"
        << "  override_quant=4-bit/g4 affine\n"
        << "  packed_weights_preserved=PASS\n"
        << "  language_model_prefix=PASS\n"
        << "  bad_geometry_rejected=PASS\n"
        << "  non_affine_rejected=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-26B MLX affine checkpoint binding: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
