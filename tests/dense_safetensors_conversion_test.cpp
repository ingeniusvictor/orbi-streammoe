#include <algorithm>
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

#include "orbi/streammoe/container/safetensors.hpp"
#include "orbi/streammoe/conversion/bf16_slice.hpp"
#include "orbi/streammoe/conversion/dense_safetensors_conversion.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_bf16(
    const fs::path& path,
    const std::vector<float>& values) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create BF16 fixture");
  for (const auto value : values) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto bf16 = static_cast<std::uint16_t>(bits >> 16U);
    const char bytes[2] = {
        static_cast<char>(bf16 & 0xFFU),
        static_cast<char>((bf16 >> 8U) & 0xFFU)};
    out.write(bytes, 2);
  }
}

std::vector<std::byte> read_bytes(const fs::path& path) {
  std::vector<std::byte> out(
      static_cast<std::size_t>(fs::file_size(path)));
  std::ifstream in(path, std::ios::binary);
  in.read(
      reinterpret_cast<char*>(out.data()),
      static_cast<std::streamsize>(out.size()));
  return out;
}

DenseSourceTensorManifest source(
    const std::string& name,
    std::vector<std::size_t> shape,
    const fs::path& file) {
  const auto bytes = read_bytes(file);
  return {
      name,
      std::move(shape),
      static_cast<std::uint64_t>(bytes.size()),
      file.filename().string(),
      fnv1a64(bytes),
  };
}

bool close(float a, float b, float eps = 1e-5F) {
  return std::fabs(a - b) <= eps;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm40a-dense";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    const auto norm_file = root / "norm.bf16.bin";
    const auto matrix_file = root / "gate.bf16.bin";
    write_bf16(norm_file, {1.0F, -2.0F, 3.5F, 0.25F});
    write_bf16(
        matrix_file,
        {-1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 2.5F});

    QpackConversionPlan plan;
    plan.layer_count = 1U;
    plan.expert_count = 2U;
    plan.entries = {
        {
            "model.layers.0.mlp.shared_expert_gate.weight",
            "fixture.safetensors",
            QpackConversionClass::layer_dense,
            QpackConversionAction::affine_quantize,
            "model.safetensors",
            "model.layers.0.mlp.shared_expert_gate",
            0U,
        },
        {
            "model.norm.weight",
            "fixture.safetensors",
            QpackConversionClass::global_dense,
            QpackConversionAction::copy_bf16_to_f32,
            "model.safetensors",
            "model.norm.weight",
            std::nullopt,
        },
    };

    DenseConversionManifest manifest;
    manifest.model = "fixture";
    manifest.snapshot = "fixture";
    manifest.tensors = {
        source("model.norm.weight", {4U}, norm_file),
        source(
            "model.layers.0.mlp.shared_expert_gate.weight",
            {1U, 8U},
            matrix_file),
    };

    const auto output = root / "model.safetensors";
    const auto result = convert_dense_plan_slice(
        plan,
        manifest,
        root,
        output,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});

    require(result.source_tensor_count == 2U, "source tensor count mismatch");
    require(result.output_tensor_count == 4U, "output tensor count mismatch");
    require(result.source_bytes == 24U, "source byte count mismatch");

    SafetensorsReader reader(output);
    require(reader.contains("model.norm.weight"), "norm missing");
    require(
        reader.info("model.norm.weight").dtype == "F32",
        "norm output dtype mismatch");
    require(
        reader.info("model.norm.weight").shape ==
            std::vector<std::size_t>({4U}),
        "norm output shape mismatch");

    const auto norm = reader.read_floats("model.norm.weight");
    require(norm.size() == 4U, "norm element count mismatch");
    require(close(norm[0], 1.0F), "norm[0] mismatch");
    require(close(norm[1], -2.0F), "norm[1] mismatch");

    const std::string base =
        "model.layers.0.mlp.shared_expert_gate";
    require(
        reader.info(base + ".weight").dtype == "U32",
        "packed matrix dtype mismatch");
    require(
        reader.info(base + ".weight").shape ==
            std::vector<std::size_t>({1U, 1U}),
        "packed matrix shape mismatch");
    require(
        reader.info(base + ".scales").shape ==
            std::vector<std::size_t>({1U, 2U}),
        "scale shape mismatch");
    require(
        reader.info(base + ".biases").shape ==
            std::vector<std::size_t>({1U, 2U}),
        "bias shape mismatch");

    const auto packed = reader.read_u32(base + ".weight");
    const auto scales = reader.read_floats(base + ".scales");
    const auto biases = reader.read_floats(base + ".biases");
    require(packed.size() == 1U, "packed word count mismatch");
    require(scales.size() == 2U && biases.size() == 2U, "affine metadata mismatch");
    require(std::isfinite(scales[0]) && scales[0] > 0.0F, "scale must be finite");
    require(std::isfinite(biases[0]), "bias must be finite");

    const auto first_bytes = read_bytes(output);
    const auto first_hash = fnv1a64(first_bytes);
    const auto second = convert_dense_plan_slice(
        plan,
        manifest,
        root,
        output,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    require(second.output_tensor_count == 4U, "repeat tensor count mismatch");
    require(
        fnv1a64(read_bytes(output)) == first_hash,
        "deterministic safetensors output changed on repeat");

    fs::remove_all(root);
    std::cout
        << "OSM-40A bounded dense/global conversion: PASS\n"
        << "  BF16_to_F32=PASS\n"
        << "  BF16_to_Q4_affine=PASS\n"
        << "  deterministic_safetensors=PASS\n"
        << "  production_reader_roundtrip=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-40A bounded dense/global conversion: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
