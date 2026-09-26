#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/container/safetensors.hpp"
#include "orbi/streammoe/conversion/dense_safetensors_conversion.hpp"
#include "orbi/streammoe/conversion/qpack_plan.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_dense_conversion_tests "
          "<metadata-dir> <dense-pilot-dir>");
    }

    const fs::path metadata_dir(argv[1]);
    const fs::path pilot_dir(argv[2]);

    const auto plan = build_qpack_conversion_plan(metadata_dir);
    const auto manifest = inspect_dense_conversion_manifest(
        pilot_dir / "dense-pilot.json");
    const auto output = pilot_dir / "model.safetensors";

    const auto result = convert_dense_plan_slice(
        plan,
        manifest,
        pilot_dir,
        output,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 64U});

    require(result.source_tensor_count == 2U, "official source tensor count mismatch");
    require(result.output_tensor_count == 4U, "official output tensor count mismatch");
    require(result.source_bytes == 8192ULL, "official source byte count mismatch");
    require(
        result.output_payload_bytes == 9472ULL,
        "official output payload byte count mismatch");

    SafetensorsReader reader(output);
    require(
        reader.info("model.norm.weight").dtype == "F32",
        "official norm dtype mismatch");
    require(
        reader.info("model.norm.weight").shape ==
            std::vector<std::size_t>({2048U}),
        "official norm shape mismatch");

    const std::string base =
        "model.layers.0.mlp.shared_expert_gate";
    require(
        reader.info(base + ".weight").dtype == "U32",
        "official shared gate packed dtype mismatch");
    require(
        reader.info(base + ".weight").shape ==
            std::vector<std::size_t>({1U, 256U}),
        "official shared gate packed shape mismatch");
    require(
        reader.info(base + ".scales").shape ==
            std::vector<std::size_t>({1U, 32U}),
        "official shared gate scales shape mismatch");
    require(
        reader.info(base + ".biases").shape ==
            std::vector<std::size_t>({1U, 32U}),
        "official shared gate biases shape mismatch");

    const auto norm = reader.read_floats("model.norm.weight");
    const auto scales = reader.read_floats(base + ".scales");
    require(norm.size() == 2048U, "official norm element count mismatch");
    require(scales.size() == 32U, "official scale element count mismatch");
    require(
        std::all_of(norm.begin(), norm.end(), [](float v) { return std::isfinite(v); }),
        "official norm contains non-finite values");
    require(
        std::all_of(scales.begin(), scales.end(), [](float v) {
          return std::isfinite(v) && v > 0.0F;
        }),
        "official affine scales invalid");

    std::cout
        << "OSM-40A official bounded dense/global conversion: PASS\n"
        << "  source_tensors=2\n"
        << "  source_bytes=" << result.source_bytes << "\n"
        << "  output_tensors=" << result.output_tensor_count << "\n"
        << "  output_payload_bytes=" << result.output_payload_bytes << "\n"
        << "  model_norm_BF16_to_F32=PASS\n"
        << "  shared_gate_BF16_to_Q4_affine=PASS\n"
        << "  production_safetensors_reader=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-40A official bounded dense/global conversion: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
