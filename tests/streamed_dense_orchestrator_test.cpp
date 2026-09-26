#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/container/safetensors.hpp"
#include "orbi/streammoe/conversion/bf16_slice.hpp"
#include "orbi/streammoe/conversion/streamed_dense_orchestrator.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::vector<std::byte> make_bf16(
    std::size_t count,
    float base) {
  std::vector<std::byte> out(count * 2U);
  for (std::size_t i = 0U; i < count; ++i) {
    const float value =
        base + static_cast<float>(i % 9U) * 0.125F;
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto bf16 = static_cast<std::uint16_t>(bits >> 16U);
    out[i * 2U] = static_cast<std::byte>(bf16 & 0xFFU);
    out[i * 2U + 1U] =
        static_cast<std::byte>((bf16 >> 8U) & 0xFFU);
  }
  return out;
}

json write_chunk(
    const fs::path& root,
    const std::string& filename,
    std::size_t first_row,
    std::size_t row_count,
    std::size_t row_elements,
    float base) {
  const auto bytes = make_bf16(row_count * row_elements, base);
  const auto path = root / filename;
  std::ofstream out(path, std::ios::binary);
  out.write(
      reinterpret_cast<const char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (!out) throw std::runtime_error("unable to write BF16 fixture");
  return {
      {"first_row", first_row},
      {"row_count", row_count},
      {"source_file", filename},
      {"source_byte_size", bytes.size()},
      {"source_fnv1a64", hex64(fnv1a64(bytes))},
  };
}

QpackConversionPlan fixture_plan() {
  QpackConversionPlan plan;
  plan.source_checkpoint = "fixture";
  plan.layer_count = 1U;
  plan.expert_count = 1U;
  plan.entries = {
      {
          "model.embed_tokens.weight",
          "fixture.safetensors",
          QpackConversionClass::global_dense,
          QpackConversionAction::affine_quantize,
          "model.safetensors",
          "model.embed_tokens",
          std::nullopt,
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
  return plan;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm40c-dense-orchestrator";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    json tensors = json::array();

    tensors.push_back({
        {"source_tensor", "model.embed_tokens.weight"},
        {"source_shape", {4U, 8U}},
        {"chunks", json::array({
            write_chunk(root, "embed_0_2.bf16", 0U, 2U, 8U, -1.0F),
            write_chunk(root, "embed_2_2.bf16", 2U, 2U, 8U, 1.0F),
        })},
    });
    tensors.push_back({
        {"source_tensor", "model.norm.weight"},
        {"source_shape", {4U}},
        {"chunks", json::array({
            write_chunk(root, "norm_0_2.bf16", 0U, 2U, 1U, 0.5F),
            write_chunk(root, "norm_2_2.bf16", 2U, 2U, 1U, 1.5F),
        })},
    });

    const json manifest_json = {
        {"schema_version", 1U},
        {"model", "fixture"},
        {"snapshot", "fixture"},
        {"tensors", tensors},
    };
    const auto manifest_path = root / "dense-stream.json";
    {
      std::ofstream out(manifest_path);
      out << manifest_json.dump(2) << "\n";
    }

    const auto manifest =
        inspect_streamed_dense_manifest(manifest_path);
    const auto plan = fixture_plan();
    const auto quantization =
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U};
    const auto specs =
        plan_streamed_dense_output_specs(plan, manifest, quantization);

    require(specs.size() == 4U, "target tensor count mismatch");
    require(
        specs[0].name == "model.embed_tokens.biases" &&
        specs[1].name == "model.embed_tokens.scales" &&
        specs[2].name == "model.embed_tokens.weight" &&
        specs[3].name == "model.norm.weight",
        "target tensor inventory mismatch");

    const auto output = root / "model.safetensors";
    const auto journal = root / "model.progress.json";
    auto builder =
        StreamedSafetensorsBuilder::open(output, journal, specs);

    const auto first = execute_streamed_dense_manifest(
        plan,
        manifest,
        root,
        builder,
        quantization);

    require(first.tensor_count == 2U, "source tensor count mismatch");
    require(first.requested_chunks == 4U, "requested chunk count mismatch");
    require(first.converted_chunks == 4U, "converted chunk count mismatch");
    require(first.skipped_chunks == 0U, "unexpected first-pass skips");
    require(first.source_bytes == 72U, "first-pass source bytes mismatch");
    require(
        std::isfinite(first.max_abs_error) &&
        std::isfinite(first.weighted_mean_abs_error),
        "aggregate quantization metrics must be finite");
    require(builder.state().complete(), "builder should be complete");
    builder.finalize();

    SafetensorsReader reader(output);
    require(reader.tensor_names().size() == 4U, "reader inventory mismatch");
    require(
        reader.info("model.embed_tokens.weight").shape ==
            std::vector<std::size_t>({4U, 1U}),
        "packed embedding shape mismatch");
    require(
        reader.info("model.embed_tokens.scales").shape ==
            std::vector<std::size_t>({4U, 2U}),
        "embedding scale shape mismatch");
    require(
        reader.info("model.norm.weight").shape ==
            std::vector<std::size_t>({4U}),
        "norm output shape mismatch");

    fs::remove(root / "embed_0_2.bf16");
    fs::remove(root / "embed_2_2.bf16");
    fs::remove(root / "norm_0_2.bf16");
    fs::remove(root / "norm_2_2.bf16");

    auto resumed =
        StreamedSafetensorsBuilder::open(output, journal, specs);
    const auto second = execute_streamed_dense_manifest(
        plan,
        manifest,
        root,
        resumed,
        quantization);
    require(second.converted_chunks == 0U, "resume must not reconvert");
    require(second.skipped_chunks == 4U, "resume must skip all chunks");
    require(second.source_bytes == 0U, "resume must read no BF16 source");
    resumed.finalize();

    bool gap_rejected = false;
    try {
      auto bad_json = manifest_json;
      bad_json["tensors"][0]["chunks"][0]["first_row"] = 1U;
      const auto bad_path = root / "bad-stream.json";
      std::ofstream out(bad_path);
      out << bad_json.dump(2) << "\n";
      out.close();

      const auto bad = inspect_streamed_dense_manifest(bad_path);
      const auto bad_output = root / "bad.safetensors";
      const auto bad_journal = root / "bad.progress.json";
      auto bad_builder = StreamedSafetensorsBuilder::open(
          bad_output,
          bad_journal,
          plan_streamed_dense_output_specs(plan, bad, quantization));
      (void)execute_streamed_dense_manifest(
          plan, bad, root, bad_builder, quantization);
    } catch (const std::exception&) {
      gap_rejected = true;
    }
    require(gap_rejected, "non-zero initial source gap must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-40C streamed dense/global orchestrator: PASS\n"
        << "  plan_to_output_inventory=PASS\n"
        << "  BF16_to_F32_streaming=PASS\n"
        << "  BF16_to_affine_Q4_streaming=PASS\n"
        << "  multi_tensor_chunk_execution=PASS\n"
        << "  restart_skip_before_source_read=PASS\n"
        << "  production_reader_roundtrip=PASS\n"
        << "  source_gap_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-40C streamed dense/global orchestrator: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
