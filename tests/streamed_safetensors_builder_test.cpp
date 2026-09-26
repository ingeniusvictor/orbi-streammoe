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
#include "orbi/streammoe/conversion/streamed_safetensors_builder.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> bf16_bytes(const std::vector<float>& values) {
  std::vector<std::byte> out(values.size() * 2U);
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    const auto bf16 = static_cast<std::uint16_t>(bits >> 16U);
    out[i * 2U] = static_cast<std::byte>(bf16 & 0xFFU);
    out[i * 2U + 1U] = static_cast<std::byte>((bf16 >> 8U) & 0xFFU);
  }
  return out;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm40b-streamed";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    const std::string base = "model.embed_tokens";
    std::vector<StreamedSafetensorSpec> specs = {
        {base + ".biases", "F32", {4U, 2U}},
        {base + ".scales", "F32", {4U, 2U}},
        {base + ".weight", "U32", {4U, 1U}},
    };

    const auto output = root / "model.safetensors";
    const auto journal = root / "model.progress.json";
    auto builder = StreamedSafetensorsBuilder::open(
        output, journal, specs);

    require(!builder.state().complete(), "fresh builder must be incomplete");

    const auto first_bf16 = bf16_bytes({
        -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 2.5F,
        -2.0F, -1.5F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 1.5F,
    });
    const auto first = convert_bf16_affine_q4_row_chunk(
        first_bf16, 2U, 8U, 4U);

    require(
        builder.write_rows(base + ".weight", 0U, 2U, first.packed_weight_bytes),
        "first weight chunk must commit");
    require(
        builder.write_rows(base + ".scales", 0U, 2U, first.scale_bytes),
        "first scale chunk must commit");
    require(
        builder.write_rows(base + ".biases", 0U, 2U, first.bias_bytes),
        "first bias chunk must commit");

    bool incomplete_rejected = false;
    try {
      builder.finalize();
    } catch (const std::exception&) {
      incomplete_rejected = true;
    }
    require(incomplete_rejected, "incomplete streamed file must not finalize");

    auto resumed = StreamedSafetensorsBuilder::open(
        output, journal, specs);

    require(
        !resumed.write_rows(base + ".weight", 0U, 2U, first.packed_weight_bytes),
        "exact replay must be idempotent");

    bool gap_rejected = false;
    try {
      (void)resumed.write_rows(
          base + ".weight", 3U, 1U,
          std::span<const std::byte>(
              first.packed_weight_bytes.data(),
              first.packed_weight_bytes.size() / 2U));
    } catch (const std::exception&) {
      gap_rejected = true;
    }
    require(gap_rejected, "non-contiguous chunk must be rejected");

    const auto second_bf16 = bf16_bytes({
        0.25F, 0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F, 2.0F,
        3.0F, 2.5F, 2.0F, 1.5F, 1.0F, 0.5F, 0.0F, -0.5F,
    });
    const auto second = convert_bf16_affine_q4_row_chunk(
        second_bf16, 2U, 8U, 4U);

    require(
        resumed.write_rows(base + ".weight", 2U, 2U, second.packed_weight_bytes),
        "second weight chunk must commit");
    require(
        resumed.write_rows(base + ".scales", 2U, 2U, second.scale_bytes),
        "second scale chunk must commit");
    require(
        resumed.write_rows(base + ".biases", 2U, 2U, second.bias_bytes),
        "second bias chunk must commit");

    require(resumed.state().complete(), "all tensor rows must be complete");
    resumed.finalize();

    SafetensorsReader reader(output);
    require(reader.tensor_names().size() == 3U, "tensor count mismatch");
    require(
        reader.info(base + ".weight").shape ==
            std::vector<std::size_t>({4U, 1U}),
        "weight shape mismatch");
    require(
        reader.info(base + ".scales").shape ==
            std::vector<std::size_t>({4U, 2U}),
        "scale shape mismatch");

    const auto weights = reader.read_u32(base + ".weight");
    const auto scales = reader.read_floats(base + ".scales");
    require(weights.size() == 4U, "packed row count mismatch");
    require(scales.size() == 8U, "scale element count mismatch");
    require(std::isfinite(scales.front()), "scale must be finite");

    {
      std::fstream corrupt(
          output, std::ios::binary | std::ios::in | std::ios::out);
      const auto offset = reader.absolute_offset(base + ".weight");
      corrupt.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
      char original = 0;
      corrupt.read(&original, 1);
      corrupt.clear();
      corrupt.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
      const char changed = static_cast<char>(
          static_cast<unsigned char>(original) ^ 0x01U);
      corrupt.write(&changed, 1);
      corrupt.flush();
    }

    bool corruption_detected = false;
    try {
      (void)StreamedSafetensorsBuilder::open(output, journal, specs);
    } catch (const std::exception&) {
      corruption_detected = true;
    }
    require(corruption_detected, "resume must detect output corruption");

    fs::remove_all(root);
    std::cout
        << "OSM-40B streamed safetensors builder: PASS\n"
        << "  deterministic_layout=PASS\n"
        << "  contiguous_chunk_write=PASS\n"
        << "  resume_journal=PASS\n"
        << "  idempotent_replay=PASS\n"
        << "  gap_guard=PASS\n"
        << "  finalization_guard=PASS\n"
        << "  production_reader_roundtrip=PASS\n"
        << "  corruption_detection=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-40B streamed safetensors builder: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
