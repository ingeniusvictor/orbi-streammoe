#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/conversion/bf16_slice.hpp"
#include "orbi/streammoe/conversion/qpack_plan.hpp"
#include "orbi/streammoe/conversion/streamed_safetensors_builder.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> read_all(const fs::path& path) {
  std::vector<std::byte> bytes(
      static_cast<std::size_t>(fs::file_size(path)));
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("short official global row read");
  }
  return bytes;
}

const QpackConversionPlanEntry& find_entry(
    const QpackConversionPlan& plan,
    const std::string& name) {
  for (const auto& entry : plan.entries) {
    if (entry.source_tensor == name) return entry;
  }
  throw std::runtime_error("missing conversion plan entry: " + name);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_streamed_global_rows_tests "
          "<metadata-dir> <row-chunk-dir>");
    }

    const fs::path metadata_dir(argv[1]);
    const fs::path chunk_dir(argv[2]);
    const auto plan = build_qpack_conversion_plan(metadata_dir);

    std::ifstream input(chunk_dir / "global-row-chunks.json");
    json root;
    input >> root;
    require(root.at("schema_version").get<unsigned>() == 1U, "schema mismatch");
    require(root.at("rows").get<std::size_t>() == 2U, "row count mismatch");
    require(
        root.at("total_fetched_bytes").get<std::uint64_t>() == 16384ULL,
        "expected exactly 16 KiB of official BF16 source rows");

    std::size_t verified = 0U;
    for (const auto& item : root.at("tensors")) {
      const auto name = item.at("source_tensor").get<std::string>();
      const auto shape = item.at("source_shape").get<std::vector<std::size_t>>();
      const auto rows = item.at("rows").get<std::size_t>();
      const auto cols = item.at("cols").get<std::size_t>();
      require(shape.size() == 2U, "official global tensor must be rank 2");
      require(shape[1] == 2048U, "official hidden size changed");
      require(rows == 2U && cols == 2048U, "official row chunk geometry mismatch");

      const auto& entry = find_entry(plan, name);
      require(
          entry.tensor_class == QpackConversionClass::global_dense,
          "official row chunk must map to global_dense");
      require(
          entry.action == QpackConversionAction::affine_quantize,
          "embed/lm-head must use affine quantization");

      const auto bytes = read_all(
          chunk_dir / item.at("source_file").get<std::string>());
      require(
          fnv1a64(bytes) ==
              std::stoull(
                  item.at("source_fnv1a64").get<std::string>(),
                  nullptr,
                  16),
          "official source row FNV mismatch");

      const auto converted = convert_bf16_affine_q4_row_chunk(
          bytes, rows, cols, 64U);
      require(converted.packed_cols == 256U, "packed column count mismatch");
      require(converted.groups_per_row == 32U, "group count mismatch");
      require(
          converted.packed_weight_bytes.size() == 2048U,
          "official packed row chunk byte count mismatch");
      require(
          converted.scale_bytes.size() == 256U &&
              converted.bias_bytes.size() == 256U,
          "official affine metadata byte count mismatch");
      ++verified;
    }

    require(verified == 2U, "expected embed_tokens and lm_head row chunks");
    std::cout
        << "OSM-40B official streamed global rows: PASS\n"
        << "  tensors=2\n"
        << "  rows_per_tensor=2\n"
        << "  source_bytes=16384\n"
        << "  hidden_size=2048\n"
        << "  affine_q4_chunking=PASS\n"
        << "  bounded_network_fetch=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-40B official streamed global rows: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
