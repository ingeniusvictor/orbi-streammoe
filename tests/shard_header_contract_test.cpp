#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/model/shard_header_contract.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write fixture");
  out << text;
}

std::string valid_manifest() {
  return R"JSON({
    "schema_version": 1,
    "model": "fixture/qwen",
    "snapshot": "fixture-snapshot",
    "selected_tensors": ["tensor.a", "tensor.c"],
    "shards": [
      {
        "filename": "model-00001-of-00002.safetensors",
        "file_size": 1000,
        "header_size": 100,
        "fetched_bytes": 108,
        "tensor_count": 2,
        "selected_tensor_metadata": {
          "tensor.a": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
          }
        }
      },
      {
        "filename": "model-00002-of-00002.safetensors",
        "file_size": 500,
        "header_size": 80,
        "fetched_bytes": 88,
        "tensor_count": 1,
        "selected_tensor_metadata": {
          "tensor.c": {
            "dtype": "U32",
            "shape": [2],
            "data_offsets": [0, 8]
          }
        }
      }
    ],
    "total_fetched_bytes": 196
  })JSON";
}

QwenShardedCheckpointContract fixture_checkpoint() {
  QwenShardedCheckpointContract checkpoint;
  checkpoint.shards = {
      "model-00001-of-00002.safetensors",
      "model-00002-of-00002.safetensors"};
  checkpoint.shard_count = checkpoint.shards.size();
  checkpoint.weight_map.emplace(
      "tensor.a", "model-00001-of-00002.safetensors");
  checkpoint.weight_map.emplace(
      "tensor.c", "model-00002-of-00002.safetensors");
  return checkpoint;
}

void require_rejected(
    const fs::path& path,
    const std::string& text,
    const std::string& label) {
  write_text(path, text);
  bool rejected = false;
  try {
    (void)inspect_safetensors_range_manifest(path);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, label + " must be rejected");
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm38c-range-contract";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    const auto manifest_path = root / "manifest.json";
    write_text(manifest_path, valid_manifest());

    const auto manifest =
        inspect_safetensors_range_manifest(manifest_path);
    require(manifest.model == "fixture/qwen", "model mismatch");
    require(manifest.snapshot == "fixture-snapshot", "snapshot mismatch");
    require(manifest.shards.size() == 2U, "shard count mismatch");
    require(manifest.tensors.size() == 2U, "tensor probe count mismatch");
    require(manifest.total_fetched_bytes == 196U, "fetched byte sum mismatch");

    const auto& a = manifest.tensor("tensor.a");
    require(a.dtype == "F32", "tensor.a dtype mismatch");
    require(a.shape == std::vector<std::size_t>({2U, 2U}), "tensor.a shape mismatch");
    require(a.byte_size() == 16U, "tensor.a byte size mismatch");

    validate_qwen_shard_range_pilot(manifest, fixture_checkpoint());

    auto wrong = fixture_checkpoint();
    wrong.weight_map["tensor.a"] = "model-00002-of-00002.safetensors";
    bool mapping_rejected = false;
    try {
      validate_qwen_shard_range_pilot(manifest, wrong);
    } catch (const std::exception&) {
      mapping_rejected = true;
    }
    require(mapping_rejected, "wrong weight_map shard must be rejected");

    auto bad_fetch = valid_manifest();
    const auto fetched_pos = bad_fetch.find("\"fetched_bytes\": 108");
    require(fetched_pos != std::string::npos, "fixture fetched_bytes marker missing");
    bad_fetch.replace(
        fetched_pos,
        std::string("\"fetched_bytes\": 108").size(),
        "\"fetched_bytes\": 109");
    require_rejected(root / "bad-fetch.json", bad_fetch, "bad fetched byte count");

    auto bad_range = valid_manifest();
    const auto range_pos = bad_range.find("\"data_offsets\": [0, 16]");
    require(range_pos != std::string::npos, "fixture range marker missing");
    bad_range.replace(
        range_pos,
        std::string("\"data_offsets\": [0, 16]").size(),
        "\"data_offsets\": [0, 12]");
    require_rejected(root / "bad-range.json", bad_range, "dtype/shape mismatch");

    auto duplicate = valid_manifest();
    const auto selected_pos =
        duplicate.find("\"selected_tensors\": [\"tensor.a\", \"tensor.c\"]");
    require(selected_pos != std::string::npos, "fixture selected marker missing");
    duplicate.replace(
        selected_pos,
        std::string("\"selected_tensors\": [\"tensor.a\", \"tensor.c\"]").size(),
        "\"selected_tensors\": [\"tensor.a\", \"tensor.a\"]");
    require_rejected(root / "duplicate.json", duplicate, "duplicate selected tensor");

    fs::remove_all(root);
    std::cout
        << "OSM-38C safetensors range manifest contract: PASS\n"
        << "  strict_header_bytes=PASS\n"
        << "  dtype_shape_ranges=PASS\n"
        << "  weight_map_mapping=PASS\n"
        << "  malformed_guards=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-38C safetensors range manifest contract: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
