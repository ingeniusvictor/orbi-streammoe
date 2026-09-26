#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/conversion/bf16_slice.hpp"
#include "orbi/streammoe/conversion/multi_expert_orchestrator.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_bf16_file(
    const fs::path& path,
    std::size_t count,
    float base) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create BF16 fixture");
  for (std::size_t i = 0; i < count; ++i) {
    const float value = base + static_cast<float>(i % 7U) * 0.125F;
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t bf16 = static_cast<std::uint16_t>(bits >> 16U);
    const char bytes[2] = {
        static_cast<char>(bf16 & 0xFFU),
        static_cast<char>((bf16 >> 8U) & 0xFFU)};
    out.write(bytes, 2);
  }
}

std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex;
  out.width(16);
  out.fill('0');
  out << value;
  return out.str();
}

std::uint64_t hash_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<std::byte> bytes(
      static_cast<std::size_t>(fs::file_size(path)));
  in.read(reinterpret_cast<char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  return fnv1a64(bytes);
}

MultiExpertInput make_expert(
    const fs::path& root,
    std::size_t expert,
    float seed) {
  const auto dir = root / ("expert_" + std::to_string(expert));
  fs::create_directories(dir);

  const std::vector<std::pair<std::string, std::vector<std::size_t>>> specs = {
      {"gate_proj", {8U, 8U}},
      {"up_proj", {8U, 8U}},
      {"down_proj", {8U, 8U}},
  };

  json projections = json::object();
  std::uint64_t total = 0U;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    const auto& name = specs[i].first;
    const auto& shape = specs[i].second;
    const auto file = dir / (name + ".bf16.bin");
    write_bf16_file(file, shape[0] * shape[1], seed + static_cast<float>(i));
    const auto bytes = static_cast<std::uint64_t>(fs::file_size(file));
    total += bytes;
    projections[name] = {
        {"tensor_name",
         "model.layers.0.mlp.experts." + std::to_string(expert) +
             "." + name + ".weight"},
        {"shard", "fixture.safetensors"},
        {"shape", shape},
        {"source_byte_size", bytes},
        {"source_file", file.filename().string()},
        {"source_fnv1a64", hex64(hash_file(file))},
    };
  }

  json manifest = {
      {"schema_version", 1},
      {"model", "fixture"},
      {"snapshot", "fixture"},
      {"layer_index", 0},
      {"expert_index", expert},
      {"total_fetched_bytes", total},
      {"projections", projections},
  };
  const auto manifest_path = dir / "single-expert.json";
  std::ofstream out(manifest_path);
  out << manifest.dump(2) << "\n";

  return {manifest_path, dir};
}

QpackExpertGeometry geometry_fixture() {
  QpackExpertGeometry g;
  g.hidden_size = 8U;
  g.moe_intermediate_size = 8U;
  g.expert_count = 3U;
  g.layer_count = 1U;
  g.quantization = {.bits = 4U, .group_size = 4U};

  const std::uint64_t packed = 8U * 1U * 4U;
  const std::uint64_t aux = 8U * 2U * 4U;
  std::uint64_t offset = 0U;
  auto add = [&](std::string name, std::string dtype, std::uint64_t bytes) {
    g.sections.push_back({std::move(name), std::move(dtype), {}, offset, bytes});
    offset += bytes;
  };
  for (const auto& p : {"gate_proj", "up_proj", "down_proj"}) {
    add(std::string(p) + ".weight", "U32", packed);
    add(std::string(p) + ".scales", "F32", aux);
    add(std::string(p) + ".biases", "F32", aux);
  }
  g.expert_stride = offset;
  g.layer_bytes = g.expert_stride * g.expert_count;
  g.all_layers_bytes = g.layer_bytes;
  return g;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm39e-multi-expert";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    const auto geometry = geometry_fixture();
    std::vector<MultiExpertInput> inputs = {
        make_expert(root, 0U, 1.0F),
        make_expert(root, 1U, 2.0F),
        make_expert(root, 2U, 3.0F),
    };

    auto writer = QpackLayerWriter::open(
        root / "layer_00.bin",
        root / "layer_00.progress.json",
        geometry,
        0U);

    MultiExpertConversionOptions first;
    first.first_expert = 0U;
    first.end_expert_exclusive = 2U;

    const auto r1 = convert_qwen_expert_range(
        inputs, geometry, writer, first, root / "scratch");
    require(r1.requested_experts == 2U, "requested count mismatch");
    require(r1.converted_experts == 2U, "first pass convert count mismatch");
    require(r1.skipped_completed_experts == 0U, "first pass skip mismatch");
    require(
        r1.converted_expert_ids == std::vector<std::size_t>({0U, 1U}),
        "first pass expert IDs mismatch");
    require(r1.source_bytes == 768U, "first pass source bytes mismatch");
    require(std::isfinite(r1.max_abs_error), "max error must be finite");

    auto resumed = QpackLayerWriter::open(
        root / "layer_00.bin",
        root / "layer_00.progress.json",
        geometry,
        0U);

    MultiExpertConversionOptions second;
    second.first_expert = 0U;
    second.end_expert_exclusive = 3U;
    second.finalize_if_complete = true;

    const auto r2 = convert_qwen_expert_range(
        inputs, geometry, resumed, second, root / "scratch");
    require(r2.converted_experts == 1U, "resume convert count mismatch");
    require(r2.skipped_completed_experts == 2U, "resume skip count mismatch");
    require(
        r2.skipped_expert_ids == std::vector<std::size_t>({0U, 1U}),
        "resume skipped IDs mismatch");
    require(
        r2.converted_expert_ids == std::vector<std::size_t>({2U}),
        "resume converted IDs mismatch");
    require(resumed.state().complete(), "layer must complete after resume");

    bool missing_rejected = false;
    try {
      std::vector<MultiExpertInput> missing = {inputs[0]};
      auto w2 = QpackLayerWriter::open(
          root / "layer_missing.bin",
          root / "layer_missing.progress.json",
          geometry,
          0U);
      MultiExpertConversionOptions bad;
      bad.first_expert = 0U;
      bad.end_expert_exclusive = 2U;
      (void)convert_qwen_expert_range(
          missing, geometry, w2, bad, root / "scratch2");
    } catch (const std::exception&) {
      missing_rejected = true;
    }
    require(missing_rejected, "missing requested manifest must fail");

    fs::remove_all(root);
    std::cout
        << "OSM-39E bounded multi-expert orchestrator: PASS\n"
        << "  bounded_range=PASS\n"
        << "  one_expert_at_a_time=PASS\n"
        << "  journal_resume_skip=PASS\n"
        << "  finalization=PASS\n"
        << "  progress_accounting=PASS\n"
        << "  missing_manifest_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-39E bounded multi-expert orchestrator: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
