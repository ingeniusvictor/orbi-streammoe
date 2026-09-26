#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/conversion/qpack_layer_writer.hpp"
#include "orbi/streammoe/conversion/qpack_package.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("unable to write fixture file");
  out << text;
}

std::vector<std::byte> blob(
    std::size_t bytes,
    std::size_t layer,
    std::size_t expert) {
  std::vector<std::byte> out(bytes);
  for (std::size_t i = 0U; i < bytes; ++i) {
    out[i] = static_cast<std::byte>(
        (layer * 31U + expert * 17U + i) & 0xFFU);
  }
  return out;
}

fs::path make_source(const fs::path& root) {
  fs::create_directories(root);
  write_text(
      root / "config.json",
      R"JSON({
  "model_type": "qwen3_next",
  "hidden_size": 8,
  "moe_intermediate_size": 8,
  "num_experts": 3,
  "num_hidden_layers": 4,
  "full_attention_interval": 4
}
)JSON");
  return root;
}

void create_complete_layers(
    const fs::path& source,
    const fs::path& output) {
  const auto geometry = derive_qpack_expert_geometry(
      source,
      QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
  const auto packed = output / "packed_experts";
  fs::create_directories(packed);

  for (std::size_t layer = 0U; layer < geometry.layer_count; ++layer) {
    const auto stem =
        "layer_0" + std::to_string(layer);
    auto writer = QpackLayerWriter::open(
        packed / (stem + ".bin"),
        packed / (stem + ".progress.json"),
        geometry,
        layer);
    for (std::size_t expert = 0U; expert < geometry.expert_count; ++expert) {
      const auto bytes = blob(
          static_cast<std::size_t>(geometry.expert_stride),
          layer,
          expert);
      require(
          writer.write_expert(expert, bytes),
          "fixture expert commit failed");
    }
    writer.finalize();
  }
}

json read_json(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  json root;
  input >> root;
  return root;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm39g-package";
  fs::remove_all(root);

  try {
    const auto source = make_source(root / "source");
    const auto output = root / "package";
    create_complete_layers(source, output);

    // Force OSM-39D backup-journal recovery during package finalization.
    const auto primary =
        output / "packed_experts" / "layer_01.progress.json";
    const auto backup = fs::path(primary.string() + ".bak");
    fs::rename(primary, backup);

    const QpackExpertPackageOptions options{
        .model_name = "qwen3_next",
        .source_checkpoint = "fixture-checkpoint",
        .source_snapshot = "fixture-snapshot",
    };

    const auto plan = derive_qpack_expert_package_plan(
        source,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    require(plan.geometry.layer_count == 4U, "plan layer count mismatch");
    require(plan.geometry.expert_count == 3U, "plan expert count mismatch");
    require(plan.geometry.expert_stride == 480U, "plan expert stride mismatch");
    require(
        plan.linear_layers ==
            std::vector<bool>({true, true, true, false}),
        "D-D-D-G package pattern mismatch");

    const auto first = finalize_qpack_expert_package(
        source,
        output,
        options,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});

    require(first.layer_count == 4U, "finalized layer count mismatch");
    require(first.declared_file_count == 7U, "manifest file count mismatch");
    require(first.expert_payload_bytes == 5760U, "expert payload bytes mismatch");
    require(fs::exists(primary), "backup journal was not recovered");
    require(!fs::exists(backup), "backup journal was not consumed");

    const QpackReader reader(output);
    require(reader.layout().layer_count == 4U, "reader layer count mismatch");
    require(reader.layout().expert_count == 3U, "reader expert count mismatch");
    require(
        reader.layout().linear_layers ==
            std::vector<bool>({true, true, true, false}),
        "reader D-D-D-G pattern mismatch");

    const auto expected = blob(480U, 3U, 2U);
    const auto actual = reader.read_expert(3U, 2U);
    require(actual == expected, "runtime reader expert payload mismatch");

    const auto provenance =
        read_json(output / "conversion-provenance.json");
    require(
        provenance.at("stage").get<std::string>() == "expert-shell",
        "provenance stage mismatch");
    require(
        provenance.at("layers").size() == 4U,
        "provenance layer inventory mismatch");
    require(
        provenance.at("layers").at(0).at("expert_fnv1a64").size() == 3U,
        "provenance expert hash inventory mismatch");

    // Deterministic metadata writes must be idempotent.
    const auto second = finalize_qpack_expert_package(
        source,
        output,
        options,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    require(
        fs::file_size(first.manifest_path) ==
            fs::file_size(second.manifest_path),
        "idempotent manifest size mismatch");

    // A conflicting deterministic metadata file is never overwritten.
    write_text(output / "config.json", "{}\n");
    bool conflict_rejected = false;
    try {
      (void)finalize_qpack_expert_package(
          source,
          output,
          options,
          QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    } catch (const std::exception&) {
      conflict_rejected = true;
    }
    require(conflict_rejected, "conflicting config must be rejected");

    const auto incomplete = root / "incomplete";
    fs::create_directories(incomplete / "packed_experts");
    const auto geometry = derive_qpack_expert_geometry(
        source,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    auto partial = QpackLayerWriter::open(
        incomplete / "packed_experts" / "layer_00.bin",
        incomplete / "packed_experts" / "layer_00.progress.json",
        geometry,
        0U);
    for (std::size_t expert = 0U; expert < geometry.expert_count; ++expert) {
      const auto bytes = blob(
          static_cast<std::size_t>(geometry.expert_stride),
          0U,
          expert);
      (void)partial.write_expert(expert, bytes);
    }
    partial.finalize();

    bool incomplete_rejected = false;
    try {
      (void)finalize_qpack_expert_package(
          source,
          incomplete,
          options,
          QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    } catch (const std::exception&) {
      incomplete_rejected = true;
    }
    require(incomplete_rejected, "incomplete package must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-39G deterministic expert package finalization: PASS\n"
        << "  D-D-D-G_layout=PASS\n"
        << "  complete_journal_verification=PASS\n"
        << "  backup_journal_recovery=PASS\n"
        << "  deterministic_metadata=PASS\n"
        << "  provenance_inventory=PASS\n"
        << "  runtime_QpackReader_validation=PASS\n"
        << "  incomplete_package_guard=PASS\n"
        << "  conflicting_metadata_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-39G deterministic expert package finalization: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
