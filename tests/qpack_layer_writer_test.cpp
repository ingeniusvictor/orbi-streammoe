#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/conversion/qpack_layer_writer.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

QpackExpertGeometry fixture_geometry() {
  QpackExpertGeometry geometry;
  geometry.hidden_size = 8U;
  geometry.moe_intermediate_size = 8U;
  geometry.expert_count = 3U;
  geometry.layer_count = 4U;
  geometry.quantization = {.bits = 4U, .group_size = 4U};
  geometry.expert_stride = 16U;
  geometry.layer_bytes = 48U;
  geometry.all_layers_bytes = 192U;
  geometry.sections = {
      {"payload", "U8", {16U}, 0U, 16U},
      {"dummy1", "U8", {1U}, 0U, 1U},
      {"dummy2", "U8", {1U}, 1U, 1U},
      {"dummy3", "U8", {1U}, 2U, 1U},
      {"dummy4", "U8", {1U}, 3U, 1U},
      {"dummy5", "U8", {1U}, 4U, 1U},
      {"dummy6", "U8", {1U}, 5U, 1U},
      {"dummy7", "U8", {1U}, 6U, 1U},
      {"dummy8", "U8", {1U}, 7U, 1U},
  };
  return geometry;
}

std::vector<std::byte> expert_blob(std::uint8_t seed) {
  std::vector<std::byte> out(16U);
  for (std::size_t i = 0U; i < out.size(); ++i) {
    out[i] = static_cast<std::byte>(seed + static_cast<std::uint8_t>(i));
  }
  return out;
}

std::vector<std::byte> read_all(const fs::path& path) {
  const auto size = fs::file_size(path);
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  input.read(
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  require(
      input.gcount() == static_cast<std::streamsize>(bytes.size()),
      "short layer fixture read");
  return bytes;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm39d-layer-writer";
  fs::remove_all(root);
  fs::create_directories(root);

  try {
    const auto geometry = fixture_geometry();
    const auto layer = root / "layer_00.bin";
    const auto journal = root / "layer_00.progress.json";

    auto writer = QpackLayerWriter::open(
        layer,
        journal,
        geometry,
        0U);

    require(fs::file_size(layer) == 48U, "exact layer size mismatch");
    require(writer.state().completed_experts == 0U, "fresh journal must be empty");

    const auto e0 = expert_blob(10U);
    const auto e1 = expert_blob(40U);
    const auto e2 = expert_blob(70U);

    require(writer.write_expert(2U, e2), "expert 2 first commit failed");
    require(writer.write_expert(0U, e0), "expert 0 first commit failed");
    require(
        writer.state().completed_experts == 2U,
        "two experts should be committed");

    bool incomplete_rejected = false;
    try {
      writer.finalize();
    } catch (const std::exception&) {
      incomplete_rejected = true;
    }
    require(incomplete_rejected, "incomplete layer must not finalize");

    auto resumed = QpackLayerWriter::open(
        layer,
        journal,
        geometry,
        0U);
    require(
        resumed.state().completed_experts == 2U,
        "resume must restore completed experts");
    require(
        !resumed.write_expert(0U, e0),
        "identical completed expert must be idempotent");

    bool overwrite_rejected = false;
    auto wrong_e0 = e0;
    wrong_e0.front() = std::byte{0xFF};
    try {
      (void)resumed.write_expert(0U, wrong_e0);
    } catch (const std::exception&) {
      overwrite_rejected = true;
    }
    require(
        overwrite_rejected,
        "completed expert with different bytes must be rejected");

    require(resumed.write_expert(1U, e1), "expert 1 commit failed");
    resumed.finalize();
    require(resumed.state().complete(), "final state must be complete");

    const auto bytes = read_all(layer);
    require(
        std::equal(e0.begin(), e0.end(), bytes.begin()),
        "expert 0 fixed offset mismatch");
    require(
        std::equal(e1.begin(), e1.end(), bytes.begin() + 16U),
        "expert 1 fixed offset mismatch");
    require(
        std::equal(e2.begin(), e2.end(), bytes.begin() + 32U),
        "expert 2 fixed offset mismatch");

    auto verified = QpackLayerWriter::open(
        layer,
        journal,
        geometry,
        0U);
    verified.finalize();

    const auto backup = fs::path(journal.string() + ".bak");
    fs::rename(journal, backup);
    auto recovered = QpackLayerWriter::open(
        layer,
        journal,
        geometry,
        0U);
    recovered.finalize();
    require(fs::exists(journal), "backup journal must be restored");
    require(!fs::exists(backup), "recovered backup journal must be consumed");

    {
      std::fstream corrupt(
          layer,
          std::ios::binary | std::ios::in | std::ios::out);
      corrupt.seekp(16U, std::ios::beg);
      const char byte = 0;
      corrupt.write(&byte, 1);
    }

    bool corruption_detected = false;
    try {
      (void)QpackLayerWriter::open(
          layer,
          journal,
          geometry,
          0U);
    } catch (const std::exception&) {
      corruption_detected = true;
    }
    require(
        corruption_detected,
        "resume must detect committed expert corruption");

    fs::remove_all(root);
    std::cout
        << "OSM-39D resumable QPACK layer writer: PASS\n"
        << "  exact_layer_size=PASS\n"
        << "  fixed_stride_offsets=PASS\n"
        << "  out_of_order_commits=PASS\n"
        << "  resume_journal=PASS\n"
        << "  backup_journal_recovery=PASS\n"
        << "  idempotent_replay=PASS\n"
        << "  overwrite_guard=PASS\n"
        << "  readback_checksum=PASS\n"
        << "  corruption_detection=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-39D resumable QPACK layer writer: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
