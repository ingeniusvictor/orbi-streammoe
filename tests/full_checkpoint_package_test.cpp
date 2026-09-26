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
#include "orbi/streammoe/container/safetensors.hpp"
#include "orbi/streammoe/conversion/full_checkpoint_package.hpp"
#include "orbi/streammoe/conversion/qpack_layer_writer.hpp"
#include "orbi/streammoe/conversion/qpack_package.hpp"
#include "orbi/streammoe/conversion/streamed_safetensors_builder.hpp"

using namespace orbi::streammoe;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& value) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("unable to write fixture");
  out << value;
}

json read_json(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("unable to read fixture JSON");
  json root;
  input >> root;
  return root;
}

fs::path make_source(const fs::path& root) {
  fs::create_directories(root);
  write_text(
      root / "config.json",
      R"JSON({
  "model_type": "qwen3_next",
  "hidden_size": 8,
  "vocab_size": 4,
  "tie_word_embeddings": false,
  "num_hidden_layers": 4,
  "full_attention_interval": 4,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "head_dim": 8,
  "linear_num_value_heads": 1,
  "linear_num_key_heads": 1,
  "linear_key_head_dim": 8,
  "linear_value_head_dim": 8,
  "linear_conv_kernel_dim": 2,
  "num_experts": 3,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 8,
  "shared_expert_intermediate_size": 8,
  "norm_topk_prob": true
}
)JSON");
  return root;
}

std::vector<std::byte> expert_blob(
    std::size_t bytes,
    std::size_t layer,
    std::size_t expert) {
  std::vector<std::byte> out(bytes);
  for (std::size_t i = 0; i < bytes; ++i) {
    out[i] = static_cast<std::byte>(
        (layer * 29U + expert * 13U + i) & 0xFFU);
  }
  return out;
}

void build_expert_shell(
    const fs::path& source,
    const fs::path& output) {
  const auto geometry = derive_qpack_expert_geometry(
      source,
      QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
  const auto packed = output / "packed_experts";
  fs::create_directories(packed);

  for (std::size_t layer = 0U; layer < geometry.layer_count; ++layer) {
    const auto stem = "layer_0" + std::to_string(layer);
    auto writer = QpackLayerWriter::open(
        packed / (stem + ".bin"),
        packed / (stem + ".progress.json"),
        geometry,
        layer);
    for (std::size_t expert = 0U; expert < geometry.expert_count; ++expert) {
      const auto bytes = expert_blob(
          static_cast<std::size_t>(geometry.expert_stride),
          layer,
          expert);
      (void)writer.write_expert(expert, bytes);
    }
    writer.finalize();
  }

  const QpackExpertPackageOptions options{
      .model_name = "qwen3_next",
      .source_checkpoint = "fixture-checkpoint",
      .source_snapshot = "fixture-snapshot",
  };
  (void)finalize_qpack_expert_package(
      source,
      output,
      options,
      QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
}

std::vector<StreamedSafetensorSpec> dense_specs() {
  return {
      {"lm_head.biases", "F32", {4U, 2U}},
      {"lm_head.scales", "F32", {4U, 2U}},
      {"lm_head.weight", "U32", {4U, 1U}},
      {"model.embed_tokens.biases", "F32", {4U, 2U}},
      {"model.embed_tokens.scales", "F32", {4U, 2U}},
      {"model.embed_tokens.weight", "U32", {4U, 1U}},
      {"model.norm.weight", "F32", {8U}},
  };
}

void complete_dense(
    const fs::path& output,
    const fs::path& journal,
    bool leave_last_incomplete) {
  auto builder = StreamedSafetensorsBuilder::open(
      output,
      journal,
      dense_specs());

  const auto& tensors = builder.state().tensors;
  for (std::size_t i = 0U; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const auto rows = tensor.spec.shape.front();
    const auto commit_rows =
        leave_last_incomplete && i + 1U == tensors.size()
            ? rows - 1U
            : rows;
    if (commit_rows == 0U) continue;

    const auto bytes_count =
        static_cast<std::size_t>(tensor.row_bytes) * commit_rows;
    std::vector<std::byte> bytes(bytes_count);
    for (std::size_t j = 0U; j < bytes.size(); ++j) {
      bytes[j] = static_cast<std::byte>((i * 19U + j + 1U) & 0xFFU);
    }
    (void)builder.write_rows(
        tensor.spec.name,
        0U,
        commit_rows,
        bytes);
  }
}

void finish_last_dense_row(
    const fs::path& output,
    const fs::path& journal) {
  auto builder = StreamedSafetensorsBuilder::open(
      output,
      journal,
      dense_specs());
  for (const auto& tensor : builder.state().tensors) {
    const auto total = tensor.spec.shape.front();
    if (tensor.completed_rows == total) continue;
    require(
        tensor.completed_rows + 1U == total,
        "fixture expects exactly one missing final row");
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(tensor.row_bytes),
        std::byte{0x5A});
    (void)builder.write_rows(
        tensor.spec.name,
        tensor.completed_rows,
        1U,
        bytes);
  }
  builder.finalize();
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm40f-full-checkpoint";
  fs::remove_all(root);

  try {
    const auto source = make_source(root / "source");
    const auto output = root / "package";
    build_expert_shell(source, output);

    const auto dense = output / "model.safetensors";
    const auto journal = output / "model.progress.json";
    complete_dense(dense, journal, true);

    bool incomplete_rejected = false;
    try {
      (void)finalize_full_qpack_checkpoint(
          output,
          journal,
          QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    } catch (const std::exception&) {
      incomplete_rejected = true;
    }
    require(
        incomplete_rejected,
        "incomplete dense journal must block full checkpoint promotion");

    finish_last_dense_row(dense, journal);

    const auto first = finalize_full_qpack_checkpoint(
        output,
        journal,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});

    require(first.dense_tensor_count == 7U, "dense tensor count mismatch");
    require(first.dense_file_bytes == fs::file_size(dense), "dense byte mismatch");
    require(first.expert_payload_bytes == 5760U, "expert payload mismatch");

    const auto manifest = read_json(output / "manifest.json");
    require(
        manifest.at("packageStage").get<std::string>() == "full-checkpoint",
        "manifest stage mismatch");
    require(
        manifest.at("files").contains("model.safetensors"),
        "manifest must declare dense payload");
    require(
        manifest.at("files").contains("full-checkpoint-provenance.json"),
        "manifest must declare full checkpoint provenance");

    const auto config = read_json(output / "config.json");
    require(
        config.at("quantization").at("bits").get<std::uint32_t>() == 4U,
        "runtime quantization bits mismatch");
    require(
        config.at("quantization").at("group_size").get<std::uint32_t>() == 4U,
        "runtime quantization group mismatch");

    const auto provenance =
        read_json(output / "full-checkpoint-provenance.json");
    require(
        provenance.at("stage").get<std::string>() == "full-checkpoint",
        "full provenance stage mismatch");
    require(
        provenance.at("dense").at("tensor_count").get<std::size_t>() == 7U,
        "full provenance dense inventory mismatch");

    QpackReader reader(output);
    require(
        reader.layout().layer_count == 4U,
        "runtime reader layer count mismatch");
    require(
        reader.manifest().files.count("model.safetensors") == 1U,
        "runtime reader must validate dense file declaration");

    const auto second = finalize_full_qpack_checkpoint(
        output,
        journal,
        QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    require(
        second.dense_file_bytes == first.dense_file_bytes,
        "idempotent finalization changed dense byte count");

    bool quantization_conflict = false;
    auto conflicted = read_json(output / "config.json");
    conflicted["quantization"]["group_size"] = 8U;
    write_text(output / "config.json", conflicted.dump(2) + "\n");
    try {
      (void)finalize_full_qpack_checkpoint(
          output,
          journal,
          QpackExpertQuantizationSpec{.bits = 4U, .group_size = 4U});
    } catch (const std::exception&) {
      quantization_conflict = true;
    }
    require(
        quantization_conflict,
        "conflicting runtime quantization must be rejected");

    fs::remove_all(root);
    std::cout
        << "OSM-40F full checkpoint package readiness: PASS\n"
        << "  expert_shell_gate=PASS\n"
        << "  dense_journal_completion=PASS\n"
        << "  dense_readback_verification=PASS\n"
        << "  runtime_quantization_promotion=PASS\n"
        << "  manifest_full_checkpoint_promotion=PASS\n"
        << "  full_provenance=PASS\n"
        << "  runtime_reopen=PASS\n"
        << "  idempotent_finalization=PASS\n"
        << "  quantization_conflict_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-40F full checkpoint package readiness: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
