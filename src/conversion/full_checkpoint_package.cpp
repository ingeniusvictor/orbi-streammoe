#include "orbi/streammoe/conversion/full_checkpoint_package.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/container/safetensors.hpp"
#include "orbi/streammoe/conversion/streamed_safetensors_builder.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

json read_json(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "full checkpoint package: unable to open JSON: " + path.string());
  }
  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("full checkpoint package: invalid JSON: ") + e.what());
  }
  return root;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "full checkpoint package: unable to open text: " + path.string());
  }
  std::string value(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  if (!input.good() && !input.eof()) {
    throw std::runtime_error(
        "full checkpoint package: unable to read text: " + path.string());
  }
  return value;
}

void replace_if_different(
    const std::filesystem::path& path,
    const std::string& content) {
  if (std::filesystem::exists(path) && read_text(path) == content) {
    return;
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const auto temp = std::filesystem::path(path.string() + ".tmp");
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error(
          "full checkpoint package: unable to create temp file");
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.flush();
    if (!out) {
      throw std::runtime_error(
          "full checkpoint package: unable to flush temp file");
    }
  }
  std::error_code ec;
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    // Windows cannot atomically replace an existing file with rename.
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(temp, path, ec);
  }
  if (ec) {
    std::filesystem::remove(temp);
    throw std::runtime_error(
        "full checkpoint package: unable to commit deterministic file");
  }
}

std::uint64_t file_size_checked(const std::filesystem::path& path) {
  std::error_code ec;
  const auto value = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error(
        "full checkpoint package: unable to stat file: " + path.string());
  }
  return value;
}

std::vector<StreamedSafetensorSpec> specs_from_dense(
    const SafetensorsReader& reader) {
  std::vector<StreamedSafetensorSpec> specs;
  for (const auto& name : reader.tensor_names()) {
    const auto& info = reader.info(name);
    specs.push_back({name, info.dtype, info.shape});
  }
  if (specs.empty()) {
    throw std::runtime_error(
        "full checkpoint package: dense tensor inventory is empty");
  }
  return specs;
}

json quantization_json(QpackExpertQuantizationSpec quantization) {
  return {
      {"bits", quantization.bits},
      {"group_size", quantization.group_size},
      {"mode", "affine"},
  };
}

void promote_runtime_config(
    const std::filesystem::path& path,
    QpackExpertQuantizationSpec quantization) {
  auto root = read_json(path);
  const auto expected = quantization_json(quantization);
  if (root.contains("quantization")) {
    if (root.at("quantization") != expected) {
      throw std::runtime_error(
          "full checkpoint package: existing runtime quantization config conflicts");
    }
  } else {
    root["quantization"] = expected;
  }
  replace_if_different(path, root.dump(2) + "\n");
}

std::uint64_t expert_payload_bytes(const QpackReader& qpack) {
  const auto& layout = qpack.layout();
  return static_cast<std::uint64_t>(layout.layer_count) *
         static_cast<std::uint64_t>(layout.expert_count) *
         layout.expert_stride;
}

}  // namespace

FullCheckpointPackageResult finalize_full_qpack_checkpoint(
    const std::filesystem::path& output_dir,
    const std::filesystem::path& dense_journal_path,
    QpackExpertQuantizationSpec quantization) {
  if (quantization.bits != 4U || quantization.group_size == 0U) {
    throw std::invalid_argument(
        "full checkpoint package: requires affine Q4 quantization");
  }

  // The existing manifest must already be a valid expert shell (or an
  // idempotently re-opened full checkpoint) before dense promotion.
  const QpackReader expert_reader(output_dir);
  const auto expert_bytes = expert_payload_bytes(expert_reader);

  const auto dense_path = output_dir / "model.safetensors";
  if (!std::filesystem::exists(dense_path)) {
    throw std::runtime_error(
        "full checkpoint package: model.safetensors is missing");
  }

  const SafetensorsReader initial_dense(dense_path);
  auto builder = StreamedSafetensorsBuilder::open(
      dense_path,
      dense_journal_path,
      specs_from_dense(initial_dense));
  if (!builder.state().complete()) {
    throw std::runtime_error(
        "full checkpoint package: dense conversion journal is incomplete");
  }
  builder.finalize();

  const auto config_path = output_dir / "config.json";
  promote_runtime_config(config_path, quantization);

  const SafetensorsReader verified_dense(dense_path);
  const auto tensor_names = verified_dense.tensor_names();
  const auto dense_bytes = file_size_checked(dense_path);

  const auto provenance_path =
      output_dir / "full-checkpoint-provenance.json";
  const json provenance = {
      {"schema_version", 1U},
      {"stage", "full-checkpoint"},
      {"expert_provenance", "conversion-provenance.json"},
      {"dense", {
          {"path", "model.safetensors"},
          {"bytes", dense_bytes},
          {"tensor_count", tensor_names.size()},
          {"journal", std::filesystem::relative(
              dense_journal_path, output_dir).generic_string()},
      }},
      {"quantization", quantization_json(quantization)},
      {"expert_payload_bytes", expert_bytes},
  };
  replace_if_different(provenance_path, provenance.dump(2) + "\n");

  const auto manifest_path = output_dir / "manifest.json";
  auto manifest = read_json(manifest_path);
  const auto stage = manifest.value("packageStage", std::string{});
  if (!stage.empty() &&
      stage != "expert-shell" &&
      stage != "full-checkpoint") {
    throw std::runtime_error(
        "full checkpoint package: unsupported packageStage: " + stage);
  }
  if (!manifest.contains("files") || !manifest.at("files").is_object()) {
    throw std::runtime_error(
        "full checkpoint package: manifest.files must be an object");
  }
  manifest["packageStage"] = "full-checkpoint";
  manifest["files"]["config.json"] = file_size_checked(config_path);
  manifest["files"]["model.safetensors"] = dense_bytes;
  manifest["files"]["full-checkpoint-provenance.json"] =
      file_size_checked(provenance_path);
  replace_if_different(manifest_path, manifest.dump(2) + "\n");

  // Final production-runtime gate.
  const QpackReader full_reader(output_dir);
  QpackMlxCheckpoint checkpoint(full_reader);
  const auto config = parse_qwen3_next_dense_config(full_reader);
  if (config.num_hidden_layers != full_reader.layout().layer_count) {
    throw std::runtime_error(
        "full checkpoint package: dense config layer count disagrees with experts");
  }

  if (!checkpoint.dense().contains("model.norm.weight") ||
      !checkpoint.is_quantized("model.embed_tokens") ||
      !checkpoint.is_quantized("lm_head")) {
    throw std::runtime_error(
        "full checkpoint package: required global dense inventory is incomplete");
  }
  const auto embedding_quant =
      checkpoint.quant_spec_for("model.embed_tokens");
  const auto lm_head_quant =
      checkpoint.quant_spec_for("lm_head");
  if (!embedding_quant.has_value() || !lm_head_quant.has_value() ||
      embedding_quant->bits != quantization.bits ||
      lm_head_quant->bits != quantization.bits ||
      embedding_quant->group_size != quantization.group_size ||
      lm_head_quant->group_size != quantization.group_size) {
    throw std::runtime_error(
        "full checkpoint package: runtime global quantization metadata mismatch");
  }

  FullCheckpointPackageResult result;
  result.manifest_path = manifest_path;
  result.config_path = config_path;
  result.dense_path = dense_path;
  result.provenance_path = provenance_path;
  result.dense_tensor_count = tensor_names.size();
  result.dense_file_bytes = dense_bytes;
  result.expert_payload_bytes = expert_bytes;
  return result;
}

}  // namespace orbi::streammoe
