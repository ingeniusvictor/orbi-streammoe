#include "orbi/streammoe/conversion/qpack_package.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/conversion/qpack_layer_writer.hpp"

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

json read_json(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "QPACK package: unable to open JSON file: " + path.string());
  }
  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("QPACK package: invalid JSON: ") + e.what());
  }
  return root;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "QPACK package: unable to open text file: " + path.string());
  }
  std::ostringstream out;
  out << input.rdbuf();
  if (!input.good() && !input.eof()) {
    throw std::runtime_error(
        "QPACK package: unable to read text file: " + path.string());
  }
  return out.str();
}

void write_exact_or_create(
    const std::filesystem::path& path,
    const std::string& content) {
  if (std::filesystem::exists(path)) {
    if (read_text(path) != content) {
      throw std::runtime_error(
          "QPACK package: existing deterministic file differs: " +
          path.string());
    }
    return;
  }

  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const auto temp = std::filesystem::path(path.string() + ".tmp");
  {
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error(
          "QPACK package: unable to create temp file: " + temp.string());
    }
    output.write(
        content.data(),
        static_cast<std::streamsize>(content.size()));
    output.flush();
    if (!output) {
      throw std::runtime_error(
          "QPACK package: unable to flush temp file: " + temp.string());
    }
  }

  std::error_code ec;
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    std::filesystem::remove(temp);
    throw std::runtime_error(
        "QPACK package: unable to commit deterministic file: " +
        path.string());
  }
}

std::size_t required_size(
    const json& root,
    const char* key) {
  try {
    const auto value = root.at(key).get<std::size_t>();
    if (value == 0U) {
      throw std::runtime_error(
          std::string("QPACK package: zero config field: ") + key);
    }
    return value;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("QPACK package: malformed config field ") +
        key + ": " + e.what());
  }
}

std::string layer_file_name(std::size_t layer) {
  std::ostringstream out;
  out << "layer_" << std::setfill('0') << std::setw(2) << layer << ".bin";
  return out.str();
}

std::string journal_file_name(std::size_t layer) {
  std::ostringstream out;
  out << "layer_" << std::setfill('0') << std::setw(2)
      << layer << ".progress.json";
  return out.str();
}

std::string hex_u64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}

json section_json(const QpackSection& section) {
  return {
      {"name", section.name},
      {"dtype", section.dtype},
      {"shape", section.shape},
      {"offset", section.offset},
      {"size", section.size},
  };
}

std::uint64_t file_size_checked(const std::filesystem::path& path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error(
        "QPACK package: unable to stat file: " + path.string());
  }
  return size;
}

}  // namespace

QpackExpertPackagePlan derive_qpack_expert_package_plan(
    const std::filesystem::path& model_dir,
    QpackExpertQuantizationSpec quantization) {
  const auto config = read_json(model_dir / "config.json");
  if (config.value("model_type", std::string{}) != "qwen3_next") {
    throw std::runtime_error(
        "QPACK package: model_type must be qwen3_next");
  }

  QpackExpertPackagePlan plan;
  plan.geometry =
      derive_qpack_expert_geometry(model_dir, quantization);
  plan.full_attention_interval =
      required_size(config, "full_attention_interval");

  plan.linear_layers.reserve(plan.geometry.layer_count);
  plan.layer_relative_paths.reserve(plan.geometry.layer_count);
  for (std::size_t layer = 0U;
       layer < plan.geometry.layer_count;
       ++layer) {
    plan.linear_layers.push_back(
        ((layer + 1U) % plan.full_attention_interval) != 0U);
    plan.layer_relative_paths.push_back(
        "packed_experts/" + layer_file_name(layer));
  }
  plan.expert_payload_bytes = plan.geometry.all_layers_bytes;
  return plan;
}

QpackExpertPackageResult finalize_qpack_expert_package(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& output_dir,
    const QpackExpertPackageOptions& options,
    QpackExpertQuantizationSpec quantization) {
  if (options.model_name.empty() ||
      options.source_checkpoint.empty() ||
      options.source_snapshot.empty()) {
    throw std::invalid_argument(
        "QPACK package: model/source checkpoint/source snapshot must be non-empty");
  }

  const auto plan =
      derive_qpack_expert_package_plan(model_dir, quantization);
  const auto packed_dir = output_dir / "packed_experts";
  std::filesystem::create_directories(packed_dir);

  json provenance_layers = json::array();

  for (std::size_t layer = 0U;
       layer < plan.geometry.layer_count;
       ++layer) {
    const auto layer_path =
        packed_dir / layer_file_name(layer);
    const auto journal_path =
        packed_dir / journal_file_name(layer);
    const auto journal_backup =
        std::filesystem::path(journal_path.string() + ".bak");

    if (!std::filesystem::exists(layer_path)) {
      throw std::runtime_error(
          "QPACK package: missing converted expert layer: " +
          layer_path.string());
    }
    if (!std::filesystem::exists(journal_path) &&
        !std::filesystem::exists(journal_backup)) {
      throw std::runtime_error(
          "QPACK package: missing layer journal: " +
          journal_path.string());
    }

    auto writer = QpackLayerWriter::open(
        layer_path,
        journal_path,
        plan.geometry,
        layer);
    if (!writer.state().complete()) {
      throw std::runtime_error(
          "QPACK package: layer journal is incomplete: " +
          std::to_string(layer));
    }
    writer.finalize();

    json hashes = json::array();
    for (const auto hash : writer.state().expert_fnv1a64) {
      if (hash == 0U) {
        throw std::runtime_error(
            "QPACK package: completed layer contains zero expert hash");
      }
      hashes.push_back(hex_u64(hash));
    }

    provenance_layers.push_back({
        {"layer", layer},
        {"path", plan.layer_relative_paths[layer]},
        {"bytes", file_size_checked(layer_path)},
        {"expert_fnv1a64", hashes},
    });
  }

  const auto source_config_path = model_dir / "config.json";
  const auto source_config = read_text(source_config_path);
  const auto output_config_path = output_dir / "config.json";
  write_exact_or_create(output_config_path, source_config);

  json sections = json::array();
  for (const auto& section : plan.geometry.sections) {
    sections.push_back(section_json(section));
  }

  const json layout = {
      {"expertCount", plan.geometry.expert_count},
      {"layerCount", plan.geometry.layer_count},
      {"expertStride", plan.geometry.expert_stride},
      {"sections", sections},
      {"linearLayers", plan.linear_layers},
  };
  const auto layout_path = packed_dir / "layout.json";
  const auto layout_text = layout.dump(2) + "\n";
  write_exact_or_create(layout_path, layout_text);

  const json provenance = {
      {"schema_version", 1U},
      {"stage", "expert-shell"},
      {"model_name", options.model_name},
      {"source_checkpoint", options.source_checkpoint},
      {"source_snapshot", options.source_snapshot},
      {"quantization", {
          {"bits", plan.geometry.quantization.bits},
          {"group_size", plan.geometry.quantization.group_size},
          {"mode", "affine"},
      }},
      {"geometry", {
          {"hidden_size", plan.geometry.hidden_size},
          {"moe_intermediate_size", plan.geometry.moe_intermediate_size},
          {"expert_count", plan.geometry.expert_count},
          {"layer_count", plan.geometry.layer_count},
          {"expert_stride", plan.geometry.expert_stride},
          {"layer_bytes", plan.geometry.layer_bytes},
          {"all_layers_bytes", plan.geometry.all_layers_bytes},
          {"full_attention_interval", plan.full_attention_interval},
      }},
      {"layers", provenance_layers},
  };
  const auto provenance_path = output_dir / "conversion-provenance.json";
  const auto provenance_text = provenance.dump(2) + "\n";
  write_exact_or_create(provenance_path, provenance_text);

  json files = json::object();
  files["config.json"] = file_size_checked(output_config_path);
  files["conversion-provenance.json"] =
      file_size_checked(provenance_path);
  files["packed_experts/layout.json"] =
      file_size_checked(layout_path);
  for (std::size_t layer = 0U;
       layer < plan.layer_relative_paths.size();
       ++layer) {
    files[plan.layer_relative_paths[layer]] =
        file_size_checked(
            output_dir / plan.layer_relative_paths[layer]);
  }

  const json manifest = {
      {"magic", std::string(QpackReader::kMagic)},
      {"version", QpackReader::kManifestVersion},
      {"modelName", options.model_name},
      {"sourceCheckpoint", options.source_checkpoint},
      {"quantBits", plan.geometry.quantization.bits},
      {"quantGroupSize", plan.geometry.quantization.group_size},
      {"packageStage", "expert-shell"},
      {"sourceSnapshot", options.source_snapshot},
      {"files", files},
  };
  const auto manifest_path = output_dir / "manifest.json";
  const auto manifest_text = manifest.dump(2) + "\n";
  write_exact_or_create(manifest_path, manifest_text);

  // Runtime compatibility gate: the generated expert shell must be accepted by
  // the same QPACK reader used by inference.
  const QpackReader reader(output_dir);
  if (reader.layout().layer_count != plan.geometry.layer_count ||
      reader.layout().expert_count != plan.geometry.expert_count ||
      reader.layout().expert_stride != plan.geometry.expert_stride) {
    throw std::runtime_error(
        "QPACK package: runtime reader disagrees with finalized geometry");
  }

  QpackExpertPackageResult result;
  result.manifest_path = manifest_path;
  result.layout_path = layout_path;
  result.provenance_path = provenance_path;
  result.layer_count = plan.geometry.layer_count;
  result.declared_file_count = files.size();
  result.expert_payload_bytes = plan.expert_payload_bytes;
  return result;
}

}  // namespace orbi::streammoe
