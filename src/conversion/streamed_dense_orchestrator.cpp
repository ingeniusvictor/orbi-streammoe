#include "orbi/streammoe/conversion/streamed_dense_orchestrator.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/conversion/bf16_slice.hpp"

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::uint64_t parse_hex_u64(const std::string& value) {
  if (value.size() != 16U) {
    throw std::runtime_error(
        "streamed dense orchestrator: source hash must be 16 hex digits");
  }
  std::uint64_t out = 0U;
  for (const char ch : value) {
    std::uint64_t digit{};
    if (ch >= '0' && ch <= '9') digit = static_cast<std::uint64_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f') digit = 10U + static_cast<std::uint64_t>(ch - 'a');
    else if (ch >= 'A' && ch <= 'F') digit = 10U + static_cast<std::uint64_t>(ch - 'A');
    else throw std::runtime_error(
        "streamed dense orchestrator: invalid source hash");
    out = (out << 4U) | digit;
  }
  return out;
}

std::uint64_t checked_mul(
    std::uint64_t a,
    std::uint64_t b,
    const char* label) {
  if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
    throw std::runtime_error(
        std::string("streamed dense orchestrator: overflow: ") + label);
  }
  return a * b;
}

std::uint64_t row_elements(const std::vector<std::size_t>& shape) {
  if (shape.empty() || shape.front() == 0U) {
    throw std::runtime_error(
        "streamed dense orchestrator: source shape must contain rows");
  }
  std::uint64_t out = 1U;
  for (std::size_t i = 1U; i < shape.size(); ++i) {
    if (shape[i] == 0U) {
      throw std::runtime_error(
          "streamed dense orchestrator: zero source dimension");
    }
    out = checked_mul(
        out,
        static_cast<std::uint64_t>(shape[i]),
        "row elements");
  }
  return out;
}

const QpackConversionPlanEntry& dense_entry(
    const QpackConversionPlan& plan,
    const std::string& source_tensor) {
  const auto it = std::lower_bound(
      plan.entries.begin(),
      plan.entries.end(),
      source_tensor,
      [](const auto& entry, const std::string& name) {
        return entry.source_tensor < name;
      });
  if (it == plan.entries.end() || it->source_tensor != source_tensor) {
    throw std::runtime_error(
        "streamed dense orchestrator: source absent from conversion plan: " +
        source_tensor);
  }
  if (it->tensor_class == QpackConversionClass::routed_expert ||
      it->tensor_class == QpackConversionClass::auxiliary_mtp ||
      it->target_file != "model.safetensors") {
    throw std::runtime_error(
        "streamed dense orchestrator: source is not dense/global: " +
        source_tensor);
  }
  if (it->action != QpackConversionAction::copy_bf16_to_f32 &&
      it->action != QpackConversionAction::affine_quantize) {
    throw std::runtime_error(
        "streamed dense orchestrator: unsupported action for " +
        source_tensor);
  }
  return *it;
}

std::vector<std::byte> read_exact_source(
    const std::filesystem::path& path,
    std::uint64_t expected,
    std::uint64_t hash) {
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec || actual != expected ||
      expected > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "streamed dense orchestrator: source file size mismatch");
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(expected));
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "streamed dense orchestrator: unable to open source chunk");
  }
  input.read(
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error(
        "streamed dense orchestrator: short source read");
  }
  if (fnv1a64(bytes) != hash) {
    throw std::runtime_error(
        "streamed dense orchestrator: source chunk checksum mismatch");
  }
  return bytes;
}

std::vector<std::byte> f32_bytes(std::span<const float> values) {
  std::vector<std::byte> out(values.size() * sizeof(float));
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      out[i * 4U + lane] = static_cast<std::byte>(
          (bits >> (lane * 8U)) & 0xFFU);
    }
  }
  return out;
}

const StreamedSafetensorState& target_state(
    const StreamedSafetensorsBuilder& builder,
    const std::string& name) {
  const auto& tensors = builder.state().tensors;
  const auto it = std::find_if(
      tensors.begin(), tensors.end(),
      [&](const auto& item) { return item.spec.name == name; });
  if (it == tensors.end()) {
    throw std::runtime_error(
        "streamed dense orchestrator: builder missing target tensor: " + name);
  }
  return *it;
}

std::size_t completed_rows_for_source(
    const StreamedSafetensorsBuilder& builder,
    const QpackConversionPlanEntry& entry) {
  if (entry.action == QpackConversionAction::copy_bf16_to_f32) {
    return target_state(builder, entry.target_path).completed_rows;
  }

  const auto weight =
      target_state(builder, entry.target_path + ".weight").completed_rows;
  const auto scales =
      target_state(builder, entry.target_path + ".scales").completed_rows;
  const auto biases =
      target_state(builder, entry.target_path + ".biases").completed_rows;
  if (weight != scales || weight != biases) {
    throw std::runtime_error(
        "streamed dense orchestrator: affine target progress diverged");
  }
  return weight;
}

}  // namespace

StreamedDenseManifest inspect_streamed_dense_manifest(
    const std::filesystem::path& manifest_path) {
  std::ifstream input(manifest_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "streamed dense orchestrator: unable to open manifest");
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("streamed dense orchestrator: invalid JSON: ") + e.what());
  }
  if (root.value("schema_version", 0U) != 1U) {
    throw std::runtime_error(
        "streamed dense orchestrator: unsupported manifest schema");
  }

  StreamedDenseManifest out;
  try {
    out.model = root.at("model").get<std::string>();
    out.snapshot = root.at("snapshot").get<std::string>();
    for (const auto& tensor_value : root.at("tensors")) {
      StreamedDenseSourceTensor tensor;
      tensor.source_tensor =
          tensor_value.at("source_tensor").get<std::string>();
      tensor.source_shape =
          tensor_value.at("source_shape").get<std::vector<std::size_t>>();
      for (const auto& chunk_value : tensor_value.at("chunks")) {
        StreamedDenseSourceChunk chunk;
        chunk.first_row = chunk_value.at("first_row").get<std::size_t>();
        chunk.row_count = chunk_value.at("row_count").get<std::size_t>();
        chunk.source_file =
            chunk_value.at("source_file").get<std::string>();
        chunk.source_byte_size =
            chunk_value.at("source_byte_size").get<std::uint64_t>();
        chunk.source_fnv1a64 =
            parse_hex_u64(chunk_value.at("source_fnv1a64").get<std::string>());
        tensor.chunks.push_back(std::move(chunk));
      }
      out.tensors.push_back(std::move(tensor));
    }
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("streamed dense orchestrator: malformed manifest: ") +
        e.what());
  }

  if (out.model.empty() || out.snapshot.empty() || out.tensors.empty()) {
    throw std::runtime_error(
        "streamed dense orchestrator: model/snapshot/tensors must be non-empty");
  }

  std::vector<std::string> names;
  for (const auto& tensor : out.tensors) {
    if (tensor.source_tensor.empty() || tensor.source_shape.empty() ||
        tensor.chunks.empty()) {
      throw std::runtime_error(
          "streamed dense orchestrator: malformed tensor entry");
    }
    names.push_back(tensor.source_tensor);
    const auto per_row = row_elements(tensor.source_shape);
    const auto total_rows = tensor.source_shape.front();
    std::size_t previous_end = 0U;
    for (const auto& chunk : tensor.chunks) {
      if (chunk.row_count == 0U || chunk.source_file.empty() ||
          chunk.source_fnv1a64 == 0U ||
          chunk.first_row < previous_end ||
          chunk.first_row > total_rows ||
          chunk.row_count > total_rows - chunk.first_row) {
        throw std::runtime_error(
            "streamed dense orchestrator: invalid chunk range");
      }
      const auto expected = checked_mul(
          checked_mul(
              static_cast<std::uint64_t>(chunk.row_count),
              per_row,
              "source chunk elements"),
          2U,
          "source chunk bytes");
      if (expected != chunk.source_byte_size) {
        throw std::runtime_error(
            "streamed dense orchestrator: BF16 chunk byte size mismatch");
      }
      previous_end = chunk.first_row + chunk.row_count;
    }
  }
  std::sort(names.begin(), names.end());
  if (std::adjacent_find(names.begin(), names.end()) != names.end()) {
    throw std::runtime_error(
        "streamed dense orchestrator: duplicate source tensor");
  }
  return out;
}

std::vector<StreamedSafetensorSpec>
plan_streamed_dense_output_specs(
    const QpackConversionPlan& plan,
    const StreamedDenseManifest& manifest,
    QpackExpertQuantizationSpec quantization) {
  if (quantization.bits != 4U || quantization.group_size == 0U) {
    throw std::invalid_argument(
        "streamed dense orchestrator: requires affine Q4 target");
  }

  std::vector<StreamedSafetensorSpec> specs;
  for (const auto& source : manifest.tensors) {
    const auto& entry = dense_entry(plan, source.source_tensor);
    if (entry.action == QpackConversionAction::copy_bf16_to_f32) {
      specs.push_back({
          entry.target_path,
          "F32",
          source.source_shape,
      });
      continue;
    }

    if (source.source_shape.size() != 2U) {
      throw std::runtime_error(
          "streamed dense orchestrator: affine source must be rank 2");
    }
    const auto rows = source.source_shape[0];
    const auto cols = source.source_shape[1];
    if ((cols % 8U) != 0U || (cols % quantization.group_size) != 0U) {
      throw std::runtime_error(
          "streamed dense orchestrator: affine column geometry unsupported");
    }
    specs.push_back({
        entry.target_path + ".weight",
        "U32",
        {rows, cols / 8U},
    });
    specs.push_back({
        entry.target_path + ".scales",
        "F32",
        {rows, cols / quantization.group_size},
    });
    specs.push_back({
        entry.target_path + ".biases",
        "F32",
        {rows, cols / quantization.group_size},
    });
  }

  std::sort(
      specs.begin(), specs.end(),
      [](const auto& a, const auto& b) { return a.name < b.name; });
  for (std::size_t i = 1U; i < specs.size(); ++i) {
    if (specs[i - 1U].name == specs[i].name) {
      throw std::runtime_error(
          "streamed dense orchestrator: duplicate target tensor");
    }
  }
  return specs;
}

StreamedDenseOrchestratorResult execute_streamed_dense_manifest(
    const QpackConversionPlan& plan,
    const StreamedDenseManifest& manifest,
    const std::filesystem::path& manifest_dir,
    StreamedSafetensorsBuilder& builder,
    QpackExpertQuantizationSpec quantization) {
  if (quantization.bits != 4U || quantization.group_size == 0U) {
    throw std::invalid_argument(
        "streamed dense orchestrator: requires affine Q4 target");
  }

  StreamedDenseOrchestratorResult result;
  result.tensor_count = manifest.tensors.size();
  double weighted_error_sum = 0.0;
  double weighted_value_count = 0.0;

  for (const auto& source : manifest.tensors) {
    const auto& entry = dense_entry(plan, source.source_tensor);
    const auto source_row_elements = row_elements(source.source_shape);

    for (const auto& chunk : source.chunks) {
      ++result.requested_chunks;
      const auto completed = completed_rows_for_source(builder, entry);
      const auto chunk_end = chunk.first_row + chunk.row_count;

      if (chunk_end <= completed) {
        ++result.skipped_chunks;
        continue;
      }
      if (chunk.first_row != completed) {
        throw std::runtime_error(
            "streamed dense orchestrator: chunk does not continue target progress");
      }

      const auto bytes = read_exact_source(
          manifest_dir / chunk.source_file,
          chunk.source_byte_size,
          chunk.source_fnv1a64);
      result.source_bytes += chunk.source_byte_size;

      if (entry.action == QpackConversionAction::copy_bf16_to_f32) {
        const auto values = decode_bf16_le(bytes);
        const auto out = f32_bytes(values);
        if (!builder.write_rows(
                entry.target_path,
                chunk.first_row,
                chunk.row_count,
                out)) {
          throw std::runtime_error(
              "streamed dense orchestrator: unexpected copy replay after source read");
        }
      } else {
        if (source.source_shape.size() != 2U) {
          throw std::runtime_error(
              "streamed dense orchestrator: affine source must be rank 2");
        }
        const auto converted = convert_bf16_affine_q4_row_chunk(
            bytes,
            chunk.row_count,
            source.source_shape[1],
            quantization.group_size);

        if (!builder.write_rows(
                entry.target_path + ".weight",
                chunk.first_row,
                chunk.row_count,
                converted.packed_weight_bytes) ||
            !builder.write_rows(
                entry.target_path + ".scales",
                chunk.first_row,
                chunk.row_count,
                converted.scale_bytes) ||
            !builder.write_rows(
                entry.target_path + ".biases",
                chunk.first_row,
                chunk.row_count,
                converted.bias_bytes)) {
          throw std::runtime_error(
              "streamed dense orchestrator: unexpected affine replay after source read");
        }

        result.max_abs_error =
            std::max(result.max_abs_error, converted.max_abs_error);
        const auto values =
            static_cast<double>(chunk.row_count) *
            static_cast<double>(source_row_elements);
        weighted_error_sum += converted.mean_abs_error * values;
        weighted_value_count += values;
      }

      ++result.converted_chunks;
    }
  }

  if (weighted_value_count > 0.0) {
    result.weighted_mean_abs_error =
        weighted_error_sum / weighted_value_count;
  }
  if (!std::isfinite(result.weighted_mean_abs_error) ||
      !std::isfinite(result.max_abs_error)) {
    throw std::runtime_error(
        "streamed dense orchestrator: non-finite aggregate metrics");
  }
  return result;
}

}  // namespace orbi::streammoe
