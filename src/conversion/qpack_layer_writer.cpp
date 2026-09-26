#include "orbi/streammoe/conversion/qpack_layer_writer.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/conversion/bf16_slice.hpp"

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::string hex_u64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << value;
  return out.str();
}

std::uint64_t parse_hex_u64(const std::string& value) {
  if (value.size() != 16U) {
    throw std::runtime_error(
        "QPACK layer writer: journal hash must be 16 hex digits");
  }
  std::uint64_t out = 0U;
  for (const char ch : value) {
    std::uint64_t digit{};
    if (ch >= '0' && ch <= '9') digit = static_cast<std::uint64_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f') digit = 10U + static_cast<std::uint64_t>(ch - 'a');
    else if (ch >= 'A' && ch <= 'F') digit = 10U + static_cast<std::uint64_t>(ch - 'A');
    else {
      throw std::runtime_error(
          "QPACK layer writer: journal hash contains non-hex digit");
    }
    out = (out << 4U) | digit;
  }
  return out;
}

std::uint64_t expected_layer_bytes(
    std::size_t expert_count,
    std::uint64_t expert_stride) {
  if (expert_count == 0U || expert_stride == 0U) {
    throw std::runtime_error(
        "QPACK layer writer: expert count/stride must be non-zero");
  }
  const auto count = static_cast<std::uint64_t>(expert_count);
  if (expert_stride > std::numeric_limits<std::uint64_t>::max() / count) {
    throw std::runtime_error(
        "QPACK layer writer: layer byte size overflows uint64");
  }
  return expert_stride * count;
}

std::uint64_t expert_offset(
    std::size_t expert_index,
    std::uint64_t expert_stride) {
  if (expert_index >
      std::numeric_limits<std::uint64_t>::max() / expert_stride) {
    throw std::runtime_error(
        "QPACK layer writer: expert offset overflows uint64");
  }
  return static_cast<std::uint64_t>(expert_index) * expert_stride;
}

void create_exact_layer_file(
    const std::filesystem::path& path,
    std::uint64_t bytes) {
  if (bytes == 0U ||
      bytes > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(
        "QPACK layer writer: layer file size exceeds stream limits");
  }

  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error(
        "QPACK layer writer: unable to create layer file");
  }
  out.seekp(static_cast<std::streamoff>(bytes - 1U), std::ios::beg);
  if (!out) {
    throw std::runtime_error(
        "QPACK layer writer: unable to size layer file");
  }
  const char zero = 0;
  out.write(&zero, 1);
  out.flush();
  if (!out) {
    throw std::runtime_error(
        "QPACK layer writer: unable to finalize layer file size");
  }
}

void verify_layer_file_size(
    const std::filesystem::path& path,
    std::uint64_t expected) {
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec || actual != expected) {
    throw std::runtime_error(
        "QPACK layer writer: layer file size mismatch");
  }
}

json journal_json(const QpackLayerResumeState& state) {
  json completed = json::array();
  for (std::size_t i = 0U; i < state.expert_fnv1a64.size(); ++i) {
    if (state.expert_fnv1a64[i] == 0U) continue;
    completed.push_back(
        {{"expert", i}, {"fnv1a64", hex_u64(state.expert_fnv1a64[i])}});
  }

  return {
      {"schema_version", 1U},
      {"layer_index", state.layer_index},
      {"expert_count", state.expert_count},
      {"expert_stride", state.expert_stride},
      {"completed", completed},
  };
}

void write_journal_safely(
    const std::filesystem::path& path,
    const QpackLayerResumeState& state) {
  std::filesystem::create_directories(path.parent_path());
  const auto temp = path.string() + ".tmp";
  const auto backup = path.string() + ".bak";

  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error(
          "QPACK layer writer: unable to create journal temp file");
    }
    out << journal_json(state).dump(2) << "\n";
    out.flush();
    if (!out) {
      throw std::runtime_error(
          "QPACK layer writer: unable to flush journal temp file");
    }
  }

  std::error_code ec;
  std::filesystem::remove(backup, ec);
  ec.clear();

  if (std::filesystem::exists(path)) {
    std::filesystem::rename(path, backup, ec);
    if (ec) {
      std::filesystem::remove(temp);
      throw std::runtime_error(
          "QPACK layer writer: unable to rotate journal");
    }
  }

  ec.clear();
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    if (std::filesystem::exists(backup)) {
      std::error_code restore_ec;
      std::filesystem::rename(backup, path, restore_ec);
    }
    std::filesystem::remove(temp);
    throw std::runtime_error(
        "QPACK layer writer: unable to commit journal");
  }

  std::filesystem::remove(backup, ec);
}

std::filesystem::path recover_journal_path(
    const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) return path;
  const auto backup = std::filesystem::path(path.string() + ".bak");
  if (std::filesystem::exists(backup)) return backup;
  return path;
}

QpackLayerResumeState read_journal(
    const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "QPACK layer writer: unable to open resume journal");
  }

  json root;
  try {
    input >> root;
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("QPACK layer writer: invalid journal JSON: ") + e.what());
  }

  if (root.value("schema_version", 0U) != 1U) {
    throw std::runtime_error(
        "QPACK layer writer: unsupported journal schema");
  }

  QpackLayerResumeState state;
  try {
    state.layer_index = root.at("layer_index").get<std::size_t>();
    state.expert_count = root.at("expert_count").get<std::size_t>();
    state.expert_stride = root.at("expert_stride").get<std::uint64_t>();
    state.expert_fnv1a64.assign(state.expert_count, 0U);

    const auto& completed = root.at("completed");
    if (!completed.is_array()) {
      throw std::runtime_error(
          "QPACK layer writer: journal completed must be an array");
    }

    for (const auto& item : completed) {
      const auto expert = item.at("expert").get<std::size_t>();
      const auto hash =
          parse_hex_u64(item.at("fnv1a64").get<std::string>());
      if (expert >= state.expert_count || hash == 0U ||
          state.expert_fnv1a64[expert] != 0U) {
        throw std::runtime_error(
            "QPACK layer writer: malformed completed expert entry");
      }
      state.expert_fnv1a64[expert] = hash;
      ++state.completed_experts;
    }
  } catch (const json::exception& e) {
    throw std::runtime_error(
        std::string("QPACK layer writer: malformed journal: ") + e.what());
  }

  return state;
}

std::vector<std::byte> read_expert_bytes(
    const std::filesystem::path& path,
    std::size_t expert_index,
    std::uint64_t expert_stride) {
  if (expert_stride >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      expert_stride >
      static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error(
        "QPACK layer writer: expert stride exceeds host stream limits");
  }
  const auto offset = expert_offset(expert_index, expert_stride);
  if (offset >
      static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error(
        "QPACK layer writer: expert offset exceeds streamoff");
  }

  std::vector<std::byte> bytes(static_cast<std::size_t>(expert_stride));
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "QPACK layer writer: unable to open layer file for readback");
  }
  input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  input.read(
      reinterpret_cast<char*>(bytes.data()),
      static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error(
        "QPACK layer writer: short expert readback");
  }
  return bytes;
}

void write_expert_bytes(
    const std::filesystem::path& path,
    std::size_t expert_index,
    std::uint64_t expert_stride,
    std::span<const std::byte> blob) {
  const auto offset = expert_offset(expert_index, expert_stride);
  if (offset >
      static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
      blob.size() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error(
        "QPACK layer writer: expert write exceeds stream limits");
  }

  std::fstream output(
      path,
      std::ios::binary | std::ios::in | std::ios::out);
  if (!output) {
    throw std::runtime_error(
        "QPACK layer writer: unable to open layer file for expert write");
  }
  output.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  output.write(
      reinterpret_cast<const char*>(blob.data()),
      static_cast<std::streamsize>(blob.size()));
  output.flush();
  if (!output) {
    throw std::runtime_error(
        "QPACK layer writer: expert write failed");
  }
}

}  // namespace

QpackLayerWriter::QpackLayerWriter(
    std::filesystem::path layer_path,
    std::filesystem::path journal_path,
    QpackLayerResumeState state)
    : layer_path_(std::move(layer_path)),
      journal_path_(std::move(journal_path)),
      state_(std::move(state)) {}

QpackLayerWriter QpackLayerWriter::open(
    std::filesystem::path layer_path,
    std::filesystem::path journal_path,
    const QpackExpertGeometry& geometry,
    std::size_t layer_index) {
  if (layer_index >= geometry.layer_count) {
    throw std::out_of_range(
        "QPACK layer writer: layer index out of range");
  }
  const auto layer_bytes =
      expected_layer_bytes(geometry.expert_count, geometry.expert_stride);

  const auto recoverable_journal = recover_journal_path(journal_path);
  if (std::filesystem::exists(recoverable_journal)) {
    auto state = read_journal(recoverable_journal);
    if (state.layer_index != layer_index ||
        state.expert_count != geometry.expert_count ||
        state.expert_stride != geometry.expert_stride) {
      throw std::runtime_error(
          "QPACK layer writer: resume journal disagrees with geometry");
    }
    verify_layer_file_size(layer_path, layer_bytes);

    QpackLayerWriter writer(
        std::move(layer_path),
        std::move(journal_path),
        std::move(state));
    for (std::size_t expert = 0U;
         expert < writer.state_.expert_count;
         ++expert) {
      if (writer.state_.expert_complete(expert)) {
        writer.verify_expert(expert);
      }
    }

    if (recoverable_journal != writer.journal_path_) {
      write_journal_safely(writer.journal_path_, writer.state_);
      std::error_code ec;
      std::filesystem::remove(recoverable_journal, ec);
    }
    return writer;
  }

  if (std::filesystem::exists(layer_path)) {
    throw std::runtime_error(
        "QPACK layer writer: layer exists without a resume journal");
  }

  create_exact_layer_file(layer_path, layer_bytes);

  QpackLayerResumeState state;
  state.layer_index = layer_index;
  state.expert_count = geometry.expert_count;
  state.expert_stride = geometry.expert_stride;
  state.expert_fnv1a64.assign(state.expert_count, 0U);
  write_journal_safely(journal_path, state);

  return QpackLayerWriter(
      std::move(layer_path),
      std::move(journal_path),
      std::move(state));
}

bool QpackLayerWriter::write_expert(
    std::size_t expert_index,
    std::span<const std::byte> blob) {
  if (expert_index >= state_.expert_count) {
    throw std::out_of_range(
        "QPACK layer writer: expert index out of range");
  }
  if (blob.size() != state_.expert_stride) {
    throw std::invalid_argument(
        "QPACK layer writer: expert blob size must equal expertStride");
  }

  const auto hash = fnv1a64(blob);
  if (hash == 0U) {
    throw std::runtime_error(
        "QPACK layer writer: zero expert hash is reserved");
  }

  if (state_.expert_complete(expert_index)) {
    if (state_.expert_fnv1a64[expert_index] != hash) {
      throw std::runtime_error(
          "QPACK layer writer: completed expert checksum mismatch");
    }
    verify_expert(expert_index);
    return false;
  }

  write_expert_bytes(
      layer_path_,
      expert_index,
      state_.expert_stride,
      blob);

  const auto readback =
      read_expert_bytes(layer_path_, expert_index, state_.expert_stride);
  if (fnv1a64(readback) != hash) {
    throw std::runtime_error(
        "QPACK layer writer: expert readback checksum mismatch");
  }

  state_.expert_fnv1a64[expert_index] = hash;
  ++state_.completed_experts;
  write_journal_safely(journal_path_, state_);
  return true;
}

void QpackLayerWriter::verify_expert(
    std::size_t expert_index) const {
  if (expert_index >= state_.expert_count ||
      !state_.expert_complete(expert_index)) {
    throw std::runtime_error(
        "QPACK layer writer: expert is not committed");
  }
  const auto bytes =
      read_expert_bytes(layer_path_, expert_index, state_.expert_stride);
  if (fnv1a64(bytes) != state_.expert_fnv1a64[expert_index]) {
    throw std::runtime_error(
        "QPACK layer writer: committed expert readback checksum mismatch");
  }
}

void QpackLayerWriter::finalize() const {
  if (!state_.complete()) {
    throw std::runtime_error(
        "QPACK layer writer: cannot finalize an incomplete layer");
  }
  for (std::size_t expert = 0U; expert < state_.expert_count; ++expert) {
    verify_expert(expert);
  }
}

}  // namespace orbi::streammoe
