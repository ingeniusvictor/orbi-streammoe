#include "orbi/streammoe/conversion/streamed_safetensors_builder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/container/safetensors.hpp"
#include "orbi/streammoe/conversion/bf16_slice.hpp"
#include "orbi/streammoe/conversion/single_expert_pilot.hpp"

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* label) {
  if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a) {
    throw std::runtime_error(std::string("streamed safetensors: overflow: ") + label);
  }
  return a * b;
}

std::uint64_t tensor_rows(const StreamedSafetensorSpec& spec) {
  if (spec.shape.empty() || spec.shape.front() == 0U) {
    throw std::runtime_error("streamed safetensors: tensor shape must have rows");
  }
  return static_cast<std::uint64_t>(spec.shape.front());
}

std::uint64_t row_bytes(const StreamedSafetensorSpec& spec) {
  const auto bpe = SafetensorsReader::bytes_per_element(spec.dtype);
  std::uint64_t elements = 1U;
  for (std::size_t i = 1U; i < spec.shape.size(); ++i) {
    if (spec.shape[i] == 0U) {
      throw std::runtime_error("streamed safetensors: zero tensor dimension");
    }
    elements = checked_mul(
        elements,
        static_cast<std::uint64_t>(spec.shape[i]),
        "row elements");
  }
  return checked_mul(elements, static_cast<std::uint64_t>(bpe), "row bytes");
}

std::string hex_u64(std::uint64_t value) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string out(16U, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<std::size_t>(i)] = digits[value & 0xFU];
    value >>= 4U;
  }
  return out;
}

std::uint64_t parse_hex_u64(const std::string& value) {
  if (value.size() != 16U) {
    throw std::runtime_error("streamed safetensors: invalid journal hash");
  }
  std::uint64_t out = 0U;
  for (const char ch : value) {
    std::uint64_t digit{};
    if (ch >= '0' && ch <= '9') digit = static_cast<std::uint64_t>(ch - '0');
    else if (ch >= 'a' && ch <= 'f') digit = 10U + static_cast<std::uint64_t>(ch - 'a');
    else if (ch >= 'A' && ch <= 'F') digit = 10U + static_cast<std::uint64_t>(ch - 'A');
    else throw std::runtime_error("streamed safetensors: non-hex journal hash");
    out = (out << 4U) | digit;
  }
  return out;
}

void append_u64_le(std::ofstream& out, std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (std::size_t i = 0U; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (i * 8U)) & 0xFFU);
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

StreamedSafetensorsState plan_layout(std::vector<StreamedSafetensorSpec> specs) {
  if (specs.empty()) {
    throw std::runtime_error("streamed safetensors: no tensor specs");
  }
  std::sort(specs.begin(), specs.end(), [](const auto& a, const auto& b) {
    return a.name < b.name;
  });

  json header = json::object();
  std::uint64_t payload_offset = 0U;
  std::string previous;

  StreamedSafetensorsState state;
  state.tensors.reserve(specs.size());

  for (auto& spec : specs) {
    if (spec.name.empty() || (!previous.empty() && spec.name <= previous)) {
      throw std::runtime_error(
          "streamed safetensors: tensor names must be unique");
    }
    previous = spec.name;

    const auto rows = tensor_rows(spec);
    const auto bytes_per_row = row_bytes(spec);
    const auto bytes = checked_mul(rows, bytes_per_row, "tensor bytes");
    const auto begin = payload_offset;
    if (bytes > std::numeric_limits<std::uint64_t>::max() - payload_offset) {
      throw std::runtime_error("streamed safetensors: payload offset overflow");
    }
    payload_offset += bytes;

    header[spec.name] = {
        {"dtype", spec.dtype},
        {"shape", spec.shape},
        {"data_offsets", {begin, payload_offset}},
    };

    StreamedSafetensorState tensor;
    tensor.spec = std::move(spec);
    tensor.payload_offset = begin;
    tensor.row_bytes = bytes_per_row;
    state.tensors.push_back(std::move(tensor));
  }

  std::string header_text = header.dump();
  while ((header_text.size() % 8U) != 0U) header_text.push_back(' ');
  state.data_start = 8U + static_cast<std::uint64_t>(header_text.size());
  state.payload_bytes = payload_offset;
  return state;
}

std::string header_text_for_state(const StreamedSafetensorsState& state) {
  json header = json::object();
  for (const auto& tensor : state.tensors) {
    const auto bytes = checked_mul(
        tensor_rows(tensor.spec), tensor.row_bytes, "tensor bytes");
    header[tensor.spec.name] = {
        {"dtype", tensor.spec.dtype},
        {"shape", tensor.spec.shape},
        {"data_offsets", {tensor.payload_offset, tensor.payload_offset + bytes}},
    };
  }
  std::string text = header.dump();
  while ((text.size() % 8U) != 0U) text.push_back(' ');
  return text;
}

void create_output_file(
    const std::filesystem::path& path,
    const StreamedSafetensorsState& state) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const auto header = header_text_for_state(state);
  if (8U + header.size() != state.data_start) {
    throw std::runtime_error("streamed safetensors: header plan drift");
  }
  const auto total = state.data_start + state.payload_bytes;
  if (total == 0U ||
      total > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error("streamed safetensors: output exceeds stream limits");
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("streamed safetensors: create output failed");
  append_u64_le(out, static_cast<std::uint64_t>(header.size()));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (state.payload_bytes != 0U) {
    out.seekp(static_cast<std::streamoff>(total - 1U), std::ios::beg);
    const char zero = 0;
    out.write(&zero, 1);
  }
  out.flush();
  if (!out) throw std::runtime_error("streamed safetensors: preallocation failed");
}

json journal_json(const StreamedSafetensorsState& state) {
  json tensors = json::array();
  for (const auto& tensor : state.tensors) {
    json chunks = json::array();
    for (const auto& chunk : tensor.chunks) {
      chunks.push_back({
          {"first_row", chunk.first_row},
          {"row_count", chunk.row_count},
          {"fnv1a64", hex_u64(chunk.fnv1a64)},
      });
    }
    tensors.push_back({
        {"name", tensor.spec.name},
        {"dtype", tensor.spec.dtype},
        {"shape", tensor.spec.shape},
        {"payload_offset", tensor.payload_offset},
        {"row_bytes", tensor.row_bytes},
        {"completed_rows", tensor.completed_rows},
        {"chunks", chunks},
    });
  }
  return {
      {"schema_version", 1U},
      {"data_start", state.data_start},
      {"payload_bytes", state.payload_bytes},
      {"tensors", tensors},
  };
}

void write_journal(
    const std::filesystem::path& path,
    const StreamedSafetensorsState& state) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const auto temp = std::filesystem::path(path.string() + ".tmp");
  const auto backup = std::filesystem::path(path.string() + ".bak");
  {
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("streamed safetensors: journal create failed");
    out << journal_json(state).dump(2) << "\n";
    out.flush();
    if (!out) throw std::runtime_error("streamed safetensors: journal flush failed");
  }

  std::error_code ec;
  std::filesystem::remove(backup, ec);
  ec.clear();
  if (std::filesystem::exists(path)) {
    std::filesystem::rename(path, backup, ec);
    if (ec) {
      std::filesystem::remove(temp);
      throw std::runtime_error("streamed safetensors: journal rotation failed");
    }
  }
  ec.clear();
  std::filesystem::rename(temp, path, ec);
  if (ec) {
    if (std::filesystem::exists(backup)) {
      std::error_code restore;
      std::filesystem::rename(backup, path, restore);
    }
    std::filesystem::remove(temp);
    throw std::runtime_error("streamed safetensors: journal commit failed");
  }
  std::filesystem::remove(backup, ec);
}

std::filesystem::path recover_journal(const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) return path;
  const auto backup = std::filesystem::path(path.string() + ".bak");
  if (std::filesystem::exists(backup)) return backup;
  return path;
}

StreamedSafetensorsState read_journal(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("streamed safetensors: journal open failed");
  json root;
  in >> root;
  if (root.value("schema_version", 0U) != 1U) {
    throw std::runtime_error("streamed safetensors: unsupported journal schema");
  }

  StreamedSafetensorsState state;
  state.data_start = root.at("data_start").get<std::uint64_t>();
  state.payload_bytes = root.at("payload_bytes").get<std::uint64_t>();
  for (const auto& value : root.at("tensors")) {
    StreamedSafetensorState tensor;
    tensor.spec.name = value.at("name").get<std::string>();
    tensor.spec.dtype = value.at("dtype").get<std::string>();
    tensor.spec.shape = value.at("shape").get<std::vector<std::size_t>>();
    tensor.payload_offset = value.at("payload_offset").get<std::uint64_t>();
    tensor.row_bytes = value.at("row_bytes").get<std::uint64_t>();
    tensor.completed_rows = value.at("completed_rows").get<std::size_t>();
    for (const auto& item : value.at("chunks")) {
      tensor.chunks.push_back({
          item.at("first_row").get<std::size_t>(),
          item.at("row_count").get<std::size_t>(),
          parse_hex_u64(item.at("fnv1a64").get<std::string>()),
      });
    }
    state.tensors.push_back(std::move(tensor));
  }
  return state;
}

StreamedSafetensorState& find_tensor(
    StreamedSafetensorsState& state,
    const std::string& name) {
  const auto it = std::lower_bound(
      state.tensors.begin(), state.tensors.end(), name,
      [](const auto& tensor, const std::string& value) {
        return tensor.spec.name < value;
      });
  if (it == state.tensors.end() || it->spec.name != name) {
    throw std::runtime_error("streamed safetensors: missing tensor: " + name);
  }
  return *it;
}

const StreamedSafetensorState& find_tensor(
    const StreamedSafetensorsState& state,
    const std::string& name) {
  const auto it = std::lower_bound(
      state.tensors.begin(), state.tensors.end(), name,
      [](const auto& tensor, const std::string& value) {
        return tensor.spec.name < value;
      });
  if (it == state.tensors.end() || it->spec.name != name) {
    throw std::runtime_error("streamed safetensors: missing tensor: " + name);
  }
  return *it;
}

std::vector<std::byte> read_exact(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::size_t size) {
  if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
      size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("streamed safetensors: read exceeds stream limits");
  }
  std::vector<std::byte> bytes(size);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("streamed safetensors: output read failed");
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("streamed safetensors: short output read");
  }
  return bytes;
}

void write_exact(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::span<const std::byte> bytes) {
  if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
      bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("streamed safetensors: write exceeds stream limits");
  }
  std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
  if (!out) throw std::runtime_error("streamed safetensors: output write failed");
  out.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  out.flush();
  if (!out) throw std::runtime_error("streamed safetensors: output chunk flush failed");
}

bool same_specs(
    const StreamedSafetensorsState& a,
    const StreamedSafetensorsState& b) {
  if (a.data_start != b.data_start ||
      a.payload_bytes != b.payload_bytes ||
      a.tensors.size() != b.tensors.size()) return false;
  for (std::size_t i = 0U; i < a.tensors.size(); ++i) {
    const auto& x = a.tensors[i];
    const auto& y = b.tensors[i];
    if (x.spec.name != y.spec.name || x.spec.dtype != y.spec.dtype ||
        x.spec.shape != y.spec.shape || x.payload_offset != y.payload_offset ||
        x.row_bytes != y.row_bytes) return false;
  }
  return true;
}

std::vector<std::byte> f32_bytes(std::span<const float> values) {
  std::vector<std::byte> out(values.size() * 4U);
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      out[i * 4U + lane] =
          static_cast<std::byte>((bits >> (lane * 8U)) & 0xFFU);
    }
  }
  return out;
}

std::vector<std::byte> u32_bytes(std::span<const std::uint32_t> values) {
  std::vector<std::byte> out(values.size() * 4U);
  for (std::size_t i = 0U; i < values.size(); ++i) {
    for (std::size_t lane = 0U; lane < 4U; ++lane) {
      out[i * 4U + lane] =
          static_cast<std::byte>((values[i] >> (lane * 8U)) & 0xFFU);
    }
  }
  return out;
}

}  // namespace

bool StreamedSafetensorsState::complete() const noexcept {
  if (tensors.empty()) return false;
  for (const auto& tensor : tensors) {
    if (tensor.spec.shape.empty() ||
        tensor.completed_rows != tensor.spec.shape.front()) return false;
  }
  return true;
}

StreamedSafetensorsBuilder::StreamedSafetensorsBuilder(
    std::filesystem::path output_path,
    std::filesystem::path journal_path,
    StreamedSafetensorsState state)
    : output_path_(std::move(output_path)),
      journal_path_(std::move(journal_path)),
      state_(std::move(state)) {}

StreamedSafetensorsBuilder StreamedSafetensorsBuilder::open(
    std::filesystem::path output_path,
    std::filesystem::path journal_path,
    std::vector<StreamedSafetensorSpec> specs) {
  const auto planned = plan_layout(std::move(specs));
  const auto journal = recover_journal(journal_path);

  if (std::filesystem::exists(journal)) {
    auto restored = read_journal(journal);
    if (!same_specs(planned, restored)) {
      throw std::runtime_error(
          "streamed safetensors: journal disagrees with requested layout");
    }
    const auto expected_size = restored.data_start + restored.payload_bytes;
    std::error_code ec;
    const auto actual_size = std::filesystem::file_size(output_path, ec);
    if (ec || actual_size != expected_size) {
      throw std::runtime_error("streamed safetensors: output file size mismatch");
    }

    StreamedSafetensorsBuilder builder(
        std::move(output_path), std::move(journal_path), std::move(restored));
    for (const auto& tensor : builder.state_.tensors) {
      builder.verify_tensor(tensor.spec.name);
    }
    if (journal != builder.journal_path_) {
      write_journal(builder.journal_path_, builder.state_);
      std::error_code remove_ec;
      std::filesystem::remove(journal, remove_ec);
    }
    return builder;
  }

  if (std::filesystem::exists(output_path)) {
    throw std::runtime_error(
        "streamed safetensors: output exists without resume journal");
  }
  create_output_file(output_path, planned);
  write_journal(journal_path, planned);
  return StreamedSafetensorsBuilder(
      std::move(output_path), std::move(journal_path), planned);
}

bool StreamedSafetensorsBuilder::write_rows(
    const std::string& tensor_name,
    std::size_t first_row,
    std::size_t row_count,
    std::span<const std::byte> bytes) {
  auto& tensor = find_tensor(state_, tensor_name);
  const auto total_rows = tensor.spec.shape.front();
  if (row_count == 0U || first_row > total_rows ||
      row_count > total_rows - first_row) {
    throw std::out_of_range("streamed safetensors: row range out of bounds");
  }
  const auto expected = checked_mul(
      static_cast<std::uint64_t>(row_count), tensor.row_bytes, "chunk bytes");
  if (expected != bytes.size()) {
    throw std::invalid_argument("streamed safetensors: chunk byte size mismatch");
  }

  if (first_row < tensor.completed_rows) {
    const auto it = std::find_if(
        tensor.chunks.begin(), tensor.chunks.end(),
        [&](const auto& chunk) {
          return chunk.first_row == first_row && chunk.row_count == row_count;
        });
    if (it == tensor.chunks.end()) {
      throw std::runtime_error(
          "streamed safetensors: replay must match an exact committed chunk");
    }
    if (fnv1a64(bytes) != it->fnv1a64) {
      throw std::runtime_error(
          "streamed safetensors: replay chunk checksum mismatch");
    }
    const auto offset =
        state_.data_start + tensor.payload_offset +
        static_cast<std::uint64_t>(first_row) * tensor.row_bytes;
    const auto readback = read_exact(output_path_, offset, bytes.size());
    if (fnv1a64(readback) != it->fnv1a64) {
      throw std::runtime_error(
          "streamed safetensors: committed replay readback mismatch");
    }
    return false;
  }

  if (first_row != tensor.completed_rows) {
    throw std::runtime_error(
        "streamed safetensors: new chunks must be contiguous");
  }

  const auto hash = fnv1a64(bytes);
  if (hash == 0U) {
    throw std::runtime_error("streamed safetensors: zero FNV hash is reserved");
  }
  const auto offset =
      state_.data_start + tensor.payload_offset +
      static_cast<std::uint64_t>(first_row) * tensor.row_bytes;
  write_exact(output_path_, offset, bytes);
  const auto readback = read_exact(output_path_, offset, bytes.size());
  if (fnv1a64(readback) != hash) {
    throw std::runtime_error("streamed safetensors: chunk readback mismatch");
  }

  tensor.chunks.push_back({first_row, row_count, hash});
  tensor.completed_rows += row_count;
  write_journal(journal_path_, state_);
  return true;
}

void StreamedSafetensorsBuilder::verify_tensor(
    const std::string& tensor_name) const {
  const auto& tensor = find_tensor(state_, tensor_name);
  std::size_t expected_row = 0U;
  for (const auto& chunk : tensor.chunks) {
    if (chunk.first_row != expected_row || chunk.row_count == 0U) {
      throw std::runtime_error(
          "streamed safetensors: journal chunk sequence is not contiguous");
    }
    const auto size64 = checked_mul(
        static_cast<std::uint64_t>(chunk.row_count), tensor.row_bytes,
        "verify chunk bytes");
    if (size64 > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error("streamed safetensors: verify chunk too large");
    }
    const auto offset =
        state_.data_start + tensor.payload_offset +
        static_cast<std::uint64_t>(chunk.first_row) * tensor.row_bytes;
    const auto bytes =
        read_exact(output_path_, offset, static_cast<std::size_t>(size64));
    if (fnv1a64(bytes) != chunk.fnv1a64) {
      throw std::runtime_error(
          "streamed safetensors: committed chunk checksum mismatch");
    }
    expected_row += chunk.row_count;
  }
  if (expected_row != tensor.completed_rows) {
    throw std::runtime_error(
        "streamed safetensors: completed row count disagrees with chunks");
  }
}

void StreamedSafetensorsBuilder::finalize() const {
  if (!state_.complete()) {
    throw std::runtime_error(
        "streamed safetensors: cannot finalize incomplete output");
  }
  for (const auto& tensor : state_.tensors) {
    verify_tensor(tensor.spec.name);
  }

  const SafetensorsReader reader(output_path_);
  const auto names = reader.tensor_names();
  if (names.size() != state_.tensors.size()) {
    throw std::runtime_error(
        "streamed safetensors: production reader tensor count mismatch");
  }
  for (std::size_t i = 0U; i < names.size(); ++i) {
    if (names[i] != state_.tensors[i].spec.name) {
      throw std::runtime_error(
          "streamed safetensors: production reader inventory mismatch");
    }
  }
}

StreamedAffineQ4Chunk convert_bf16_affine_q4_row_chunk(
    std::span<const std::byte> bf16_bytes,
    std::size_t rows,
    std::size_t cols,
    std::size_t group_size) {
  if (rows == 0U || cols == 0U ||
      rows > std::numeric_limits<std::size_t>::max() / cols ||
      bf16_bytes.size() != rows * cols * 2U) {
    throw std::invalid_argument(
        "streamed safetensors: invalid BF16 row chunk geometry");
  }

  const auto values = decode_bf16_le(bf16_bytes);
  const auto affine =
      quantize_affine_q4_rows(values, rows, cols, group_size);

  StreamedAffineQ4Chunk out;
  out.rows = rows;
  out.cols = cols;
  out.packed_cols = affine.packed_cols;
  out.groups_per_row = affine.groups_per_row;
  out.packed_weight_bytes = u32_bytes(affine.packed);
  out.scale_bytes = f32_bytes(affine.scales);
  out.bias_bytes = f32_bytes(affine.biases);
  out.max_abs_error = affine.max_abs_error;
  out.mean_abs_error = affine.mean_abs_error;
  return out;
}

}  // namespace orbi::streammoe
