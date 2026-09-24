#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace orbi::streammoe {

struct QpackSection {
  std::string name;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::uint64_t offset{};
  std::uint64_t size{};
};

struct QpackLayout {
  std::uint32_t expert_count{};
  std::uint32_t layer_count{};
  std::uint64_t expert_stride{};
  std::vector<QpackSection> sections;
  std::vector<bool> linear_layers;

  [[nodiscard]] const QpackSection* find_section(std::string_view name) const noexcept;
};

struct QpackManifest {
  std::string magic;
  std::uint32_t version{};
  std::string model_name;
  std::string source_checkpoint;
  std::optional<std::uint32_t> quant_bits;
  std::optional<std::uint32_t> quant_group_size;
  std::unordered_map<std::string, std::uint64_t> files;
};

class QpackReader {
 public:
  static constexpr std::string_view kMagic = "QPACK";
  static constexpr std::uint32_t kManifestVersion = 1;

  explicit QpackReader(std::filesystem::path container_dir);

  [[nodiscard]] const std::filesystem::path& container_dir() const noexcept {
    return container_dir_;
  }

  [[nodiscard]] const QpackManifest& manifest() const noexcept {
    return manifest_;
  }

  [[nodiscard]] const QpackLayout& layout() const noexcept {
    return layout_;
  }

  [[nodiscard]] const QpackSection* find_section(std::string_view name) const noexcept {
    return layout_.find_section(name);
  }

  // Reads exactly one fixed-stride expert blob into caller-owned memory.
  // Throws on an invalid layer/expert, undersized destination or short read.
  void read_expert(
      std::uint32_t layer,
      std::uint32_t expert,
      std::span<std::byte> destination) const;

  [[nodiscard]] std::vector<std::byte> read_expert(
      std::uint32_t layer,
      std::uint32_t expert) const;

 private:
  std::filesystem::path container_dir_;
  QpackManifest manifest_;
  QpackLayout layout_;

  [[nodiscard]] std::filesystem::path layer_path(std::uint32_t layer) const;
  void validate_container() const;
};

}  // namespace orbi::streammoe
