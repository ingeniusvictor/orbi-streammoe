#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "orbi/streammoe/storage/expert_storage.hpp"

namespace orbi::streammoe {

struct ExpertCacheEntry {
  ExpertId id{};
  std::size_t slot{};
  std::span<const std::byte> bytes{};
};

struct ExpertCacheStats {
  std::size_t hits{};
  std::size_t misses{};
  std::size_t allocated_slots{};
  std::size_t capacity_slots{};
  std::size_t logical_bytes{};
  std::size_t budget_bytes{};
};

class ExpertCache {
 public:
  ExpertCache(
      ExpertStorage& storage,
      std::size_t budget_bytes,
      std::size_t minimum_slots = 1);

  [[nodiscard]] std::vector<ExpertCacheEntry> fetch(
      std::uint32_t layer,
      std::span<const std::uint32_t> experts);

  [[nodiscard]] std::optional<std::size_t> resident_slot(
      std::uint32_t layer,
      std::uint32_t expert) const noexcept;

  [[nodiscard]] ExpertCacheStats stats() const noexcept;
  [[nodiscard]] std::size_t stride_bytes() const noexcept { return stride_; }

 private:
  struct SlotMeta {
    std::optional<std::uint64_t> key{};
    std::size_t frequency{};
    std::uint64_t last_use{};
  };

  struct FillPlan {
    std::size_t slot{};
    std::uint32_t expert{};
  };

  ExpertStorage& storage_;
  std::size_t stride_{};
  std::size_t budget_bytes_{};
  std::size_t max_slots_{};

  std::vector<std::vector<std::byte>> slots_;
  std::vector<SlotMeta> slot_meta_;
  std::unordered_map<std::uint64_t, std::size_t> key_to_slot_;
  std::unordered_map<std::uint64_t, std::size_t> frequency_;

  std::uint64_t tick_{};
  std::size_t hits_{};
  std::size_t misses_{};

  [[nodiscard]] static std::uint64_t make_key(
      std::uint32_t layer,
      std::uint32_t expert) noexcept;

  [[nodiscard]] std::size_t slot_for_fill(
      const std::unordered_set<std::size_t>& protected_slots);

  [[nodiscard]] std::size_t existing_slot_for_fill(
      const std::unordered_set<std::size_t>& protected_slots) const;

  [[nodiscard]] std::size_t distinct_request_count(
      std::uint32_t layer,
      std::span<const std::uint32_t> experts) const;
};

}  // namespace orbi::streammoe
