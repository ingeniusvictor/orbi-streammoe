#include "orbi/streammoe/cache/expert_cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace orbi::streammoe {

ExpertCache::ExpertCache(
    ExpertStorage& storage,
    std::size_t budget_bytes,
    std::size_t minimum_slots)
    : storage_(storage),
      stride_(storage.expert_stride_bytes()),
      budget_bytes_(budget_bytes) {
  if (stride_ == 0) {
    throw std::invalid_argument("expert cache: storage stride must be non-zero");
  }

  max_slots_ = budget_bytes_ / stride_;
  if (max_slots_ < minimum_slots) {
    throw std::invalid_argument(
        "expert cache: budget cannot satisfy minimum slot count");
  }

  slots_.reserve(max_slots_);
  slot_meta_.reserve(max_slots_);
}

std::uint64_t ExpertCache::make_key(
    std::uint32_t layer,
    std::uint32_t expert) noexcept {
  return (static_cast<std::uint64_t>(layer) << 32U) |
         static_cast<std::uint64_t>(expert);
}

std::size_t ExpertCache::distinct_request_count(
    std::uint32_t layer,
    std::span<const std::uint32_t> experts) const {
  std::unordered_set<std::uint64_t> unique;
  unique.reserve(experts.size());
  for (const auto expert : experts) {
    unique.insert(make_key(layer, expert));
  }
  return unique.size();
}

std::size_t ExpertCache::existing_slot_for_fill(
    const std::unordered_set<std::size_t>& protected_slots) const {
  for (std::size_t slot = 0; slot < slot_meta_.size(); ++slot) {
    if (!slot_meta_[slot].key.has_value()) {
      return slot;
    }
  }

  std::size_t victim = std::numeric_limits<std::size_t>::max();
  std::tuple<std::size_t, std::uint64_t, std::size_t> best{
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::size_t>::max(),
  };

  for (std::size_t slot = 0; slot < slot_meta_.size(); ++slot) {
    if (protected_slots.contains(slot)) {
      continue;
    }

    const auto score = std::tuple{
        slot_meta_[slot].frequency,
        slot_meta_[slot].last_use,
        slot,
    };

    if (score < best) {
      best = score;
      victim = slot;
    }
  }

  if (victim == std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error(
        "expert cache: batch requires more simultaneous slots than budget allows");
  }

  return victim;
}

std::size_t ExpertCache::slot_for_fill(
    const std::unordered_set<std::size_t>& protected_slots) {
  if (slots_.size() < max_slots_) {
    slots_.emplace_back(stride_);
    slot_meta_.push_back({});
    return slots_.size() - 1;
  }

  return existing_slot_for_fill(protected_slots);
}

std::vector<ExpertCacheEntry> ExpertCache::fetch(
    std::uint32_t layer,
    std::span<const std::uint32_t> experts) {
  if (experts.empty()) {
    return {};
  }

  if (distinct_request_count(layer, experts) > max_slots_) {
    throw std::runtime_error(
        "expert cache: request contains more distinct experts than cache slots");
  }

  ++tick_;

  std::unordered_set<std::size_t> protected_slots;
  protected_slots.reserve(experts.size());

  std::vector<std::size_t> result_slots;
  result_slots.reserve(experts.size());

  std::vector<FillPlan> fills;
  fills.reserve(experts.size());

  for (const auto expert : experts) {
    const auto key = make_key(layer, expert);
    const auto new_frequency = frequency_[key] + 1U;
    frequency_[key] = new_frequency;

    if (const auto it = key_to_slot_.find(key); it != key_to_slot_.end()) {
      const auto slot = it->second;
      ++hits_;
      slot_meta_[slot].frequency = new_frequency;
      slot_meta_[slot].last_use = tick_;
      protected_slots.insert(slot);
      result_slots.push_back(slot);
      continue;
    }

    ++misses_;
    const auto slot = slot_for_fill(protected_slots);

    if (slot_meta_[slot].key.has_value()) {
      key_to_slot_.erase(*slot_meta_[slot].key);
    }

    slot_meta_[slot].key = key;
    slot_meta_[slot].frequency = new_frequency;
    slot_meta_[slot].last_use = tick_;
    key_to_slot_[key] = slot;

    protected_slots.insert(slot);
    result_slots.push_back(slot);
    fills.push_back({slot, expert});
  }

  if (!fills.empty()) {
    std::vector<ExpertReadRequest> requests;
    requests.reserve(fills.size());

    for (const auto& fill : fills) {
      requests.push_back({
          .id = {.layer = layer, .expert = fill.expert},
          .destination = std::span<std::byte>(
              slots_[fill.slot].data(),
              slots_[fill.slot].size()),
      });
    }

    try {
      storage_.read_experts(requests);
    } catch (...) {
      for (const auto& fill : fills) {
        const auto key = slot_meta_[fill.slot].key;
        if (key.has_value()) {
          key_to_slot_.erase(*key);
        }
        slot_meta_[fill.slot].key.reset();
        slot_meta_[fill.slot].frequency = 0;
        slot_meta_[fill.slot].last_use = 0;
      }
      throw;
    }
  }

  std::vector<ExpertCacheEntry> result;
  result.reserve(experts.size());

  for (std::size_t i = 0; i < experts.size(); ++i) {
    const auto slot = result_slots[i];
    result.push_back({
        .id = {.layer = layer, .expert = experts[i]},
        .slot = slot,
        .bytes = std::span<const std::byte>(
            slots_[slot].data(),
            slots_[slot].size()),
    });
  }

  return result;
}

std::optional<std::size_t> ExpertCache::resident_slot(
    std::uint32_t layer,
    std::uint32_t expert) const noexcept {
  const auto it = key_to_slot_.find(make_key(layer, expert));
  if (it == key_to_slot_.end()) {
    return std::nullopt;
  }
  return it->second;
}

ExpertCacheStats ExpertCache::stats() const noexcept {
  return {
      .hits = hits_,
      .misses = misses_,
      .allocated_slots = slots_.size(),
      .capacity_slots = max_slots_,
      .logical_bytes = slots_.size() * stride_,
      .budget_bytes = budget_bytes_,
  };
}

}  // namespace orbi::streammoe
