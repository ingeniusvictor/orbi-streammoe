#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"

#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string value) {
  if (target != nullptr) *target = std::move(value);
}

}  // namespace

VulkanResidentExpertCache::VulkanResidentExpertCache(
    VulkanComputeContext& context,
    const QpackReader& reader,
    ExpertCache& host_cache,
    std::size_t budget_bytes)
    : context_(context),
      reader_(reader),
      host_cache_(host_cache),
      budget_bytes_(budget_bytes) {
  if (!context_.valid()) {
    throw std::invalid_argument(
        "resident expert cache: Vulkan context must be valid");
  }
  if (budget_bytes_ == 0U) {
    throw std::invalid_argument(
        "resident expert cache: budget must be non-zero");
  }
}

std::uint64_t VulkanResidentExpertCache::make_key(
    std::uint32_t layer,
    std::uint32_t expert) noexcept {
  return (static_cast<std::uint64_t>(layer) << 32U) |
         static_cast<std::uint64_t>(expert);
}

std::uint64_t VulkanResidentExpertCache::eviction_victim() const {
  if (entries_.empty()) {
    throw std::runtime_error(
        "resident expert cache: no eviction victim available");
  }

  std::uint64_t victim = 0;
  bool have_victim = false;
  std::tuple<std::size_t, std::uint64_t, std::uint64_t> best{
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::uint64_t>::max(),
  };

  for (const auto& [key, node] : entries_) {
    const auto score = std::tuple{
        node.frequency,
        node.last_use,
        key,
    };
    if (!have_victim || score < best) {
      best = score;
      victim = key;
      have_victim = true;
    }
  }

  if (!have_victim) {
    throw std::runtime_error(
        "resident expert cache: failed to select eviction victim");
  }
  return victim;
}

VulkanResidentExpert* VulkanResidentExpertCache::get(
    std::uint32_t layer,
    std::uint32_t expert,
    std::string* diagnostic) noexcept {
  try {
    ++tick_;
    const auto key = make_key(layer, expert);
    const auto new_frequency = frequency_history_[key] + 1U;
    frequency_history_[key] = new_frequency;

    if (const auto it = entries_.find(key); it != entries_.end()) {
      ++hits_;
      it->second.frequency = new_frequency;
      it->second.last_use = tick_;
      set_diagnostic(
          diagnostic,
          "resident expert cache hit");
      return it->second.resident.get();
    }

    ++misses_;

    const std::vector<std::uint32_t> request{expert};
    const auto host_entries = host_cache_.fetch(layer, request);
    if (host_entries.size() != 1U) {
      set_diagnostic(
          diagnostic,
          "resident expert cache: host cache returned unexpected entry count");
      return nullptr;
    }

    std::string local;
    auto resident = VulkanResidentExpert::create(
        context_,
        reader_,
        host_entries.front(),
        &local);
    if (!resident.has_value()) {
      set_diagnostic(
          diagnostic,
          "resident expert cache: Vulkan upload failed: " + local);
      return nullptr;
    }

    const auto incoming_bytes = resident->accounted_bytes();
    if (incoming_bytes == 0U) {
      set_diagnostic(
          diagnostic,
          "resident expert cache: resident expert byte accounting is zero");
      return nullptr;
    }
    if (incoming_bytes > budget_bytes_) {
      set_diagnostic(
          diagnostic,
          "resident expert cache: one expert exceeds the configured budget");
      return nullptr;
    }

    while (!entries_.empty() &&
           resident_bytes_ + incoming_bytes > budget_bytes_) {
      const auto victim = eviction_victim();
      const auto it = entries_.find(victim);
      if (it == entries_.end()) {
        throw std::runtime_error(
            "resident expert cache: selected victim disappeared");
      }

      resident_bytes_ -= it->second.accounted_bytes;
      entries_.erase(it);
      ++evictions_;
    }

    auto owned = std::make_unique<VulkanResidentExpert>(
        std::move(*resident));
    auto* result = owned.get();

    Node node{
        .resident = std::move(owned),
        .accounted_bytes = incoming_bytes,
        .frequency = new_frequency,
        .last_use = tick_,
    };

    const auto [it, inserted] =
        entries_.emplace(key, std::move(node));
    if (!inserted) {
      set_diagnostic(
          diagnostic,
          "resident expert cache: duplicate insertion");
      return nullptr;
    }

    resident_bytes_ += incoming_bytes;
    ++loads_;

    set_diagnostic(
        diagnostic,
        "resident expert cache miss loaded into Vulkan");
    return result;
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("resident expert cache exception: ") + e.what());
    return nullptr;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "resident expert cache encountered an unknown exception");
    return nullptr;
  }
}

VulkanResidentExpertResult VulkanResidentExpertCache::run(
    std::uint32_t layer,
    std::uint32_t expert,
    std::span<const float> x) noexcept {
  VulkanResidentExpertResult result;
  std::string diagnostic;
  auto* resident = get(layer, expert, &diagnostic);
  if (resident == nullptr) {
    result.diagnostic = diagnostic;
    return result;
  }

  result = resident->run(context_, x);
  if (!result.executed && !diagnostic.empty()) {
    result.diagnostic =
        diagnostic + "; execution: " + result.diagnostic;
  }
  return result;
}

bool VulkanResidentExpertCache::contains(
    std::uint32_t layer,
    std::uint32_t expert) const noexcept {
  return entries_.contains(make_key(layer, expert));
}

VulkanResidentExpertCacheStats VulkanResidentExpertCache::stats() const noexcept {
  return {
      .hits = hits_,
      .misses = misses_,
      .loads = loads_,
      .evictions = evictions_,
      .resident_entries = entries_.size(),
      .resident_bytes = resident_bytes_,
      .budget_bytes = budget_bytes_,
  };
}

}  // namespace orbi::streammoe
