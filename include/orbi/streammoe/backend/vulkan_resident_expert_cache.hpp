#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {

struct VulkanResidentExpertCacheStats {
  std::size_t hits{};
  std::size_t misses{};
  std::size_t loads{};
  std::size_t evictions{};
  std::size_t resident_entries{};
  std::size_t resident_bytes{};
  std::size_t budget_bytes{};
};

class VulkanResidentExpertCache {
 public:
  VulkanResidentExpertCache(
      VulkanComputeContext& context,
      const QpackReader& reader,
      ExpertCache& host_cache,
      std::size_t budget_bytes);

  VulkanResidentExpertCache(const VulkanResidentExpertCache&) = delete;
  VulkanResidentExpertCache& operator=(const VulkanResidentExpertCache&) = delete;

  [[nodiscard]] VulkanResidentExpert* get(
      std::uint32_t layer,
      std::uint32_t expert,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] VulkanResidentExpertResult run(
      std::uint32_t layer,
      std::uint32_t expert,
      std::span<const float> x) noexcept;

  [[nodiscard]] bool contains(
      std::uint32_t layer,
      std::uint32_t expert) const noexcept;

  [[nodiscard]] VulkanResidentExpertCacheStats stats() const noexcept;

 private:
  struct Node {
    std::unique_ptr<VulkanResidentExpert> resident;
    std::size_t accounted_bytes{};
    std::size_t frequency{};
    std::uint64_t last_use{};
  };

  VulkanComputeContext& context_;
  const QpackReader& reader_;
  ExpertCache& host_cache_;
  std::size_t budget_bytes_{};
  std::size_t resident_bytes_{};

  std::unordered_map<std::uint64_t, Node> entries_;
  std::unordered_map<std::uint64_t, std::size_t> frequency_history_;

  std::uint64_t tick_{};
  std::size_t hits_{};
  std::size_t misses_{};
  std::size_t loads_{};
  std::size_t evictions_{};

  [[nodiscard]] static std::uint64_t make_key(
      std::uint32_t layer,
      std::uint32_t expert) noexcept;

  [[nodiscard]] std::uint64_t eviction_victim() const;
};

}  // namespace orbi::streammoe
