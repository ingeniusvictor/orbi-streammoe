#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/cache/expert_cache.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class FakeStorage final : public ExpertStorage {
 public:
  explicit FakeStorage(std::size_t stride) : stride_(stride) {}

  [[nodiscard]] std::size_t expert_stride_bytes() const noexcept override {
    return stride_;
  }

  void read_experts(std::span<ExpertReadRequest> requests) override {
    ++batch_calls;
    total_requests += requests.size();

    if (fail_next) {
      fail_next = false;
      throw std::runtime_error("synthetic read failure");
    }

    for (auto& request : requests) {
      if (request.destination.size() < stride_) {
        throw std::runtime_error("fake storage short destination");
      }
      for (std::size_t i = 0; i < stride_; ++i) {
        const auto value = static_cast<std::uint8_t>(
            (request.id.layer * 37U +
             request.id.expert * 11U +
             static_cast<std::uint32_t>(i)) &
            0xFFU);
        request.destination[i] = static_cast<std::byte>(value);
      }
    }
  }

  bool fail_next{};
  std::size_t batch_calls{};
  std::size_t total_requests{};

 private:
  std::size_t stride_{};
};

void check_bytes(
    const ExpertCacheEntry& entry,
    std::size_t stride) {
  require(entry.bytes.size() == stride, "cache entry stride mismatch");

  for (std::size_t i = 0; i < stride; ++i) {
    const auto expected = static_cast<std::uint8_t>(
        (entry.id.layer * 37U +
         entry.id.expert * 11U +
         static_cast<std::uint32_t>(i)) &
        0xFFU);
    const auto actual = std::to_integer<std::uint8_t>(entry.bytes[i]);
    require(actual == expected, "cache entry bytes mismatch");
  }
}

void test_growth_and_hits() {
  FakeStorage storage(8);
  ExpertCache cache(storage, 8 * 3);

  const std::vector<std::uint32_t> first_ids{1, 2};
  const auto first = cache.fetch(0, first_ids);

  require(first.size() == 2, "first fetch size");
  check_bytes(first[0], 8);
  check_bytes(first[1], 8);

  auto stats = cache.stats();
  require(stats.hits == 0, "initial hits");
  require(stats.misses == 2, "initial misses");
  require(stats.allocated_slots == 2, "lazy allocation count");
  require(storage.batch_calls == 1, "misses should be one storage batch");
  require(storage.total_requests == 2, "storage request count");

  const std::vector<std::uint32_t> second_ids{2, 1};
  const auto second = cache.fetch(0, second_ids);
  check_bytes(second[0], 8);
  check_bytes(second[1], 8);

  stats = cache.stats();
  require(stats.hits == 2, "repeat request hits");
  require(stats.misses == 2, "repeat request must not miss");
  require(storage.batch_calls == 1, "all-hit batch must not read storage");
}

void test_lfu_and_recency_eviction() {
  FakeStorage storage(4);
  ExpertCache cache(storage, 4 * 2);

  const std::vector<std::uint32_t> e0{0};
  const std::vector<std::uint32_t> e1{1};
  const std::vector<std::uint32_t> e2{2};

  (void)cache.fetch(0, e0);
  (void)cache.fetch(0, e0);
  (void)cache.fetch(0, e1);
  (void)cache.fetch(0, e2);

  require(cache.resident_slot(0, 0).has_value(), "hot expert should remain resident");
  require(!cache.resident_slot(0, 1).has_value(), "colder expert should be evicted");
  require(cache.resident_slot(0, 2).has_value(), "new expert should be resident");

  FakeStorage tie_storage(4);
  ExpertCache tie_cache(tie_storage, 4 * 2);
  (void)tie_cache.fetch(0, e0);
  (void)tie_cache.fetch(0, e1);
  (void)tie_cache.fetch(0, e2);

  require(!tie_cache.resident_slot(0, 0).has_value(), "older equal-frequency expert should evict");
  require(tie_cache.resident_slot(0, 1).has_value(), "more recent equal-frequency expert should survive");
  require(tie_cache.resident_slot(0, 2).has_value(), "replacement expert should be resident");
}

void test_batch_members_are_protected() {
  FakeStorage storage(8);
  ExpertCache cache(storage, 8 * 2);

  const std::vector<std::uint32_t> initial{0, 1};
  (void)cache.fetch(0, initial);

  const std::vector<std::uint32_t> replacement{2, 3};
  const auto entries = cache.fetch(0, replacement);

  require(entries.size() == 2, "protected batch size");
  require(cache.resident_slot(0, 2).has_value(), "first replacement missing");
  require(cache.resident_slot(0, 3).has_value(), "second replacement missing");
  require(entries[0].slot != entries[1].slot, "batch experts reused one protected slot");
}

void test_read_failure_clears_new_keys() {
  FakeStorage storage(8);
  ExpertCache cache(storage, 8 * 3);

  const std::vector<std::uint32_t> stable{0};
  (void)cache.fetch(1, stable);
  const auto stable_slot = cache.resident_slot(1, 0);
  require(stable_slot.has_value(), "stable expert missing before failure");

  storage.fail_next = true;
  const std::vector<std::uint32_t> failing{1, 2};

  bool failed = false;
  try {
    (void)cache.fetch(1, failing);
  } catch (const std::runtime_error&) {
    failed = true;
  }

  require(failed, "synthetic storage failure did not propagate");
  require(cache.resident_slot(1, 0) == stable_slot, "earlier resident expert was lost");
  require(!cache.resident_slot(1, 1).has_value(), "failed fill expert 1 advertised resident");
  require(!cache.resident_slot(1, 2).has_value(), "failed fill expert 2 advertised resident");
  require(cache.stats().allocated_slots == 3, "failed fill slots should remain allocated and reusable");
}

void test_oversized_batch_rejected_before_mutation() {
  FakeStorage storage(4);
  ExpertCache cache(storage, 4 * 2);

  const std::vector<std::uint32_t> too_many{0, 1, 2};

  bool rejected = false;
  try {
    (void)cache.fetch(0, too_many);
  } catch (const std::runtime_error&) {
    rejected = true;
  }

  require(rejected, "oversized batch was accepted");
  require(cache.stats().allocated_slots == 0, "oversized batch mutated cache before rejection");
  require(storage.batch_calls == 0, "oversized batch reached storage");
}

}  // namespace

int main() {
  try {
    test_growth_and_hits();
    test_lfu_and_recency_eviction();
    test_batch_members_are_protected();
    test_read_failure_clears_new_keys();
    test_oversized_batch_rejected_before_mutation();

    std::cout << "OSM-04 expert cache: PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "OSM-04 expert cache: FAIL: " << e.what() << "\n";
    return 1;
  }
}
