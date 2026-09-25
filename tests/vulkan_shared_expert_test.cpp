#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/backend/vulkan_resident_shared_expert.hpp"
#include "orbi/streammoe/cache/expert_cache.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"
#include "orbi/streammoe/model/routed_moe.hpp"
#include "orbi/streammoe/model/shared_expert.hpp"
#include "orbi/streammoe/storage/qpack_expert_storage.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

constexpr std::size_t kExpertStride = 16384U;
constexpr std::size_t kAccountedExpertBytes = 17152U;
constexpr std::size_t kHidden = 128U;
constexpr std::size_t kIntermediate = 64U;
constexpr std::size_t kGroupSize = 64U;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool require_vulkan_compute() {
  const char* value = std::getenv("ORBI_REQUIRE_VULKAN_COMPUTE");
  return value != nullptr && std::string(value) != "0";
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to create " + path.string());
  out << text;
}

std::uint16_t bf16(float value) {
  return static_cast<std::uint16_t>(
      std::bit_cast<std::uint32_t>(value) >> 16U);
}

std::uint32_t pack_word(
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t row,
    std::size_t word) {
  std::uint32_t packed = 0;
  for (std::uint32_t lane = 0; lane < 8; ++lane) {
    const auto q = static_cast<std::uint32_t>(
        (projection_seed * 3U + expert * 5U + row * 7U +
         word * 2U + lane) & 0xFU);
    packed |= q << (lane * 4U);
  }
  return packed;
}

float scale_value(
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t row,
    std::size_t group) {
  const int bucket = static_cast<int>(
      (projection_seed + expert + row + group) % 6U) + 1;
  return static_cast<float>(bucket) * 0.125F;
}

float bias_value(
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t row,
    std::size_t group) {
  const int bucket = static_cast<int>(
      (projection_seed * 2U + expert + row + group) % 7U) - 3;
  return static_cast<float>(bucket) * 0.125F;
}

void write_u32(
    std::vector<std::byte>& blob,
    std::size_t offset,
    std::uint32_t value) {
  std::memcpy(blob.data() + offset, &value, sizeof(value));
}

void write_bf16(
    std::vector<std::byte>& blob,
    std::size_t offset,
    float value) {
  const auto bits = bf16(value);
  std::memcpy(blob.data() + offset, &bits, sizeof(bits));
}

void fill_qpack_projection(
    std::vector<std::byte>& blob,
    std::size_t projection_seed,
    std::size_t expert,
    std::size_t out_dim,
    std::size_t packed_cols,
    std::size_t groups_per_row,
    std::size_t weight_offset,
    std::size_t scales_offset,
    std::size_t biases_offset) {
  for (std::size_t row = 0; row < out_dim; ++row) {
    for (std::size_t word = 0; word < packed_cols; ++word) {
      write_u32(
          blob,
          weight_offset +
              (row * packed_cols + word) * sizeof(std::uint32_t),
          pack_word(projection_seed, expert, row, word));
    }

    for (std::size_t group = 0; group < groups_per_row; ++group) {
      const auto index = row * groups_per_row + group;
      write_bf16(
          blob,
          scales_offset + index * sizeof(std::uint16_t),
          scale_value(projection_seed, expert, row, group));
      write_bf16(
          blob,
          biases_offset + index * sizeof(std::uint16_t),
          bias_value(projection_seed, expert, row, group));
    }
  }
}

fs::path make_qpack(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root / "packed_experts");

  constexpr std::size_t gate_weight = 0;
  constexpr std::size_t gate_scales = 4096;
  constexpr std::size_t gate_biases = 4352;
  constexpr std::size_t up_weight = 4608;
  constexpr std::size_t up_scales = 8704;
  constexpr std::size_t up_biases = 8960;
  constexpr std::size_t down_weight = 9216;
  constexpr std::size_t down_scales = 13312;
  constexpr std::size_t down_biases = 13568;

  const std::string layout =
      "{\n"
      "  \"expertCount\": 3,\n"
      "  \"layerCount\": 1,\n"
      "  \"expertStride\": 16384,\n"
      "  \"sections\": [\n"
      "    {\"name\":\"gate_proj.weight\",\"dtype\":\"U32\",\"shape\":[64,16],\"offset\":0,\"size\":4096},\n"
      "    {\"name\":\"gate_proj.scales\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":4096,\"size\":256},\n"
      "    {\"name\":\"gate_proj.biases\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":4352,\"size\":256},\n"
      "    {\"name\":\"up_proj.weight\",\"dtype\":\"U32\",\"shape\":[64,16],\"offset\":4608,\"size\":4096},\n"
      "    {\"name\":\"up_proj.scales\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":8704,\"size\":256},\n"
      "    {\"name\":\"up_proj.biases\",\"dtype\":\"BF16\",\"shape\":[64,2],\"offset\":8960,\"size\":256},\n"
      "    {\"name\":\"down_proj.weight\",\"dtype\":\"U32\",\"shape\":[128,8],\"offset\":9216,\"size\":4096},\n"
      "    {\"name\":\"down_proj.scales\",\"dtype\":\"BF16\",\"shape\":[128,1],\"offset\":13312,\"size\":256},\n"
      "    {\"name\":\"down_proj.biases\",\"dtype\":\"BF16\",\"shape\":[128,1],\"offset\":13568,\"size\":256}\n"
      "  ],\n"
      "  \"linearLayers\": [true]\n"
      "}\n";

  write_text(root / "packed_experts" / "layout.json", layout);

  std::ofstream layer(
      root / "packed_experts" / "layer_00.bin",
      std::ios::binary);
  if (!layer) throw std::runtime_error("unable to create expert layer");

  for (std::size_t expert = 0; expert < 3; ++expert) {
    std::vector<std::byte> blob(kExpertStride);

    fill_qpack_projection(
        blob, 1, expert, 64, 16, 2,
        gate_weight, gate_scales, gate_biases);
    fill_qpack_projection(
        blob, 2, expert, 64, 16, 2,
        up_weight, up_scales, up_biases);
    fill_qpack_projection(
        blob, 3, expert, 128, 8, 1,
        down_weight, down_scales, down_biases);

    layer.write(
        reinterpret_cast<const char*>(blob.data()),
        static_cast<std::streamsize>(blob.size()));
  }
  layer.close();

  const auto layout_size =
      fs::file_size(root / "packed_experts" / "layout.json");
  const auto layer_size =
      fs::file_size(root / "packed_experts" / "layer_00.bin");

  const std::string manifest =
      "{\n"
      "  \"magic\": \"QPACK\",\n"
      "  \"version\": 1,\n"
      "  \"modelName\": \"qwen3_next\",\n"
      "  \"sourceCheckpoint\": \"osm25b-shared-expert-fixture\",\n"
      "  \"quantBits\": 4,\n"
      "  \"quantGroupSize\": 64,\n"
      "  \"files\": {\n"
      "    \"packed_experts/layout.json\": " + std::to_string(layout_size) + ",\n"
      "    \"packed_experts/layer_00.bin\": " + std::to_string(layer_size) + "\n"
      "  }\n"
      "}\n";

  write_text(root / "manifest.json", manifest);
  return root;
}

struct CpuExpertOracle {
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> down;
};

CpuExpertOracle make_routed_oracle(
    QpackExpertStorage& storage,
    std::uint32_t expert) {
  ExpertCache cache(storage, storage.expert_stride_bytes(), 1U);
  const std::vector<std::uint32_t> pick{expert};
  const auto entries = cache.fetch(0, pick);
  require(entries.size() == 1U, "oracle expert fetch failed");

  const auto gate =
      bind_qpack_q4_projection(storage.reader(), entries.front(), "gate_proj");
  const auto up =
      bind_qpack_q4_projection(storage.reader(), entries.front(), "up_proj");
  const auto down =
      bind_qpack_q4_projection(storage.reader(), entries.front(), "down_proj");

  CpuExpertOracle oracle;
  oracle.gate = cpu::dequantize_affine_rows(
      gate.packed, gate.out_dim, gate.packed_cols, gate.scales, gate.biases,
      cpu::AffineQuantSpec{.bits = 4, .group_size = 64});
  oracle.up = cpu::dequantize_affine_rows(
      up.packed, up.out_dim, up.packed_cols, up.scales, up.biases,
      cpu::AffineQuantSpec{.bits = 4, .group_size = 64});
  oracle.down = cpu::dequantize_affine_rows(
      down.packed, down.out_dim, down.packed_cols, down.scales, down.biases,
      cpu::AffineQuantSpec{.bits = 4, .group_size = 64});
  return oracle;
}

std::vector<float> run_routed_oracle(
    const CpuExpertOracle& oracle,
    std::span<const float> x) {
  auto hidden = cpu::matvec_row_major(oracle.gate, 64, 128, x);
  const auto up = cpu::matvec_row_major(oracle.up, 64, 128, x);
  cpu::swiglu_inplace(hidden, up);
  return cpu::matvec_row_major(oracle.down, 128, 64, hidden);
}

struct SharedQ4Fixture {
  std::vector<std::uint32_t> gate_packed;
  std::vector<float> gate_scales;
  std::vector<float> gate_biases;

  std::vector<std::uint32_t> up_packed;
  std::vector<float> up_scales;
  std::vector<float> up_biases;

  std::vector<std::uint32_t> down_packed;
  std::vector<float> down_scales;
  std::vector<float> down_biases;

  std::vector<float> scalar_gate;

  VulkanSharedExpertWeights views() const {
    return {
        .gate = {
            .packed = gate_packed,
            .scales = gate_scales,
            .biases = gate_biases,
            .out_dim = kIntermediate,
            .packed_cols = kHidden / 8U,
            .group_size = kGroupSize,
        },
        .up = {
            .packed = up_packed,
            .scales = up_scales,
            .biases = up_biases,
            .out_dim = kIntermediate,
            .packed_cols = kHidden / 8U,
            .group_size = kGroupSize,
        },
        .down = {
            .packed = down_packed,
            .scales = down_scales,
            .biases = down_biases,
            .out_dim = kHidden,
            .packed_cols = kIntermediate / 8U,
            .group_size = kGroupSize,
        },
        .scalar_gate = scalar_gate,
    };
  }
};

void fill_host_projection(
    std::vector<std::uint32_t>& packed,
    std::vector<float>& scales,
    std::vector<float>& biases,
    std::size_t seed,
    std::size_t out_dim,
    std::size_t packed_cols) {
  const auto in_dim = packed_cols * 8U;
  const auto groups_per_row = in_dim / kGroupSize;

  packed.resize(out_dim * packed_cols);
  scales.resize(out_dim * groups_per_row);
  biases.resize(out_dim * groups_per_row);

  for (std::size_t row = 0; row < out_dim; ++row) {
    for (std::size_t word = 0; word < packed_cols; ++word) {
      packed[row * packed_cols + word] =
          pack_word(seed, 0, row, word);
    }
    for (std::size_t group = 0; group < groups_per_row; ++group) {
      const auto i = row * groups_per_row + group;
      scales[i] = scale_value(seed, 0, row, group);
      biases[i] = bias_value(seed, 0, row, group);
    }
  }
}

SharedQ4Fixture make_shared_fixture() {
  SharedQ4Fixture fixture;
  fill_host_projection(
      fixture.gate_packed,
      fixture.gate_scales,
      fixture.gate_biases,
      7,
      kIntermediate,
      kHidden / 8U);
  fill_host_projection(
      fixture.up_packed,
      fixture.up_scales,
      fixture.up_biases,
      8,
      kIntermediate,
      kHidden / 8U);
  fill_host_projection(
      fixture.down_packed,
      fixture.down_scales,
      fixture.down_biases,
      9,
      kHidden,
      kIntermediate / 8U);

  fixture.scalar_gate.resize(kHidden);
  for (std::size_t i = 0; i < kHidden; ++i) {
    fixture.scalar_gate[i] =
        static_cast<float>(
            static_cast<int>((i * 7U) % 17U) - 8) * 0.015625F;
  }
  return fixture;
}

SharedExpertResult run_shared_oracle(
    const SharedQ4Fixture& fixture,
    std::span<const float> x) {
  const auto gate = cpu::dequantize_affine_rows(
      fixture.gate_packed,
      kIntermediate,
      kHidden / 8U,
      fixture.gate_scales,
      fixture.gate_biases,
      cpu::AffineQuantSpec{.bits = 4, .group_size = kGroupSize});
  const auto up = cpu::dequantize_affine_rows(
      fixture.up_packed,
      kIntermediate,
      kHidden / 8U,
      fixture.up_scales,
      fixture.up_biases,
      cpu::AffineQuantSpec{.bits = 4, .group_size = kGroupSize});
  const auto down = cpu::dequantize_affine_rows(
      fixture.down_packed,
      kHidden,
      kIntermediate / 8U,
      fixture.down_scales,
      fixture.down_biases,
      cpu::AffineQuantSpec{.bits = 4, .group_size = kGroupSize});

  return run_shared_expert_cpu(
      x,
      SharedExpertWeights{
          .gate_proj = gate,
          .up_proj = up,
          .down_proj = down,
          .scalar_gate = fixture.scalar_gate,
      },
      SharedExpertConfig{
          .hidden_size = kHidden,
          .intermediate_size = kIntermediate,
      });
}

float max_abs_error(
    std::span<const float> actual,
    std::span<const float> expected) {
  require(actual.size() == expected.size(), "result size mismatch");
  float result = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    result = std::max(result, std::fabs(actual[i] - expected[i]));
  }
  return result;
}

std::vector<float> token(std::size_t seed) {
  std::vector<float> x(kHidden);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] =
        static_cast<float>(
            static_cast<int>((i * (seed + 5U)) % 31U) - 15) *
        0.03125F;
  }
  return x;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm25b-shared-expert";

  try {
    QpackExpertStorage storage(make_qpack(root));
    const auto routed1 = make_routed_oracle(storage, 1);
    const auto routed2 = make_routed_oracle(storage, 2);
    const auto shared_fixture = make_shared_fixture();

    std::string diagnostic;
    auto context = VulkanComputeContext::create(&diagnostic);
    if (!context.has_value()) {
      if (require_vulkan_compute()) {
        throw std::runtime_error(
            "ORBI_REQUIRE_VULKAN_COMPUTE=1 but no context: " + diagnostic);
      }
      fs::remove_all(root);
      std::cout
          << "OSM-25B Vulkan shared expert: PASS (no compute device on host)\n";
      return 0;
    }

    auto shared = VulkanResidentSharedExpert::create(
        *context,
        shared_fixture.views(),
        &diagnostic);
    require(shared.has_value(), diagnostic);
    require(shared->valid(), "shared expert must be valid");
    require(shared->input_dim() == kHidden, "shared input dim mismatch");
    require(
        shared->intermediate_dim() == kIntermediate,
        "shared intermediate dim mismatch");
    require(shared->output_dim() == kHidden, "shared output dim mismatch");
    require(
        shared->accounted_vulkan_bytes() > 0U,
        "shared expert Vulkan accounting must be non-zero");

    ExpertCache host_cache(
        storage,
        storage.expert_stride_bytes(),
        1U);
    VulkanResidentExpertCache gpu_cache(
        *context,
        storage.reader(),
        host_cache,
        2U * kAccountedExpertBytes);

    auto x = token(4);
    x[0] = 1.0F;

    std::vector<float> router_weight(3U * kHidden, 0.0F);
    router_weight[0U * kHidden] = 1.0F;
    router_weight[1U * kHidden] = 3.0F;
    router_weight[2U * kHidden] = 2.0F;

    const QwenRouterConfig router_config{
        .hidden_size = kHidden,
        .expert_count = 3,
        .top_k = 2,
        .norm_topk_prob = true,
    };

    const auto route =
        route_qwen_top_k_cpu(x, router_weight, router_config);
    require(route.picks.size() == 2U, "route size mismatch");
    require(route.picks[0].expert == 1U, "expected expert 1 top-1");
    require(route.picks[1].expert == 2U, "expected expert 2 top-2");

    const auto e1 = run_routed_oracle(routed1, x);
    const auto e2 = run_routed_oracle(routed2, x);
    const auto shared_expected = run_shared_oracle(shared_fixture, x);
    require(shared_expected.executed, shared_expected.diagnostic);

    std::vector<float> expected(kHidden, 0.0F);
    for (const auto& pick : route.picks) {
      const auto& expert_values = pick.expert == 1U ? e1 : e2;
      for (std::size_t i = 0; i < expected.size(); ++i) {
        expected[i] += pick.weight * expert_values[i];
      }
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
      expected[i] += shared_expected.values[i];
    }

    const auto first = run_qwen_sparse_moe_vulkan_accum(
        *context,
        gpu_cache,
        *shared,
        0,
        x,
        router_weight,
        router_config);
    require(first.executed, first.diagnostic);
    require(first.routing.picks.size() == 2U, "result route size mismatch");
    require(
        std::fabs(first.shared_gate - shared_expected.scalar_gate) <= 1e-6F,
        "shared scalar gate mismatch");

    const auto first_error = max_abs_error(first.values, expected);
    require(
        first_error <= 0.03F,
        "complete Qwen sparse MoE CPU parity failed");

    auto stats = gpu_cache.stats();
    require(stats.hits == 0U, "cold sparse pass should have zero GPU hits");
    require(stats.misses == 2U, "cold sparse pass should miss two experts");
    require(stats.loads == 2U, "cold sparse pass should load two experts");
    require(stats.evictions == 0U, "cold sparse pass should not evict");

    const auto host_before_repeat = host_cache.stats();

    const auto second = run_qwen_sparse_moe_vulkan_accum(
        *context,
        gpu_cache,
        *shared,
        0,
        x,
        router_weight,
        router_config);
    require(second.executed, second.diagnostic);
    require(
        std::fabs(second.shared_gate - shared_expected.scalar_gate) <= 1e-6F,
        "repeat shared scalar gate mismatch");

    const auto second_error = max_abs_error(second.values, expected);
    require(
        second_error <= 0.03F,
        "repeat complete Qwen sparse MoE CPU parity failed");

    const auto host_after_repeat = host_cache.stats();
    require(
        host_after_repeat.hits == host_before_repeat.hits &&
        host_after_repeat.misses == host_before_repeat.misses,
        "warm routed experts should bypass the host cache");

    stats = gpu_cache.stats();
    require(stats.hits == 2U, "warm pass should add two GPU hits");
    require(stats.misses == 2U, "warm pass must not add misses");
    require(stats.loads == 2U, "warm pass must not add loads");
    require(stats.evictions == 0U, "warm pass must not evict");

    fs::remove_all(root);
    std::cout
        << "OSM-25B Vulkan shared expert: PASS\n"
        << "  device=" << context->info().device_name << "\n"
        << "  selected_experts="
        << first.routing.picks[0].expert << ","
        << first.routing.picks[1].expert << "\n"
        << "  shared_gate=" << first.shared_gate << "\n"
        << "  shared_vulkan_bytes=" << shared->accounted_vulkan_bytes() << "\n"
        << "  hidden_uploads_per_pass=1\n"
        << "  shared_output_readbacks=0\n"
        << "  final_sparse_readbacks_per_pass=1\n"
        << "  first_max_abs_error=" << first_error << "\n"
        << "  repeat_max_abs_error=" << second_error << "\n"
        << "  gpu_hits=" << stats.hits
        << " misses=" << stats.misses
        << " loads=" << stats.loads
        << " evictions=" << stats.evictions << "\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-25B Vulkan shared expert: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
