#include "orbi/streammoe/model/checkpoint_sparse_moe.hpp"

#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "orbi/streammoe/cpu/reference_ops.hpp"

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

void require_matrix(
    const MlxAffineModule& module,
    std::size_t rows,
    std::size_t cols,
    const char* label) {
  if (module.logical_shape.size() != 2U ||
      module.logical_shape[0] != rows ||
      module.logical_shape[1] != cols) {
    throw std::runtime_error(
        std::string("checkpoint sparse MoE: ") + label +
        " logical shape mismatch");
  }
}

VulkanSharedExpertProjectionView q4_view(
    const MlxAffineModule& module,
    const char* label) {
  if (module.quant.bits != 4U) {
    throw std::runtime_error(
        std::string("checkpoint sparse MoE: ") + label +
        " must be affine Q4 for the current Vulkan shared-expert path");
  }
  if (module.packed_shape.size() != 2U) {
    throw std::runtime_error(
        std::string("checkpoint sparse MoE: ") + label +
        " packed weight must be rank-2");
  }

  return {
      .packed = module.packed,
      .scales = module.scales,
      .biases = module.biases,
      .out_dim = module.row_count,
      .packed_cols = module.packed_shape.back(),
      .group_size = module.quant.group_size,
  };
}

}  // namespace

std::vector<float> dequantize_mlx_affine_module_cpu(
    const MlxAffineModule& module) {
  if (module.quant.bits != 4U && module.quant.bits != 8U) {
    throw std::invalid_argument(
        "checkpoint sparse MoE: CPU affine bridge supports 4-bit/8-bit only");
  }
  if (module.packed_shape.size() != 2U ||
      module.logical_shape.size() != 2U ||
      module.row_count == 0U ||
      module.packed_shape.back() == 0U) {
    throw std::invalid_argument(
        "checkpoint sparse MoE: affine module must be a non-empty rank-2 matrix");
  }

  const auto values = cpu::dequantize_affine_rows(
      module.packed,
      module.row_count,
      module.packed_shape.back(),
      module.scales,
      module.biases,
      cpu::AffineQuantSpec{
          .bits = module.quant.bits,
          .group_size = module.quant.group_size,
      });

  if (module.logical_shape[0] != module.row_count ||
      module.logical_shape[1] != module.logical_in_dim ||
      values.size() != module.row_count * module.logical_in_dim) {
    throw std::runtime_error(
        "checkpoint sparse MoE: dequantized matrix disagrees with module geometry");
  }

  return values;
}

struct QwenCheckpointSparseMoeLayer::Impl {
  std::size_t layer_index{};
  QwenRouterConfig router_config{};
  std::vector<float> router_weight;
  VulkanResidentSharedExpert shared_expert;
};

QwenCheckpointSparseMoeLayer::QwenCheckpointSparseMoeLayer() = default;
QwenCheckpointSparseMoeLayer::~QwenCheckpointSparseMoeLayer() = default;
QwenCheckpointSparseMoeLayer::QwenCheckpointSparseMoeLayer(
    QwenCheckpointSparseMoeLayer&&) noexcept = default;
QwenCheckpointSparseMoeLayer& QwenCheckpointSparseMoeLayer::operator=(
    QwenCheckpointSparseMoeLayer&&) noexcept = default;

QwenCheckpointSparseMoeLayer::QwenCheckpointSparseMoeLayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointSparseMoeLayer>
QwenCheckpointSparseMoeLayer::create(
    VulkanComputeContext& context,
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "checkpoint sparse MoE requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    if (binding.layer_index >= config.num_hidden_layers) {
      throw std::runtime_error(
          "checkpoint sparse MoE: layer index exceeds model layer count");
    }

    const auto& moe = binding.moe;

    require_matrix(
        moe.router,
        config.num_experts,
        config.hidden_size,
        "router");
    require_matrix(
        moe.shared_gate,
        config.shared_expert_intermediate_size,
        config.hidden_size,
        "shared gate");
    require_matrix(
        moe.shared_up,
        config.shared_expert_intermediate_size,
        config.hidden_size,
        "shared up");
    require_matrix(
        moe.shared_down,
        config.hidden_size,
        config.shared_expert_intermediate_size,
        "shared down");
    require_matrix(
        moe.shared_expert_gate,
        1U,
        config.hidden_size,
        "shared scalar gate");

    auto router_weight =
        dequantize_mlx_affine_module_cpu(moe.router);
    auto shared_scalar =
        dequantize_mlx_affine_module_cpu(moe.shared_expert_gate);

    if (shared_scalar.size() != config.hidden_size) {
      throw std::runtime_error(
          "checkpoint sparse MoE: shared scalar gate dequant size mismatch");
    }

    auto shared = VulkanResidentSharedExpert::create(
        context,
        VulkanSharedExpertWeights{
            .gate = q4_view(moe.shared_gate, "shared gate"),
            .up = q4_view(moe.shared_up, "shared up"),
            .down = q4_view(moe.shared_down, "shared down"),
            .scalar_gate = shared_scalar,
        },
        diagnostic);

    if (!shared.has_value()) {
      if (diagnostic != nullptr && diagnostic->empty()) {
        *diagnostic =
            "checkpoint sparse MoE: failed to create Vulkan shared expert";
      }
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->router_config = {
        .hidden_size = config.hidden_size,
        .expert_count = config.num_experts,
        .top_k = config.num_experts_per_tok,
        .norm_topk_prob = config.norm_topk_prob,
    };
    impl->router_weight = std::move(router_weight);
    impl->shared_expert = std::move(*shared);

    const auto expected_router =
        config.num_experts * config.hidden_size;
    if (impl->router_weight.size() != expected_router) {
      throw std::runtime_error(
          "checkpoint sparse MoE: router dequant size mismatch");
    }

    set_diagnostic(
        diagnostic,
        "checkpoint-bound Qwen sparse MoE layer created");
    return QwenCheckpointSparseMoeLayer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("checkpoint sparse MoE creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "checkpoint sparse MoE creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointSparseMoeLayer::valid() const noexcept {
  if (impl_ == nullptr ||
      !impl_->shared_expert.valid() ||
      impl_->router_config.hidden_size == 0U ||
      impl_->router_config.expert_count == 0U ||
      impl_->router_config.top_k == 0U) {
    return false;
  }

  if (impl_->router_config.expert_count >
      std::numeric_limits<std::size_t>::max() /
          impl_->router_config.hidden_size) {
    return false;
  }

  return impl_->router_weight.size() ==
         impl_->router_config.expert_count *
             impl_->router_config.hidden_size;
}

std::size_t QwenCheckpointSparseMoeLayer::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

const QwenRouterConfig&
QwenCheckpointSparseMoeLayer::router_config() const noexcept {
  static const QwenRouterConfig empty{};
  return impl_ != nullptr ? impl_->router_config : empty;
}

std::span<const float>
QwenCheckpointSparseMoeLayer::router_weight() const noexcept {
  if (impl_ == nullptr) return {};
  return impl_->router_weight;
}

const VulkanResidentSharedExpert*
QwenCheckpointSparseMoeLayer::shared_expert() const noexcept {
  if (!valid()) return nullptr;
  return &impl_->shared_expert;
}

QwenSparseMoeResult QwenCheckpointSparseMoeLayer::run(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::span<const float> hidden) noexcept {
  QwenSparseMoeResult result;
  if (!valid()) {
    result.diagnostic =
        "checkpoint-bound Qwen sparse MoE layer is not valid";
    return result;
  }

  return run_qwen_sparse_moe_vulkan_accum(
      context,
      expert_cache,
      impl_->shared_expert,
      static_cast<std::uint32_t>(impl_->layer_index),
      hidden,
      impl_->router_weight,
      impl_->router_config);
}

}  // namespace orbi::streammoe
