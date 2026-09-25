#include "orbi/streammoe/model/checkpoint_moe_sublayer.hpp"

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "orbi/streammoe/backend/vulkan_rmsnorm.hpp"

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

}  // namespace

std::vector<float> add_residual_cpu(
    std::span<const float> residual,
    std::span<const float> branch) {
  if (residual.size() != branch.size()) {
    throw std::invalid_argument("residual add size mismatch");
  }

  std::vector<float> output(residual.size());
  for (std::size_t i = 0; i < residual.size(); ++i) {
    output[i] = residual[i] + branch[i];
  }
  return output;
}

struct QwenCheckpointMoeSublayer::Impl {
  std::size_t layer_index{};
  std::size_t hidden_size{};
  float rms_eps{1e-6F};
  std::vector<float> post_attention_norm;
  QwenCheckpointSparseMoeLayer sparse_moe;
};

QwenCheckpointMoeSublayer::QwenCheckpointMoeSublayer() = default;
QwenCheckpointMoeSublayer::~QwenCheckpointMoeSublayer() = default;
QwenCheckpointMoeSublayer::QwenCheckpointMoeSublayer(
    QwenCheckpointMoeSublayer&&) noexcept = default;
QwenCheckpointMoeSublayer& QwenCheckpointMoeSublayer::operator=(
    QwenCheckpointMoeSublayer&&) noexcept = default;

QwenCheckpointMoeSublayer::QwenCheckpointMoeSublayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointMoeSublayer>
QwenCheckpointMoeSublayer::create(
    VulkanComputeContext& context,
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    float rms_eps,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "checkpoint MoE sublayer requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    if (binding.layer_index >= config.num_hidden_layers) {
      throw std::runtime_error(
          "checkpoint MoE sublayer layer index exceeds model layer count");
    }
    if (config.hidden_size == 0U ||
        binding.post_attention_norm.size() != config.hidden_size) {
      throw std::runtime_error(
          "checkpoint MoE sublayer post-attention RMSNorm shape mismatch");
    }
    if (!std::isfinite(rms_eps) || rms_eps < 0.0F) {
      throw std::runtime_error(
          "checkpoint MoE sublayer RMS epsilon must be finite/non-negative");
    }

    std::string local;
    auto sparse = QwenCheckpointSparseMoeLayer::create(
        context, binding, config, &local);
    if (!sparse.has_value()) {
      set_diagnostic(
          diagnostic,
          "checkpoint MoE sublayer sparse runtime creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->hidden_size = config.hidden_size;
    impl->rms_eps = rms_eps;
    impl->post_attention_norm = binding.post_attention_norm;
    impl->sparse_moe = std::move(*sparse);

    set_diagnostic(
        diagnostic,
        "checkpoint-bound post-attention MoE sublayer created");
    return QwenCheckpointMoeSublayer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("checkpoint MoE sublayer creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "checkpoint MoE sublayer creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointMoeSublayer::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->hidden_size != 0U &&
         impl_->post_attention_norm.size() == impl_->hidden_size &&
         impl_->sparse_moe.valid() &&
         std::isfinite(impl_->rms_eps) &&
         impl_->rms_eps >= 0.0F;
}

std::size_t QwenCheckpointMoeSublayer::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

std::size_t QwenCheckpointMoeSublayer::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->hidden_size : 0U;
}

float QwenCheckpointMoeSublayer::rms_eps() const noexcept {
  return impl_ != nullptr ? impl_->rms_eps : 0.0F;
}

QwenCheckpointMoeSublayerResult QwenCheckpointMoeSublayer::run(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::span<const float> residual) noexcept {
  QwenCheckpointMoeSublayerResult result;

  if (!valid()) {
    result.diagnostic = "checkpoint MoE sublayer is not valid";
    return result;
  }
  if (!context.valid()) {
    result.diagnostic = "checkpoint MoE sublayer requires a valid Vulkan context";
    return result;
  }
  if (residual.size() != impl_->hidden_size) {
    result.diagnostic = "checkpoint MoE sublayer residual shape mismatch";
    return result;
  }

  try {
    const auto norm = run_vulkan_rms_norm(
        context,
        residual,
        1U,
        impl_->hidden_size,
        impl_->post_attention_norm,
        impl_->rms_eps);
    if (!norm.executed) {
      result.diagnostic =
          "checkpoint MoE sublayer RMSNorm failed: " + norm.diagnostic;
      return result;
    }

    result.normalized = norm.values;
    result.moe = impl_->sparse_moe.run(
        context,
        expert_cache,
        result.normalized);
    if (!result.moe.executed) {
      result.diagnostic =
          "checkpoint MoE sublayer sparse MoE failed: " +
          result.moe.diagnostic;
      return result;
    }

    result.values = add_residual_cpu(residual, result.moe.values);
    result.executed = true;
    result.diagnostic =
        "checkpoint-bound RMSNorm -> sparse MoE -> residual sublayer executed";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("checkpoint MoE sublayer execution failed: ") + e.what();
    result.values.clear();
    return result;
  } catch (...) {
    result.diagnostic =
        "checkpoint MoE sublayer execution encountered an unknown exception";
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
