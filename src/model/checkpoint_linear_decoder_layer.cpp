#include "orbi/streammoe/model/checkpoint_linear_decoder_layer.hpp"

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

}  // namespace

struct QwenCheckpointLinearDecoderLayer::Impl {
  std::size_t layer_index{};
  std::size_t hidden_size{};
  float rms_eps{1e-6F};
  QwenCheckpointDeltaNetSublayer delta;
  QwenCheckpointMoeSublayer moe;
};

QwenCheckpointLinearDecoderLayer::QwenCheckpointLinearDecoderLayer() = default;
QwenCheckpointLinearDecoderLayer::~QwenCheckpointLinearDecoderLayer() = default;
QwenCheckpointLinearDecoderLayer::QwenCheckpointLinearDecoderLayer(
    QwenCheckpointLinearDecoderLayer&&) noexcept = default;
QwenCheckpointLinearDecoderLayer&
QwenCheckpointLinearDecoderLayer::operator=(
    QwenCheckpointLinearDecoderLayer&&) noexcept = default;

QwenCheckpointLinearDecoderLayer::QwenCheckpointLinearDecoderLayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointLinearDecoderLayer>
QwenCheckpointLinearDecoderLayer::create(
    VulkanComputeContext& context,
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    float rms_eps,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "linear decoder layer requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    if (!binding.is_linear || !binding.delta.has_value()) {
      throw std::runtime_error(
          "linear decoder layer requires a Gated DeltaNet binding");
    }
    if (binding.layer_index >= config.num_hidden_layers ||
        !config.is_linear_layer(binding.layer_index)) {
      throw std::runtime_error(
          "linear decoder layer layer/config classification mismatch");
    }
    if (config.hidden_size == 0U ||
        binding.input_norm.size() != config.hidden_size ||
        binding.post_attention_norm.size() != config.hidden_size) {
      throw std::runtime_error(
          "linear decoder layer RMSNorm geometry mismatch");
    }
    if (!std::isfinite(rms_eps) || rms_eps < 0.0F) {
      throw std::runtime_error(
          "linear decoder layer RMS epsilon must be finite/non-negative");
    }

    std::string local;
    auto delta = QwenCheckpointDeltaNetSublayer::create(
        context, binding, config, rms_eps, &local);
    if (!delta.has_value()) {
      set_diagnostic(
          diagnostic,
          "linear decoder DeltaNet sublayer creation failed: " + local);
      return std::nullopt;
    }

    local.clear();
    auto moe = QwenCheckpointMoeSublayer::create(
        context, binding, config, rms_eps, &local);
    if (!moe.has_value()) {
      set_diagnostic(
          diagnostic,
          "linear decoder MoE sublayer creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->hidden_size = config.hidden_size;
    impl->rms_eps = rms_eps;
    impl->delta = std::move(*delta);
    impl->moe = std::move(*moe);

    set_diagnostic(
        diagnostic,
        "checkpoint-bound complete linear Qwen decoder layer created");
    return QwenCheckpointLinearDecoderLayer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("linear decoder layer creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "linear decoder layer creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointLinearDecoderLayer::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->hidden_size != 0U &&
         impl_->delta.valid() &&
         impl_->moe.valid() &&
         impl_->delta.layer_index() == impl_->layer_index &&
         impl_->moe.layer_index() == impl_->layer_index &&
         std::isfinite(impl_->rms_eps) &&
         impl_->rms_eps >= 0.0F;
}

std::size_t QwenCheckpointLinearDecoderLayer::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

std::size_t QwenCheckpointLinearDecoderLayer::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->hidden_size : 0U;
}

const QwenGatedDeltaNetState&
QwenCheckpointLinearDecoderLayer::delta_state() const noexcept {
  static const QwenGatedDeltaNetState empty{};
  return impl_ != nullptr ? impl_->delta.state() : empty;
}

void QwenCheckpointLinearDecoderLayer::reset_state() noexcept {
  if (impl_ != nullptr) {
    impl_->delta.reset_state();
  }
}

QwenCheckpointLinearDecoderLayerResult
QwenCheckpointLinearDecoderLayer::run(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::span<const float> hidden) noexcept {
  QwenCheckpointLinearDecoderLayerResult result;

  if (!valid()) {
    result.diagnostic = "linear decoder layer is not valid";
    return result;
  }
  if (!context.valid()) {
    result.diagnostic =
        "linear decoder layer requires a valid Vulkan context";
    return result;
  }
  if (hidden.size() != impl_->hidden_size) {
    result.diagnostic = "linear decoder layer hidden shape mismatch";
    return result;
  }

  result.attention_branch = impl_->delta.run(context, hidden);
  if (!result.attention_branch.executed) {
    result.diagnostic =
        "linear decoder DeltaNet sublayer failed: " +
        result.attention_branch.diagnostic;
    return result;
  }

  result.moe_branch = impl_->moe.run(
      context,
      expert_cache,
      result.attention_branch.values);
  if (!result.moe_branch.executed) {
    result.diagnostic =
        "linear decoder MoE sublayer failed: " +
        result.moe_branch.diagnostic;
    return result;
  }

  result.values = result.moe_branch.values;
  result.executed = true;
  result.diagnostic =
      "complete checkpoint-bound linear Qwen decoder layer executed";
  return result;
}

}  // namespace orbi::streammoe
