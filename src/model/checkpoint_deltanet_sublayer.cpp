#include "orbi/streammoe/model/checkpoint_deltanet_sublayer.hpp"

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "orbi/streammoe/backend/vulkan_rmsnorm.hpp"

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

std::vector<float> add_residual(
    std::span<const float> residual,
    std::span<const float> branch) {
  if (residual.size() != branch.size()) {
    throw std::invalid_argument(
        "checkpoint DeltaNet sublayer residual size mismatch");
  }

  std::vector<float> output(residual.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] = residual[i] + branch[i];
  }
  return output;
}

}  // namespace

struct QwenCheckpointDeltaNetSublayer::Impl {
  std::size_t layer_index{};
  std::size_t hidden_size{};
  float rms_eps{1e-6F};
  std::vector<float> input_norm;
  QwenCheckpointGatedDeltaNet delta;
};

QwenCheckpointDeltaNetSublayer::QwenCheckpointDeltaNetSublayer() = default;
QwenCheckpointDeltaNetSublayer::~QwenCheckpointDeltaNetSublayer() = default;
QwenCheckpointDeltaNetSublayer::QwenCheckpointDeltaNetSublayer(
    QwenCheckpointDeltaNetSublayer&&) noexcept = default;
QwenCheckpointDeltaNetSublayer&
QwenCheckpointDeltaNetSublayer::operator=(
    QwenCheckpointDeltaNetSublayer&&) noexcept = default;

QwenCheckpointDeltaNetSublayer::QwenCheckpointDeltaNetSublayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointDeltaNetSublayer>
QwenCheckpointDeltaNetSublayer::create(
    VulkanComputeContext& context,
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    float rms_eps,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "checkpoint DeltaNet sublayer requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    if (!binding.is_linear || !binding.delta.has_value()) {
      throw std::runtime_error(
          "checkpoint DeltaNet sublayer requires a linear layer binding");
    }
    if (binding.layer_index >= config.num_hidden_layers ||
        !config.is_linear_layer(binding.layer_index)) {
      throw std::runtime_error(
          "checkpoint DeltaNet sublayer layer/config classification mismatch");
    }
    if (binding.input_norm.size() != config.hidden_size ||
        config.hidden_size == 0U) {
      throw std::runtime_error(
          "checkpoint DeltaNet sublayer input RMSNorm shape mismatch");
    }
    if (!std::isfinite(rms_eps) || rms_eps < 0.0F) {
      throw std::runtime_error(
          "checkpoint DeltaNet sublayer RMS epsilon must be finite/non-negative");
    }

    std::string local;
    auto delta = QwenCheckpointGatedDeltaNet::create(
        binding, config, rms_eps, &local);
    if (!delta.has_value()) {
      set_diagnostic(
          diagnostic,
          "checkpoint DeltaNet sublayer branch creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->hidden_size = config.hidden_size;
    impl->rms_eps = rms_eps;
    impl->input_norm = binding.input_norm;
    impl->delta = std::move(*delta);

    set_diagnostic(
        diagnostic,
        "checkpoint input RMSNorm + DeltaNet residual sublayer created");
    return QwenCheckpointDeltaNetSublayer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("checkpoint DeltaNet sublayer creation failed: ") +
            e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "checkpoint DeltaNet sublayer creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointDeltaNetSublayer::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->hidden_size != 0U &&
         impl_->input_norm.size() == impl_->hidden_size &&
         impl_->delta.valid() &&
         std::isfinite(impl_->rms_eps) &&
         impl_->rms_eps >= 0.0F;
}

std::size_t QwenCheckpointDeltaNetSublayer::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

std::size_t QwenCheckpointDeltaNetSublayer::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->hidden_size : 0U;
}

const QwenGatedDeltaNetState&
QwenCheckpointDeltaNetSublayer::state() const noexcept {
  static const QwenGatedDeltaNetState empty{};
  return impl_ != nullptr ? impl_->delta.state() : empty;
}

void QwenCheckpointDeltaNetSublayer::reset_state() noexcept {
  if (impl_ != nullptr) {
    impl_->delta.reset_state();
  }
}

QwenCheckpointDeltaNetSublayerResult
QwenCheckpointDeltaNetSublayer::run(
    VulkanComputeContext& context,
    std::span<const float> residual) noexcept {
  QwenCheckpointDeltaNetSublayerResult result;

  if (!valid()) {
    result.diagnostic = "checkpoint DeltaNet sublayer is not valid";
    return result;
  }
  if (!context.valid()) {
    result.diagnostic =
        "checkpoint DeltaNet sublayer requires a valid Vulkan context";
    return result;
  }
  if (residual.size() != impl_->hidden_size) {
    result.diagnostic =
        "checkpoint DeltaNet sublayer residual shape mismatch";
    return result;
  }

  try {
    const auto norm = run_vulkan_rms_norm(
        context,
        residual,
        1U,
        impl_->hidden_size,
        impl_->input_norm,
        impl_->rms_eps);
    if (!norm.executed) {
      result.diagnostic =
          "checkpoint DeltaNet sublayer RMSNorm failed: " +
          norm.diagnostic;
      return result;
    }

    result.normalized = norm.values;
    result.branch = impl_->delta.run(result.normalized);
    if (!result.branch.executed) {
      result.diagnostic =
          "checkpoint DeltaNet branch failed: " +
          result.branch.diagnostic;
      return result;
    }

    result.values = add_residual(residual, result.branch.values);
    result.executed = true;
    result.diagnostic =
        "checkpoint RMSNorm -> Gated DeltaNet -> residual sublayer executed";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("checkpoint DeltaNet sublayer execution failed: ") +
        e.what();
    result.values.clear();
    return result;
  } catch (...) {
    result.diagnostic =
        "checkpoint DeltaNet sublayer execution encountered an unknown exception";
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
