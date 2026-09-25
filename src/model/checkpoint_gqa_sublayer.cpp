#include "orbi/streammoe/model/checkpoint_gqa_sublayer.hpp"

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
        "checkpoint GQA sublayer residual size mismatch");
  }

  std::vector<float> output(residual.size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] = residual[i] + branch[i];
  }
  return output;
}

}  // namespace

struct QwenCheckpointGqaSublayer::Impl {
  std::size_t layer_index{};
  std::size_t hidden_size{};
  float rms_eps{1e-6F};
  std::vector<float> input_norm;
  QwenCheckpointGqa gqa;
};

QwenCheckpointGqaSublayer::QwenCheckpointGqaSublayer() = default;
QwenCheckpointGqaSublayer::~QwenCheckpointGqaSublayer() = default;
QwenCheckpointGqaSublayer::QwenCheckpointGqaSublayer(
    QwenCheckpointGqaSublayer&&) noexcept = default;
QwenCheckpointGqaSublayer&
QwenCheckpointGqaSublayer::operator=(
    QwenCheckpointGqaSublayer&&) noexcept = default;

QwenCheckpointGqaSublayer::QwenCheckpointGqaSublayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointGqaSublayer>
QwenCheckpointGqaSublayer::create(
    VulkanComputeContext& context,
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "checkpoint GQA sublayer requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    if (binding.is_linear || !binding.attention.has_value()) {
      throw std::runtime_error(
          "checkpoint GQA sublayer requires a full-attention binding");
    }
    if (binding.layer_index >= config.num_hidden_layers ||
        config.is_linear_layer(binding.layer_index)) {
      throw std::runtime_error(
          "checkpoint GQA sublayer layer/config classification mismatch");
    }
    if (config.hidden_size == 0U ||
        binding.input_norm.size() != config.hidden_size) {
      throw std::runtime_error(
          "checkpoint GQA sublayer input RMSNorm shape mismatch");
    }

    std::string local;
    auto gqa = QwenCheckpointGqa::create(
        binding,
        config,
        &local);
    if (!gqa.has_value()) {
      set_diagnostic(
          diagnostic,
          "checkpoint GQA sublayer attention creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->hidden_size = config.hidden_size;
    impl->rms_eps = config.rms_norm_eps;
    impl->input_norm = binding.input_norm;
    impl->gqa = std::move(*gqa);

    set_diagnostic(
        diagnostic,
        "checkpoint input RMSNorm + gated GQA residual sublayer created");
    return QwenCheckpointGqaSublayer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("checkpoint GQA sublayer creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "checkpoint GQA sublayer creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointGqaSublayer::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->hidden_size != 0U &&
         impl_->input_norm.size() == impl_->hidden_size &&
         impl_->gqa.valid();
}

std::size_t QwenCheckpointGqaSublayer::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

std::size_t QwenCheckpointGqaSublayer::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->hidden_size : 0U;
}

const QwenGqaState& QwenCheckpointGqaSublayer::state() const noexcept {
  static const QwenGqaState empty{};
  return impl_ != nullptr ? impl_->gqa.state() : empty;
}

void QwenCheckpointGqaSublayer::reset_state() noexcept {
  if (impl_ != nullptr) {
    impl_->gqa.reset_state();
  }
}

QwenCheckpointGqaSublayerResult
QwenCheckpointGqaSublayer::run(
    VulkanComputeContext& context,
    std::span<const float> residual) noexcept {
  QwenCheckpointGqaSublayerResult result;

  if (!valid()) {
    result.diagnostic = "checkpoint GQA sublayer is not valid";
    return result;
  }
  if (!context.valid()) {
    result.diagnostic =
        "checkpoint GQA sublayer requires a valid Vulkan context";
    return result;
  }
  if (residual.size() != impl_->hidden_size) {
    result.diagnostic =
        "checkpoint GQA sublayer residual shape mismatch";
    return result;
  }

  const auto norm = run_vulkan_rms_norm(
      context,
      residual,
      1U,
      impl_->hidden_size,
      impl_->input_norm,
      impl_->rms_eps);
  if (!norm.executed) {
    result.diagnostic =
        "checkpoint GQA sublayer RMSNorm failed: " + norm.diagnostic;
    return result;
  }

  result.normalized = norm.values;
  result.branch = impl_->gqa.run(result.normalized);
  if (!result.branch.executed) {
    result.diagnostic =
        "checkpoint GQA branch failed: " + result.branch.diagnostic;
    return result;
  }

  try {
    result.values = add_residual(residual, result.branch.values);
    result.executed = true;
    result.diagnostic =
        "checkpoint RMSNorm -> gated GQA -> residual sublayer executed";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("checkpoint GQA residual add failed: ") + e.what();
    result.values.clear();
    return result;
  }
}

}  // namespace orbi::streammoe
