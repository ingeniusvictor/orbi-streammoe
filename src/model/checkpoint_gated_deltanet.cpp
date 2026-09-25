#include "orbi/streammoe/model/checkpoint_gated_deltanet.hpp"

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "orbi/streammoe/model/checkpoint_sparse_moe.hpp"

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

}  // namespace

struct QwenCheckpointGatedDeltaNet::Impl {
  std::size_t layer_index{};
  QwenGatedDeltaNetConfig config{};

  std::vector<float> qkvz;
  std::vector<float> ba;
  std::vector<float> conv;
  std::vector<float> dt_bias;
  std::vector<float> a_log;
  std::vector<float> norm;
  std::vector<float> out_proj;

  QwenGatedDeltaNetState state;
};

QwenCheckpointGatedDeltaNet::QwenCheckpointGatedDeltaNet() = default;
QwenCheckpointGatedDeltaNet::~QwenCheckpointGatedDeltaNet() = default;
QwenCheckpointGatedDeltaNet::QwenCheckpointGatedDeltaNet(
    QwenCheckpointGatedDeltaNet&&) noexcept = default;
QwenCheckpointGatedDeltaNet& QwenCheckpointGatedDeltaNet::operator=(
    QwenCheckpointGatedDeltaNet&&) noexcept = default;

QwenCheckpointGatedDeltaNet::QwenCheckpointGatedDeltaNet(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointGatedDeltaNet>
QwenCheckpointGatedDeltaNet::create(
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    float rms_eps,
    std::string* diagnostic) noexcept {
  try {
    if (!binding.is_linear || !binding.delta.has_value()) {
      throw std::runtime_error(
          "checkpoint DeltaNet requires a linear Qwen3-Next layer binding");
    }
    if (binding.layer_index >= config.num_hidden_layers ||
        !config.is_linear_layer(binding.layer_index)) {
      throw std::runtime_error(
          "checkpoint DeltaNet layer/config classification mismatch");
    }
    if (!std::isfinite(rms_eps) || rms_eps < 0.0F) {
      throw std::runtime_error(
          "checkpoint DeltaNet RMS epsilon must be finite/non-negative");
    }

    const auto& delta = *binding.delta;
    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->config = {
        .hidden_size = config.hidden_size,
        .num_key_heads = config.linear_num_key_heads,
        .num_value_heads = config.linear_num_value_heads,
        .key_head_dim = config.linear_key_head_dim,
        .value_head_dim = config.linear_value_head_dim,
        .conv_kernel_size = config.linear_conv_kernel_dim,
        .rms_eps = rms_eps,
    };

    if (impl->config.num_key_heads == 0U ||
        impl->config.num_value_heads == 0U ||
        impl->config.num_value_heads % impl->config.num_key_heads != 0U) {
      throw std::runtime_error(
          "checkpoint DeltaNet invalid key/value head geometry");
    }

    impl->qkvz = dequantize_mlx_affine_module_cpu(delta.in_proj_qkvz);
    impl->ba = dequantize_mlx_affine_module_cpu(delta.in_proj_ba);
    impl->out_proj = dequantize_mlx_affine_module_cpu(delta.out_proj);
    impl->conv = delta.conv;
    impl->dt_bias = delta.dt_bias;
    impl->a_log = delta.a_log;
    impl->norm = delta.norm;

    const auto key_dim = impl->config.key_dim();
    const auto value_dim = impl->config.value_dim();
    const auto qkvz_out = 2U * key_dim + 2U * value_dim;
    const auto ba_out = 2U * impl->config.num_value_heads;

    if (impl->qkvz.size() != qkvz_out * impl->config.hidden_size ||
        impl->ba.size() != ba_out * impl->config.hidden_size ||
        impl->conv.size() !=
            impl->config.conv_dim() * impl->config.conv_kernel_size ||
        impl->dt_bias.size() != impl->config.num_value_heads ||
        impl->a_log.size() != impl->config.num_value_heads ||
        impl->norm.size() != impl->config.value_head_dim ||
        impl->out_proj.size() != impl->config.hidden_size * value_dim) {
      throw std::runtime_error(
          "checkpoint DeltaNet bound tensor geometry mismatch");
    }

    set_diagnostic(
        diagnostic,
        "checkpoint-bound Qwen3-Next Gated DeltaNet created");
    return QwenCheckpointGatedDeltaNet(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("checkpoint DeltaNet creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "checkpoint DeltaNet creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointGatedDeltaNet::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->config.hidden_size != 0U &&
         !impl_->qkvz.empty() &&
         !impl_->ba.empty() &&
         !impl_->conv.empty() &&
         !impl_->dt_bias.empty() &&
         !impl_->a_log.empty() &&
         !impl_->norm.empty() &&
         !impl_->out_proj.empty();
}

std::size_t QwenCheckpointGatedDeltaNet::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

const QwenGatedDeltaNetConfig&
QwenCheckpointGatedDeltaNet::config() const noexcept {
  static const QwenGatedDeltaNetConfig empty{};
  return impl_ != nullptr ? impl_->config : empty;
}

const QwenGatedDeltaNetState&
QwenCheckpointGatedDeltaNet::state() const noexcept {
  static const QwenGatedDeltaNetState empty{};
  return impl_ != nullptr ? impl_->state : empty;
}

void QwenCheckpointGatedDeltaNet::reset_state() noexcept {
  if (impl_ != nullptr) {
    impl_->state = {};
  }
}

QwenGatedDeltaNetStepResult QwenCheckpointGatedDeltaNet::run(
    std::span<const float> normalized_hidden) noexcept {
  QwenGatedDeltaNetStepResult result;
  if (!valid()) {
    result.diagnostic = "checkpoint DeltaNet runtime is not valid";
    return result;
  }

  const QwenGatedDeltaNetWeights weights{
      .in_proj_qkvz = impl_->qkvz,
      .in_proj_ba = impl_->ba,
      .conv = impl_->conv,
      .dt_bias = impl_->dt_bias,
      .a_log = impl_->a_log,
      .norm = impl_->norm,
      .out_proj = impl_->out_proj,
  };

  return run_qwen_gated_deltanet_cpu_step(
      normalized_hidden,
      weights,
      impl_->config,
      impl_->state);
}

}  // namespace orbi::streammoe
