#include "orbi/streammoe/model/checkpoint_gqa.hpp"

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

struct QwenCheckpointGqa::Impl {
  std::size_t layer_index{};
  QwenGqaConfig config{};

  std::vector<float> q_proj;
  std::vector<float> k_proj;
  std::vector<float> v_proj;
  std::vector<float> o_proj;
  std::vector<float> q_norm;
  std::vector<float> k_norm;

  QwenGqaState state;
};

QwenCheckpointGqa::QwenCheckpointGqa() = default;
QwenCheckpointGqa::~QwenCheckpointGqa() = default;
QwenCheckpointGqa::QwenCheckpointGqa(QwenCheckpointGqa&&) noexcept = default;
QwenCheckpointGqa& QwenCheckpointGqa::operator=(
    QwenCheckpointGqa&&) noexcept = default;

QwenCheckpointGqa::QwenCheckpointGqa(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointGqa>
QwenCheckpointGqa::create(
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  try {
    if (binding.is_linear || !binding.attention.has_value()) {
      throw std::runtime_error(
          "checkpoint GQA requires a full-attention Qwen3-Next layer binding");
    }
    if (binding.layer_index >= config.num_hidden_layers ||
        config.is_linear_layer(binding.layer_index)) {
      throw std::runtime_error(
          "checkpoint GQA layer/config classification mismatch");
    }

    const auto& attention = *binding.attention;

    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->config = {
        .hidden_size = config.hidden_size,
        .num_attention_heads = config.num_attention_heads,
        .num_key_value_heads = config.num_key_value_heads,
        .head_dim = config.head_dim,
        .partial_rotary_factor = config.partial_rotary_factor,
        .rope_theta = config.rope_theta,
        .rms_eps = config.rms_norm_eps,
        .max_position_embeddings = config.max_position_embeddings,
    };

    impl->q_proj =
        dequantize_mlx_affine_module_cpu(attention.q_proj);
    impl->k_proj =
        dequantize_mlx_affine_module_cpu(attention.k_proj);
    impl->v_proj =
        dequantize_mlx_affine_module_cpu(attention.v_proj);
    impl->o_proj =
        dequantize_mlx_affine_module_cpu(attention.o_proj);
    impl->q_norm = attention.q_norm;
    impl->k_norm = attention.k_norm;

    const auto q_dim =
        impl->config.num_attention_heads * impl->config.head_dim;
    const auto kv_dim =
        impl->config.num_key_value_heads * impl->config.head_dim;

    if (impl->q_proj.size() !=
            2U * q_dim * impl->config.hidden_size ||
        impl->k_proj.size() !=
            kv_dim * impl->config.hidden_size ||
        impl->v_proj.size() !=
            kv_dim * impl->config.hidden_size ||
        impl->o_proj.size() !=
            impl->config.hidden_size * q_dim ||
        impl->q_norm.size() != impl->config.head_dim ||
        impl->k_norm.size() != impl->config.head_dim) {
      throw std::runtime_error(
          "checkpoint GQA bound tensor geometry mismatch");
    }

    set_diagnostic(
        diagnostic,
        "checkpoint-bound Qwen3-Next gated GQA created");
    return QwenCheckpointGqa(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("checkpoint GQA creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "checkpoint GQA creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointGqa::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->config.hidden_size != 0U &&
         !impl_->q_proj.empty() &&
         !impl_->k_proj.empty() &&
         !impl_->v_proj.empty() &&
         !impl_->o_proj.empty() &&
         !impl_->q_norm.empty() &&
         !impl_->k_norm.empty();
}

std::size_t QwenCheckpointGqa::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

const QwenGqaConfig& QwenCheckpointGqa::config() const noexcept {
  static const QwenGqaConfig empty{};
  return impl_ != nullptr ? impl_->config : empty;
}

const QwenGqaState& QwenCheckpointGqa::state() const noexcept {
  static const QwenGqaState empty{};
  return impl_ != nullptr ? impl_->state : empty;
}

void QwenCheckpointGqa::reset_state() noexcept {
  if (impl_ != nullptr) {
    impl_->state = {};
  }
}

QwenGqaStepResult QwenCheckpointGqa::run(
    std::span<const float> normalized_hidden) noexcept {
  QwenGqaStepResult result;
  if (!valid()) {
    result.diagnostic = "checkpoint GQA runtime is not valid";
    return result;
  }

  const QwenGqaWeights weights{
      .q_proj = impl_->q_proj,
      .k_proj = impl_->k_proj,
      .v_proj = impl_->v_proj,
      .o_proj = impl_->o_proj,
      .q_norm = impl_->q_norm,
      .k_norm = impl_->k_norm,
  };

  return run_qwen_gqa_cpu_step(
      normalized_hidden,
      weights,
      impl_->config,
      impl_->state);
}

}  // namespace orbi::streammoe
