#include "orbi/streammoe/model/checkpoint_full_attention_decoder_layer.hpp"

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

struct QwenCheckpointFullAttentionDecoderLayer::Impl {
  std::size_t layer_index{};
  std::size_t hidden_size{};
  QwenCheckpointGqaSublayer gqa;
  QwenCheckpointMoeSublayer moe;
};

QwenCheckpointFullAttentionDecoderLayer::
QwenCheckpointFullAttentionDecoderLayer() = default;
QwenCheckpointFullAttentionDecoderLayer::
~QwenCheckpointFullAttentionDecoderLayer() = default;
QwenCheckpointFullAttentionDecoderLayer::
QwenCheckpointFullAttentionDecoderLayer(
    QwenCheckpointFullAttentionDecoderLayer&&) noexcept = default;
QwenCheckpointFullAttentionDecoderLayer&
QwenCheckpointFullAttentionDecoderLayer::operator=(
    QwenCheckpointFullAttentionDecoderLayer&&) noexcept = default;

QwenCheckpointFullAttentionDecoderLayer::
QwenCheckpointFullAttentionDecoderLayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointFullAttentionDecoderLayer>
QwenCheckpointFullAttentionDecoderLayer::create(
    VulkanComputeContext& context,
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "full-attention decoder layer requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    if (binding.is_linear || !binding.attention.has_value()) {
      throw std::runtime_error(
          "full-attention decoder layer requires a GQA binding");
    }
    if (binding.layer_index >= config.num_hidden_layers ||
        config.is_linear_layer(binding.layer_index)) {
      throw std::runtime_error(
          "full-attention decoder layer layer/config classification mismatch");
    }
    if (config.hidden_size == 0U ||
        binding.input_norm.size() != config.hidden_size ||
        binding.post_attention_norm.size() != config.hidden_size) {
      throw std::runtime_error(
          "full-attention decoder layer RMSNorm geometry mismatch");
    }

    std::string local;
    auto gqa = QwenCheckpointGqaSublayer::create(
        context,
        binding,
        config,
        &local);
    if (!gqa.has_value()) {
      set_diagnostic(
          diagnostic,
          "full-attention decoder GQA sublayer creation failed: " + local);
      return std::nullopt;
    }

    local.clear();
    auto moe = QwenCheckpointMoeSublayer::create(
        context,
        binding,
        config,
        config.rms_norm_eps,
        &local);
    if (!moe.has_value()) {
      set_diagnostic(
          diagnostic,
          "full-attention decoder MoE sublayer creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->layer_index = binding.layer_index;
    impl->hidden_size = config.hidden_size;
    impl->gqa = std::move(*gqa);
    impl->moe = std::move(*moe);

    set_diagnostic(
        diagnostic,
        "checkpoint-bound complete full-attention Qwen decoder layer created");
    return QwenCheckpointFullAttentionDecoderLayer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("full-attention decoder layer creation failed: ") +
            e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "full-attention decoder layer creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointFullAttentionDecoderLayer::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->hidden_size != 0U &&
         impl_->gqa.valid() &&
         impl_->moe.valid() &&
         impl_->gqa.layer_index() == impl_->layer_index &&
         impl_->moe.layer_index() == impl_->layer_index;
}

std::size_t
QwenCheckpointFullAttentionDecoderLayer::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

std::size_t
QwenCheckpointFullAttentionDecoderLayer::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->hidden_size : 0U;
}

const QwenGqaState&
QwenCheckpointFullAttentionDecoderLayer::gqa_state() const noexcept {
  static const QwenGqaState empty{};
  return impl_ != nullptr ? impl_->gqa.state() : empty;
}

void QwenCheckpointFullAttentionDecoderLayer::reset_state() noexcept {
  if (impl_ != nullptr) {
    impl_->gqa.reset_state();
  }
}

QwenCheckpointFullAttentionDecoderLayerResult
QwenCheckpointFullAttentionDecoderLayer::run(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::span<const float> hidden) noexcept {
  QwenCheckpointFullAttentionDecoderLayerResult result;

  if (!valid()) {
    result.diagnostic = "full-attention decoder layer is not valid";
    return result;
  }
  if (!context.valid()) {
    result.diagnostic =
        "full-attention decoder layer requires a valid Vulkan context";
    return result;
  }
  if (hidden.size() != impl_->hidden_size) {
    result.diagnostic =
        "full-attention decoder layer hidden shape mismatch";
    return result;
  }

  result.attention_branch = impl_->gqa.run(context, hidden);
  if (!result.attention_branch.executed) {
    result.diagnostic =
        "full-attention decoder GQA sublayer failed: " +
        result.attention_branch.diagnostic;
    return result;
  }

  result.moe_branch = impl_->moe.run(
      context,
      expert_cache,
      result.attention_branch.values);
  if (!result.moe_branch.executed) {
    result.diagnostic =
        "full-attention decoder MoE sublayer failed: " +
        result.moe_branch.diagnostic;
    return result;
  }

  result.values = result.moe_branch.values;
  result.executed = true;
  result.diagnostic =
      "complete checkpoint-bound full-attention Qwen decoder layer executed";
  return result;
}

}  // namespace orbi::streammoe
