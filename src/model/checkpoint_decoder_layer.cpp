#include "orbi/streammoe/model/checkpoint_decoder_layer.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

using LayerVariant = std::variant<
    std::monostate,
    QwenCheckpointLinearDecoderLayer,
    QwenCheckpointFullAttentionDecoderLayer>;

}  // namespace

std::optional<QwenCheckpointDecoderLayerKind>
qwen_checkpoint_decoder_layer_kind(
    const Qwen3NextDenseConfig& config,
    std::size_t layer_index) noexcept {
  if (config.full_attention_interval == 0U ||
      layer_index >= config.num_hidden_layers) {
    return std::nullopt;
  }

  return config.is_linear_layer(layer_index)
      ? QwenCheckpointDecoderLayerKind::gated_deltanet
      : QwenCheckpointDecoderLayerKind::gated_gqa;
}

struct QwenCheckpointDecoderLayer::Impl {
  QwenCheckpointDecoderLayerKind kind{
      QwenCheckpointDecoderLayerKind::gated_deltanet};
  std::size_t layer_index{};
  std::size_t hidden_size{};
  LayerVariant layer;
};

QwenCheckpointDecoderLayer::QwenCheckpointDecoderLayer() = default;
QwenCheckpointDecoderLayer::~QwenCheckpointDecoderLayer() = default;
QwenCheckpointDecoderLayer::QwenCheckpointDecoderLayer(
    QwenCheckpointDecoderLayer&&) noexcept = default;
QwenCheckpointDecoderLayer& QwenCheckpointDecoderLayer::operator=(
    QwenCheckpointDecoderLayer&&) noexcept = default;

QwenCheckpointDecoderLayer::QwenCheckpointDecoderLayer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointDecoderLayer>
QwenCheckpointDecoderLayer::create(
    VulkanComputeContext& context,
    const QwenDenseLayerBinding& binding,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "unified decoder layer requires a valid Vulkan context");
    return std::nullopt;
  }

  const auto expected =
      qwen_checkpoint_decoder_layer_kind(config, binding.layer_index);
  if (!expected.has_value()) {
    set_diagnostic(
        diagnostic,
        "unified decoder layer index/config classification is invalid");
    return std::nullopt;
  }

  const bool binding_is_linear = binding.is_linear;
  const bool expected_is_linear =
      *expected == QwenCheckpointDecoderLayerKind::gated_deltanet;
  if (binding_is_linear != expected_is_linear) {
    set_diagnostic(
        diagnostic,
        "unified decoder layer binding disagrees with config topology");
    return std::nullopt;
  }

  try {
    auto impl = std::make_unique<Impl>();
    impl->kind = *expected;
    impl->layer_index = binding.layer_index;
    impl->hidden_size = config.hidden_size;

    std::string local;
    if (*expected == QwenCheckpointDecoderLayerKind::gated_deltanet) {
      auto layer = QwenCheckpointLinearDecoderLayer::create(
          context,
          binding,
          config,
          config.rms_norm_eps,
          &local);
      if (!layer.has_value()) {
        set_diagnostic(
            diagnostic,
            "unified DeltaNet decoder creation failed: " + local);
        return std::nullopt;
      }
      impl->layer = std::move(*layer);
    } else {
      auto layer = QwenCheckpointFullAttentionDecoderLayer::create(
          context,
          binding,
          config,
          &local);
      if (!layer.has_value()) {
        set_diagnostic(
            diagnostic,
            "unified GQA decoder creation failed: " + local);
        return std::nullopt;
      }
      impl->layer = std::move(*layer);
    }

    set_diagnostic(
        diagnostic,
        *expected == QwenCheckpointDecoderLayerKind::gated_deltanet
            ? "unified decoder selected Gated DeltaNet layer"
            : "unified decoder selected gated GQA layer");
    return QwenCheckpointDecoderLayer(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("unified decoder layer creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "unified decoder layer creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointDecoderLayer::valid() const noexcept {
  if (impl_ == nullptr || impl_->hidden_size == 0U) return false;

  if (const auto* linear =
          std::get_if<QwenCheckpointLinearDecoderLayer>(&impl_->layer)) {
    return impl_->kind ==
               QwenCheckpointDecoderLayerKind::gated_deltanet &&
           linear->valid() &&
           linear->layer_index() == impl_->layer_index;
  }

  if (const auto* attention =
          std::get_if<QwenCheckpointFullAttentionDecoderLayer>(&impl_->layer)) {
    return impl_->kind ==
               QwenCheckpointDecoderLayerKind::gated_gqa &&
           attention->valid() &&
           attention->layer_index() == impl_->layer_index;
  }

  return false;
}

QwenCheckpointDecoderLayerKind
QwenCheckpointDecoderLayer::kind() const noexcept {
  return impl_ != nullptr
      ? impl_->kind
      : QwenCheckpointDecoderLayerKind::gated_deltanet;
}

std::size_t QwenCheckpointDecoderLayer::layer_index() const noexcept {
  return impl_ != nullptr ? impl_->layer_index : 0U;
}

std::size_t QwenCheckpointDecoderLayer::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->hidden_size : 0U;
}

const QwenGatedDeltaNetState*
QwenCheckpointDecoderLayer::delta_state() const noexcept {
  if (impl_ == nullptr) return nullptr;
  if (const auto* linear =
          std::get_if<QwenCheckpointLinearDecoderLayer>(&impl_->layer)) {
    return &linear->delta_state();
  }
  return nullptr;
}

const QwenGqaState*
QwenCheckpointDecoderLayer::gqa_state() const noexcept {
  if (impl_ == nullptr) return nullptr;
  if (const auto* attention =
          std::get_if<QwenCheckpointFullAttentionDecoderLayer>(&impl_->layer)) {
    return &attention->gqa_state();
  }
  return nullptr;
}

void QwenCheckpointDecoderLayer::reset_state() noexcept {
  if (impl_ == nullptr) return;

  if (auto* linear =
          std::get_if<QwenCheckpointLinearDecoderLayer>(&impl_->layer)) {
    linear->reset_state();
    return;
  }

  if (auto* attention =
          std::get_if<QwenCheckpointFullAttentionDecoderLayer>(&impl_->layer)) {
    attention->reset_state();
  }
}

QwenCheckpointDecoderLayerResult
QwenCheckpointDecoderLayer::run(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::span<const float> hidden) noexcept {
  QwenCheckpointDecoderLayerResult result;

  if (!valid()) {
    result.diagnostic = "unified decoder layer is not valid";
    return result;
  }
  result.kind = impl_->kind;

  if (auto* linear =
          std::get_if<QwenCheckpointLinearDecoderLayer>(&impl_->layer)) {
    const auto branch = linear->run(context, expert_cache, hidden);
    result.executed = branch.executed;
    result.values = branch.values;
    result.diagnostic = branch.diagnostic;
    return result;
  }

  if (auto* attention =
          std::get_if<QwenCheckpointFullAttentionDecoderLayer>(&impl_->layer)) {
    const auto branch = attention->run(context, expert_cache, hidden);
    result.executed = branch.executed;
    result.values = branch.values;
    result.diagnostic = branch.diagnostic;
    return result;
  }

  result.diagnostic = "unified decoder layer has no concrete implementation";
  return result;
}

}  // namespace orbi::streammoe
