#include "orbi/streammoe/model/checkpoint_decoder_stack.hpp"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

}  // namespace

struct QwenCheckpointDecoderStack::Impl {
  std::size_t hidden_size{};
  std::size_t delta_layers{};
  std::size_t gqa_layers{};
  std::vector<QwenCheckpointDecoderLayer> layers;
};

QwenCheckpointDecoderStack::QwenCheckpointDecoderStack() = default;
QwenCheckpointDecoderStack::~QwenCheckpointDecoderStack() = default;
QwenCheckpointDecoderStack::QwenCheckpointDecoderStack(
    QwenCheckpointDecoderStack&&) noexcept = default;
QwenCheckpointDecoderStack& QwenCheckpointDecoderStack::operator=(
    QwenCheckpointDecoderStack&&) noexcept = default;

QwenCheckpointDecoderStack::QwenCheckpointDecoderStack(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointDecoderStack>
QwenCheckpointDecoderStack::create(
    VulkanComputeContext& context,
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "decoder stack requires a valid Vulkan context");
    return std::nullopt;
  }

  if (config.hidden_size == 0U ||
      config.num_hidden_layers == 0U ||
      config.full_attention_interval == 0U) {
    set_diagnostic(
        diagnostic,
        "decoder stack requires non-zero model geometry");
    return std::nullopt;
  }

  try {
    auto impl = std::make_unique<Impl>();
    impl->hidden_size = config.hidden_size;
    impl->layers.reserve(config.num_hidden_layers);

    for (std::size_t layer_index = 0;
         layer_index < config.num_hidden_layers;
         ++layer_index) {
      const auto binding =
          bind_qwen3_next_dense_layer(
              checkpoint,
              config,
              layer_index);

      std::string local;
      auto layer = QwenCheckpointDecoderLayer::create(
          context,
          binding,
          config,
          &local);
      if (!layer.has_value()) {
        set_diagnostic(
            diagnostic,
            "decoder stack layer " +
                std::to_string(layer_index) +
                " creation failed: " + local);
        return std::nullopt;
      }

      if (layer->kind() ==
          QwenCheckpointDecoderLayerKind::gated_deltanet) {
        ++impl->delta_layers;
      } else {
        ++impl->gqa_layers;
      }

      impl->layers.push_back(std::move(*layer));
    }

    set_diagnostic(
        diagnostic,
        "checkpoint-bound Qwen3-Next decoder stack created");
    return QwenCheckpointDecoderStack(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("decoder stack creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "decoder stack creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointDecoderStack::valid() const noexcept {
  if (impl_ == nullptr ||
      impl_->hidden_size == 0U ||
      impl_->layers.empty()) {
    return false;
  }

  for (std::size_t i = 0; i < impl_->layers.size(); ++i) {
    const auto& layer = impl_->layers[i];
    if (!layer.valid() ||
        layer.layer_index() != i ||
        layer.hidden_size() != impl_->hidden_size) {
      return false;
    }
  }

  return impl_->delta_layers + impl_->gqa_layers ==
         impl_->layers.size();
}

std::size_t QwenCheckpointDecoderStack::layer_count() const noexcept {
  return impl_ != nullptr ? impl_->layers.size() : 0U;
}

std::size_t QwenCheckpointDecoderStack::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->hidden_size : 0U;
}

std::size_t
QwenCheckpointDecoderStack::delta_layer_count() const noexcept {
  return impl_ != nullptr ? impl_->delta_layers : 0U;
}

std::size_t
QwenCheckpointDecoderStack::gqa_layer_count() const noexcept {
  return impl_ != nullptr ? impl_->gqa_layers : 0U;
}

const QwenCheckpointDecoderLayer*
QwenCheckpointDecoderStack::layer(
    std::size_t index) const noexcept {
  if (impl_ == nullptr || index >= impl_->layers.size()) {
    return nullptr;
  }
  return &impl_->layers[index];
}

void QwenCheckpointDecoderStack::reset_state() noexcept {
  if (impl_ == nullptr) return;
  for (auto& layer : impl_->layers) {
    layer.reset_state();
  }
}

QwenCheckpointDecoderStackResult
QwenCheckpointDecoderStack::run(
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::span<const float> hidden) noexcept {
  QwenCheckpointDecoderStackResult result;

  if (!valid()) {
    result.diagnostic = "decoder stack is not valid";
    return result;
  }
  if (!context.valid()) {
    result.diagnostic = "decoder stack requires a valid Vulkan context";
    return result;
  }
  if (hidden.size() != impl_->hidden_size) {
    result.diagnostic = "decoder stack hidden shape mismatch";
    return result;
  }

  result.values.assign(hidden.begin(), hidden.end());

  for (std::size_t i = 0; i < impl_->layers.size(); ++i) {
    const auto step = impl_->layers[i].run(
        context,
        expert_cache,
        result.values);
    if (!step.executed) {
      result.diagnostic =
          "decoder stack layer " + std::to_string(i) +
          " failed: " + step.diagnostic;
      return result;
    }

    result.values = step.values;
    result.layers_executed = i + 1U;
  }

  result.executed = true;
  result.diagnostic =
      "checkpoint-bound Qwen3-Next decoder stack executed";
  return result;
}

}  // namespace orbi::streammoe
