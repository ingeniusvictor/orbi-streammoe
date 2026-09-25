#include "orbi/streammoe/model/checkpoint_model_shell.hpp"

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

struct QwenCheckpointModelShell::Impl {
  Qwen3NextDenseConfig config;
  QwenGlobalCheckpointBinding global;
  QwenCheckpointDecoderStack stack;
};

QwenCheckpointModelShell::QwenCheckpointModelShell() = default;
QwenCheckpointModelShell::~QwenCheckpointModelShell() = default;
QwenCheckpointModelShell::QwenCheckpointModelShell(
    QwenCheckpointModelShell&&) noexcept = default;
QwenCheckpointModelShell& QwenCheckpointModelShell::operator=(
    QwenCheckpointModelShell&&) noexcept = default;

QwenCheckpointModelShell::QwenCheckpointModelShell(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenCheckpointModelShell>
QwenCheckpointModelShell::create(
    VulkanComputeContext& context,
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(
        diagnostic,
        "checkpoint model shell requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    auto global =
        bind_qwen3_next_global_checkpoint(checkpoint, config);

    std::string local;
    auto stack = QwenCheckpointDecoderStack::create(
        context,
        checkpoint,
        config,
        &local);
    if (!stack.has_value()) {
      set_diagnostic(
          diagnostic,
          "checkpoint model shell decoder creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->config = config;
    impl->global = std::move(global);
    impl->stack = std::move(*stack);

    set_diagnostic(
        diagnostic,
        "checkpoint-bound Qwen3-Next model shell created");
    return QwenCheckpointModelShell(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("checkpoint model shell creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "checkpoint model shell creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenCheckpointModelShell::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->global.hidden_size != 0U &&
         impl_->global.vocab_size != 0U &&
         impl_->global.final_norm.size() == impl_->global.hidden_size &&
         impl_->stack.valid() &&
         impl_->stack.hidden_size() == impl_->global.hidden_size;
}

std::size_t QwenCheckpointModelShell::hidden_size() const noexcept {
  return impl_ != nullptr ? impl_->global.hidden_size : 0U;
}

std::size_t QwenCheckpointModelShell::vocab_size() const noexcept {
  return impl_ != nullptr ? impl_->global.vocab_size : 0U;
}

std::size_t QwenCheckpointModelShell::layer_count() const noexcept {
  return impl_ != nullptr ? impl_->stack.layer_count() : 0U;
}

const QwenGlobalCheckpointBinding*
QwenCheckpointModelShell::global_binding() const noexcept {
  return impl_ != nullptr ? &impl_->global : nullptr;
}

const QwenCheckpointDecoderStack*
QwenCheckpointModelShell::decoder_stack() const noexcept {
  return impl_ != nullptr ? &impl_->stack : nullptr;
}

void QwenCheckpointModelShell::reset_state() noexcept {
  if (impl_ != nullptr) {
    impl_->stack.reset_state();
  }
}

QwenCheckpointModelStepResult
QwenCheckpointModelShell::step_greedy(
    const QpackMlxCheckpoint& checkpoint,
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::size_t token_id,
    std::size_t lm_head_chunk_rows) noexcept {
  QwenCheckpointModelStepResult result;
  result.input_token = token_id;

  if (!valid()) {
    result.diagnostic = "checkpoint model shell is not valid";
    return result;
  }
  if (!context.valid()) {
    result.diagnostic =
        "checkpoint model shell requires a valid Vulkan context";
    return result;
  }
  if (lm_head_chunk_rows == 0U) {
    result.diagnostic =
        "checkpoint model shell LM-head chunk size must be non-zero";
    return result;
  }

  try {
    result.embedding = read_qwen_embedding_row(
        checkpoint,
        impl_->global,
        token_id);

    const auto decoded = impl_->stack.run(
        context,
        expert_cache,
        result.embedding);
    if (!decoded.executed) {
      result.diagnostic =
          "checkpoint model shell decoder failed: " +
          decoded.diagnostic;
      return result;
    }
    result.decoder_hidden = decoded.values;

    const auto normalized = run_vulkan_rms_norm(
        context,
        result.decoder_hidden,
        1U,
        impl_->global.hidden_size,
        impl_->global.final_norm,
        impl_->config.rms_norm_eps);
    if (!normalized.executed) {
      result.diagnostic =
          "checkpoint model shell final RMSNorm failed: " +
          normalized.diagnostic;
      return result;
    }
    result.final_hidden = normalized.values;

    result.greedy = greedy_qwen_lm_head_streaming(
        checkpoint,
        impl_->global,
        result.final_hidden,
        lm_head_chunk_rows);

    result.executed = true;
    result.diagnostic =
        "checkpoint Qwen3-Next token-to-next-token step executed";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("checkpoint model shell step failed: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "checkpoint model shell step encountered an unknown exception";
    return result;
  }
}

}  // namespace orbi::streammoe
