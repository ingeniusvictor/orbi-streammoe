#include "orbi/streammoe/session/greedy_token_session.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <utility>

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

bool contains_token(
    std::span<const std::size_t> tokens,
    std::size_t token) noexcept {
  return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

}  // namespace

struct QwenGreedyTokenSession::Impl {
  QwenCheckpointModelShell model;
};

QwenGreedyTokenSession::QwenGreedyTokenSession() = default;
QwenGreedyTokenSession::~QwenGreedyTokenSession() = default;
QwenGreedyTokenSession::QwenGreedyTokenSession(
    QwenGreedyTokenSession&&) noexcept = default;
QwenGreedyTokenSession& QwenGreedyTokenSession::operator=(
    QwenGreedyTokenSession&&) noexcept = default;

QwenGreedyTokenSession::QwenGreedyTokenSession(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenGreedyTokenSession>
QwenGreedyTokenSession::create(
    VulkanComputeContext& context,
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  try {
    std::string local;
    auto model = QwenCheckpointModelShell::create(
        context,
        checkpoint,
        config,
        &local);
    if (!model.has_value()) {
      set_diagnostic(
          diagnostic,
          "greedy token session model creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->model = std::move(*model);
    set_diagnostic(
        diagnostic,
        "Qwen greedy token-ID session created");
    return QwenGreedyTokenSession(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("greedy token session creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "greedy token session creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenGreedyTokenSession::valid() const noexcept {
  return impl_ != nullptr && impl_->model.valid();
}

std::size_t QwenGreedyTokenSession::vocab_size() const noexcept {
  return impl_ != nullptr ? impl_->model.vocab_size() : 0U;
}

const QwenCheckpointModelShell*
QwenGreedyTokenSession::model_shell() const noexcept {
  return impl_ != nullptr ? &impl_->model : nullptr;
}

void QwenGreedyTokenSession::reset() noexcept {
  if (impl_ != nullptr) {
    impl_->model.reset_state();
  }
}

QwenGreedySessionResult
QwenGreedyTokenSession::generate(
    const QpackMlxCheckpoint& checkpoint,
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::span<const std::size_t> prompt_tokens,
    const QwenGreedySessionOptions& options) noexcept {
  QwenGreedySessionResult result;

  if (!valid()) {
    result.stop_reason = QwenGreedySessionStopReason::model_error;
    result.diagnostic = "greedy token session is not valid";
    return result;
  }
  if (prompt_tokens.empty()) {
    result.stop_reason = QwenGreedySessionStopReason::invalid_request;
    result.diagnostic = "greedy token session requires at least one prompt token";
    return result;
  }
  if (options.max_new_tokens == 0U) {
    result.stop_reason = QwenGreedySessionStopReason::invalid_request;
    result.diagnostic = "greedy token session max_new_tokens must be non-zero";
    return result;
  }
  if (options.lm_head_chunk_rows == 0U) {
    result.stop_reason = QwenGreedySessionStopReason::invalid_request;
    result.diagnostic = "greedy token session LM-head chunk size must be non-zero";
    return result;
  }

  const auto vocab = vocab_size();
  for (const auto token : prompt_tokens) {
    if (token >= vocab) {
      result.stop_reason = QwenGreedySessionStopReason::invalid_request;
      result.diagnostic = "greedy token session prompt token exceeds vocabulary";
      return result;
    }
  }
  for (const auto token : options.stop_token_ids) {
    if (token >= vocab) {
      result.stop_reason = QwenGreedySessionStopReason::invalid_request;
      result.diagnostic = "greedy token session stop token exceeds vocabulary";
      return result;
    }
  }

  try {
    if (options.reset_before_prompt) {
      impl_->model.reset_state();
    }

    result.prompt_tokens.assign(prompt_tokens.begin(), prompt_tokens.end());

    QwenCheckpointModelStepResult step;
    for (const auto token : prompt_tokens) {
      step = impl_->model.step_greedy(
          checkpoint,
          context,
          expert_cache,
          token,
          options.lm_head_chunk_rows);
      ++result.model_steps;

      if (!step.executed) {
        result.stop_reason = QwenGreedySessionStopReason::model_error;
        result.diagnostic =
            "greedy token session prompt step failed: " + step.diagnostic;
        return result;
      }
    }

    auto append_prediction =
        [&](const QwenCheckpointModelStepResult& prediction) {
          result.generated_tokens.push_back(prediction.greedy.token_id);
          result.generated_logits.push_back(prediction.greedy.logit);
        };

    append_prediction(step);

    if (contains_token(
            options.stop_token_ids,
            result.generated_tokens.back())) {
      result.executed = true;
      result.stop_reason = QwenGreedySessionStopReason::stop_token;
      result.diagnostic =
          "greedy token session stopped on configured stop token";
      return result;
    }

    while (result.generated_tokens.size() < options.max_new_tokens) {
      step = impl_->model.step_greedy(
          checkpoint,
          context,
          expert_cache,
          result.generated_tokens.back(),
          options.lm_head_chunk_rows);
      ++result.model_steps;

      if (!step.executed) {
        result.stop_reason = QwenGreedySessionStopReason::model_error;
        result.diagnostic =
            "greedy token session decode step failed: " + step.diagnostic;
        return result;
      }

      append_prediction(step);
      if (contains_token(
              options.stop_token_ids,
              result.generated_tokens.back())) {
        result.executed = true;
        result.stop_reason = QwenGreedySessionStopReason::stop_token;
        result.diagnostic =
            "greedy token session stopped on configured stop token";
        return result;
      }
    }

    result.executed = true;
    result.stop_reason = QwenGreedySessionStopReason::max_new_tokens;
    result.diagnostic =
        "greedy token session reached max_new_tokens";
    return result;
  } catch (const std::exception& e) {
    result.stop_reason = QwenGreedySessionStopReason::model_error;
    result.diagnostic =
        std::string("greedy token session failed: ") + e.what();
    return result;
  } catch (...) {
    result.stop_reason = QwenGreedySessionStopReason::model_error;
    result.diagnostic =
        "greedy token session encountered an unknown exception";
    return result;
  }
}

}  // namespace orbi::streammoe
