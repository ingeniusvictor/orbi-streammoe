#include "orbi/streammoe/session/greedy_text_session.hpp"

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

bool tokens_fit_vocab(
    std::span<const std::size_t> tokens,
    std::size_t vocab_size) noexcept {
  return std::all_of(
      tokens.begin(),
      tokens.end(),
      [&](std::size_t token) { return token < vocab_size; });
}

QwenGreedyTextSessionStatus map_token_failure(
    QwenGreedySessionStopReason reason) noexcept {
  return reason == QwenGreedySessionStopReason::model_error
      ? QwenGreedyTextSessionStatus::model_error
      : QwenGreedyTextSessionStatus::invalid_request;
}

}  // namespace

struct QwenGreedyTextSession::Impl {
  QwenGreedyTokenSession token_session;
};

QwenGreedyTextSession::QwenGreedyTextSession() = default;
QwenGreedyTextSession::~QwenGreedyTextSession() = default;
QwenGreedyTextSession::QwenGreedyTextSession(
    QwenGreedyTextSession&&) noexcept = default;
QwenGreedyTextSession& QwenGreedyTextSession::operator=(
    QwenGreedyTextSession&&) noexcept = default;

QwenGreedyTextSession::QwenGreedyTextSession(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenGreedyTextSession>
QwenGreedyTextSession::create(
    VulkanComputeContext& context,
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    std::string* diagnostic) noexcept {
  try {
    std::string local;
    auto token_session = QwenGreedyTokenSession::create(
        context,
        checkpoint,
        config,
        &local);
    if (!token_session.has_value()) {
      set_diagnostic(
          diagnostic,
          "greedy text session token runtime creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->token_session = std::move(*token_session);
    set_diagnostic(diagnostic, "Qwen greedy text session created");
    return QwenGreedyTextSession(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("greedy text session creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "greedy text session creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenGreedyTextSession::valid() const noexcept {
  return impl_ != nullptr && impl_->token_session.valid();
}

std::size_t QwenGreedyTextSession::vocab_size() const noexcept {
  return impl_ != nullptr ? impl_->token_session.vocab_size() : 0U;
}

const QwenGreedyTokenSession*
QwenGreedyTextSession::token_session() const noexcept {
  return impl_ != nullptr ? &impl_->token_session : nullptr;
}

void QwenGreedyTextSession::reset() noexcept {
  if (impl_ != nullptr) {
    impl_->token_session.reset();
  }
}

QwenGreedyTextSessionResult
QwenGreedyTextSession::generate(
    const QpackMlxCheckpoint& checkpoint,
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    const Tokenizer& tokenizer,
    std::string_view prompt,
    const QwenGreedyTextSessionOptions& options) noexcept {
  QwenGreedyTextSessionResult result;

  if (!valid()) {
    result.status = QwenGreedyTextSessionStatus::model_error;
    result.stop_reason = QwenGreedySessionStopReason::model_error;
    result.diagnostic = "greedy text session is not valid";
    return result;
  }

  try {
    const auto tokenizer_vocab = tokenizer.vocab_size();
    if (tokenizer_vocab.has_value() &&
        *tokenizer_vocab != vocab_size()) {
      result.status = QwenGreedyTextSessionStatus::invalid_request;
      result.diagnostic =
          "tokenizer vocabulary size does not match model vocabulary";
      return result;
    }

    const auto prepared = prepare_qwen_text_prompt(
        tokenizer,
        prompt,
        options.generation,
        options.tokenizer);
    if (!prepared.prepared) {
      result.status = QwenGreedyTextSessionStatus::tokenizer_error;
      result.diagnostic =
          "greedy text session prompt preparation failed: " +
          prepared.diagnostic;
      return result;
    }

    if (!tokens_fit_vocab(prepared.prompt_tokens, vocab_size()) ||
        !tokens_fit_vocab(
            prepared.session_options.stop_token_ids,
            vocab_size())) {
      result.status = QwenGreedyTextSessionStatus::invalid_request;
      result.diagnostic =
          "prepared tokenizer IDs exceed model vocabulary";
      return result;
    }

    result.prompt_tokens = prepared.prompt_tokens;

    const auto token_result = impl_->token_session.generate(
        checkpoint,
        context,
        expert_cache,
        prepared.prompt_tokens,
        prepared.session_options);

    result.token_session_executed = token_result.executed;
    result.stop_reason = token_result.stop_reason;
    result.model_steps = token_result.model_steps;
    result.prompt_tokens = token_result.prompt_tokens.empty()
        ? result.prompt_tokens
        : token_result.prompt_tokens;
    result.generated_tokens = token_result.generated_tokens;
    result.generated_logits = token_result.generated_logits;

    if (!token_result.executed) {
      result.status = map_token_failure(token_result.stop_reason);
      result.diagnostic =
          "greedy text session token generation failed: " +
          token_result.diagnostic;
      return result;
    }

    const auto decoded = decode_qwen_generated_text(
        tokenizer,
        token_result,
        options.tokenizer);
    if (!decoded.decoded) {
      result.status = QwenGreedyTextSessionStatus::tokenizer_error;
      result.diagnostic =
          "greedy text session decode failed after token generation: " +
          decoded.diagnostic;
      return result;
    }

    result.completed = true;
    result.status = QwenGreedyTextSessionStatus::completed;
    result.text = decoded.text;
    result.diagnostic = "greedy text session completed";
    return result;
  } catch (const std::exception& e) {
    result.status = QwenGreedyTextSessionStatus::model_error;
    result.diagnostic =
        std::string("greedy text session failed: ") + e.what();
    return result;
  } catch (...) {
    result.status = QwenGreedyTextSessionStatus::model_error;
    result.diagnostic =
        "greedy text session encountered an unknown exception";
    return result;
  }
}

}  // namespace orbi::streammoe
