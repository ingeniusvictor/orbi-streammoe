#include "orbi/streammoe/session/text_completion_session.hpp"

#include <algorithm>
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

std::vector<std::size_t> merged_stop_tokens(
    std::span<const std::size_t> eos,
    std::span<const std::size_t> extra,
    std::size_t vocab_size) {
  std::vector<std::size_t> out;
  out.reserve(eos.size() + extra.size());
  out.insert(out.end(), eos.begin(), eos.end());
  out.insert(out.end(), extra.begin(), extra.end());

  for (const auto token : out) {
    if (token >= vocab_size) {
      throw std::runtime_error(
          "text completion: stop token exceeds vocabulary");
    }
  }

  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

bool contains_token(
    std::span<const std::size_t> ids,
    std::size_t token) noexcept {
  return std::find(ids.begin(), ids.end(), token) != ids.end();
}

}  // namespace

struct QwenTextCompletionSession::Impl {
  Qwen3NextDenseConfig config;
  QwenGenerationMetadata generation;
  std::unique_ptr<TextTokenizer> tokenizer;
  QwenGreedyTokenSession token_session;
};

QwenTextCompletionSession::QwenTextCompletionSession() = default;
QwenTextCompletionSession::~QwenTextCompletionSession() = default;
QwenTextCompletionSession::QwenTextCompletionSession(
    QwenTextCompletionSession&&) noexcept = default;
QwenTextCompletionSession& QwenTextCompletionSession::operator=(
    QwenTextCompletionSession&&) noexcept = default;

QwenTextCompletionSession::QwenTextCompletionSession(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenTextCompletionSession>
QwenTextCompletionSession::create(
    VulkanComputeContext& context,
    const QpackReader& qpack,
    const QpackMlxCheckpoint& checkpoint,
    const Qwen3NextDenseConfig& config,
    std::unique_ptr<TextTokenizer> tokenizer,
    std::string* diagnostic) noexcept {
  if (tokenizer == nullptr) {
    set_diagnostic(diagnostic, "text completion requires a tokenizer");
    return std::nullopt;
  }
  if (config.vocab_size == 0U ||
      config.max_position_embeddings == 0U) {
    set_diagnostic(
        diagnostic,
        "text completion requires non-zero vocab/context capacity");
    return std::nullopt;
  }
  if (tokenizer->vocab_size() != config.vocab_size) {
    set_diagnostic(
        diagnostic,
        "text completion tokenizer/model vocabulary mismatch");
    return std::nullopt;
  }

  try {
    auto generation = load_qwen_generation_metadata(qpack);
    for (const auto token : generation.eos_token_ids) {
      if (token >= config.vocab_size) {
        throw std::runtime_error(
            "checkpoint EOS token exceeds model vocabulary");
      }
    }

    std::string local;
    auto token_session = QwenGreedyTokenSession::create(
        context,
        checkpoint,
        config,
        &local);
    if (!token_session.has_value()) {
      set_diagnostic(
          diagnostic,
          "text completion token session creation failed: " + local);
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->config = config;
    impl->generation = std::move(generation);
    impl->tokenizer = std::move(tokenizer);
    impl->token_session = std::move(*token_session);

    set_diagnostic(
        diagnostic,
        "Qwen text completion boundary created");
    return QwenTextCompletionSession(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("text completion creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "text completion creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenTextCompletionSession::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->tokenizer != nullptr &&
         impl_->token_session.valid() &&
         impl_->tokenizer->vocab_size() == impl_->config.vocab_size &&
         impl_->config.max_position_embeddings != 0U;
}

std::size_t QwenTextCompletionSession::vocab_size() const noexcept {
  return impl_ != nullptr ? impl_->config.vocab_size : 0U;
}

std::size_t QwenTextCompletionSession::context_capacity() const noexcept {
  return impl_ != nullptr ? impl_->config.max_position_embeddings : 0U;
}

const QwenGenerationMetadata*
QwenTextCompletionSession::generation_metadata() const noexcept {
  return impl_ != nullptr ? &impl_->generation : nullptr;
}

const QwenGreedyTokenSession*
QwenTextCompletionSession::token_session() const noexcept {
  return impl_ != nullptr ? &impl_->token_session : nullptr;
}

TextTokenizer* QwenTextCompletionSession::tokenizer() noexcept {
  return impl_ != nullptr ? impl_->tokenizer.get() : nullptr;
}

void QwenTextCompletionSession::reset() noexcept {
  if (impl_ != nullptr) {
    impl_->token_session.reset();
  }
}

QwenTextCompletionResult
QwenTextCompletionSession::complete(
    const QpackMlxCheckpoint& checkpoint,
    VulkanComputeContext& context,
    VulkanResidentExpertCache& expert_cache,
    std::string_view prompt,
    const QwenTextCompletionOptions& options) noexcept {
  QwenTextCompletionResult result;
  result.requested_max_new_tokens = options.max_new_tokens;

  if (!valid()) {
    result.diagnostic = "text completion session is not valid";
    return result;
  }
  if (options.max_new_tokens == 0U) {
    result.diagnostic = "text completion max_new_tokens must be non-zero";
    return result;
  }
  if (options.lm_head_chunk_rows == 0U) {
    result.diagnostic = "text completion LM-head chunk size must be non-zero";
    return result;
  }

  try {
    result.prompt_tokens = impl_->tokenizer->encode(prompt);
    if (result.prompt_tokens.empty()) {
      result.diagnostic =
          "text completion tokenizer produced an empty prompt";
      return result;
    }

    for (const auto token : result.prompt_tokens) {
      if (token >= impl_->config.vocab_size) {
        result.diagnostic =
            "text completion tokenizer produced out-of-vocabulary token";
        return result;
      }
    }

    const auto capacity = impl_->config.max_position_embeddings;
    if (result.prompt_tokens.size() > capacity) {
      result.diagnostic =
          "text completion prompt exceeds model context capacity";
      return result;
    }

    // The prediction from the final prompt token is generated without another
    // model position. Therefore a prompt that exactly fills the context may
    // still return one next-token prediction, but cannot feed that prediction
    // back into another decode step.
    const auto available_generated =
        capacity - result.prompt_tokens.size() + 1U;
    result.admitted_max_new_tokens =
        std::min(options.max_new_tokens, available_generated);

    const auto stops = merged_stop_tokens(
        impl_->generation.eos_token_ids,
        options.extra_stop_token_ids,
        impl_->config.vocab_size);

    QwenGreedySessionOptions token_options;
    token_options.max_new_tokens = result.admitted_max_new_tokens;
    token_options.lm_head_chunk_rows = options.lm_head_chunk_rows;
    token_options.stop_token_ids = stops;
    token_options.reset_before_prompt = options.reset_before_prompt;

    result.token_session = impl_->token_session.generate(
        checkpoint,
        context,
        expert_cache,
        result.prompt_tokens,
        token_options);

    if (!result.token_session.executed) {
      result.diagnostic =
          "text completion token session failed: " +
          result.token_session.diagnostic;
      return result;
    }

    result.decoded_tokens = result.token_session.generated_tokens;

    // Preserve the raw token-session record for diagnostics, but follow the
    // usual completion convention (and frozen Swiftlet CLI behavior) by not
    // emitting a terminal EOS/stop token as user-visible text.
    if (result.token_session.stop_reason ==
            QwenGreedySessionStopReason::stop_token &&
        !result.decoded_tokens.empty() &&
        contains_token(stops, result.decoded_tokens.back())) {
      result.decoded_tokens.pop_back();
    }

    result.text = result.decoded_tokens.empty()
        ? std::string{}
        : impl_->tokenizer->decode(result.decoded_tokens);

    result.executed = true;
    result.diagnostic =
        "text completion executed through tokenizer/model/tokenizer boundary";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("text completion failed: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "text completion encountered an unknown exception";
    return result;
  }
}

}  // namespace orbi::streammoe
