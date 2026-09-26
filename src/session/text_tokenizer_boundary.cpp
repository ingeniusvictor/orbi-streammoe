#include "orbi/streammoe/session/text_tokenizer_boundary.hpp"

#include <algorithm>
#include <exception>
#include <string>

namespace orbi::streammoe {
namespace {

bool contains_token(
    std::span<const std::size_t> tokens,
    std::size_t token) noexcept {
  return std::find(tokens.begin(), tokens.end(), token) != tokens.end();
}

bool token_ids_fit_vocab(
    std::span<const std::size_t> tokens,
    std::optional<std::size_t> vocab_size) noexcept {
  if (!vocab_size.has_value()) return true;
  return std::all_of(
      tokens.begin(),
      tokens.end(),
      [&](std::size_t token) { return token < *vocab_size; });
}

}  // namespace

QwenPreparedTextPrompt prepare_qwen_text_prompt(
    const Tokenizer& tokenizer,
    std::string_view prompt,
    const QwenGreedySessionOptions& session_options,
    const QwenTextTokenizerBoundaryOptions& boundary_options) noexcept {
  QwenPreparedTextPrompt result;
  result.session_options = session_options;

  try {
    const auto encoded =
        tokenizer.encode(prompt, boundary_options.add_special_tokens);
    if (!encoded.encoded) {
      result.diagnostic =
          "tokenizer encode failed: " + encoded.diagnostic;
      return result;
    }
    if (encoded.token_ids.empty()) {
      result.diagnostic =
          "tokenizer encode produced an empty token sequence";
      return result;
    }

    const auto vocab = tokenizer.vocab_size();
    if (!token_ids_fit_vocab(encoded.token_ids, vocab)) {
      result.diagnostic =
          "tokenizer encode produced a token outside tokenizer vocabulary";
      return result;
    }

    result.prompt_tokens = encoded.token_ids;

    if (boundary_options.use_tokenizer_eos_as_stop) {
      const auto eos = tokenizer.eos_token_id();
      if (eos.has_value()) {
        if (vocab.has_value() && *eos >= *vocab) {
          result.prompt_tokens.clear();
          result.diagnostic =
              "tokenizer EOS token exceeds tokenizer vocabulary";
          return result;
        }
        if (!contains_token(
                result.session_options.stop_token_ids,
                *eos)) {
          result.session_options.stop_token_ids.push_back(*eos);
        }
      }
    }

    result.prepared = true;
    result.diagnostic =
        "text prompt encoded for Qwen greedy token session";
    return result;
  } catch (const std::exception& e) {
    result.prompt_tokens.clear();
    result.diagnostic =
        std::string("tokenizer encode threw an exception: ") + e.what();
    return result;
  } catch (...) {
    result.prompt_tokens.clear();
    result.diagnostic =
        "tokenizer encode threw an unknown exception";
    return result;
  }
}

QwenDecodedTextResult decode_qwen_generated_text(
    const Tokenizer& tokenizer,
    const QwenGreedySessionResult& session_result,
    const QwenTextTokenizerBoundaryOptions& boundary_options) noexcept {
  QwenDecodedTextResult result;

  if (!session_result.executed) {
    result.diagnostic =
        "cannot decode unsuccessful token session: " +
        session_result.diagnostic;
    return result;
  }
  if (session_result.generated_tokens.empty()) {
    result.diagnostic =
        "cannot decode an empty generated token sequence";
    return result;
  }

  const auto vocab = tokenizer.vocab_size();
  if (!token_ids_fit_vocab(session_result.generated_tokens, vocab)) {
    result.diagnostic =
        "generated token exceeds tokenizer vocabulary";
    return result;
  }

  try {
    const auto decoded = tokenizer.decode(
        session_result.generated_tokens,
        boundary_options.skip_special_tokens);
    if (!decoded.decoded) {
      result.diagnostic =
          "tokenizer decode failed: " + decoded.diagnostic;
      return result;
    }

    result.generated_tokens = session_result.generated_tokens;
    result.text = decoded.text;
    result.decoded = true;
    result.diagnostic =
        "generated token sequence decoded";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("tokenizer decode threw an exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "tokenizer decode threw an unknown exception";
    return result;
  }
}

}  // namespace orbi::streammoe
