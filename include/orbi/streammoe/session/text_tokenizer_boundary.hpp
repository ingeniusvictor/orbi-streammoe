#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "orbi/streammoe/session/greedy_token_session.hpp"
#include "orbi/streammoe/tokenizer/tokenizer.hpp"

namespace orbi::streammoe {

struct QwenTextTokenizerBoundaryOptions {
  bool add_special_tokens{true};
  bool skip_special_tokens{true};
  bool use_tokenizer_eos_as_stop{true};
};

struct QwenPreparedTextPrompt {
  bool prepared{};
  std::vector<std::size_t> prompt_tokens;
  QwenGreedySessionOptions session_options;
  std::string diagnostic;
};

struct QwenDecodedTextResult {
  bool decoded{};
  std::vector<std::size_t> generated_tokens;
  std::string text;
  std::string diagnostic;
};

/// Encode text into the token-ID contract consumed by OSM-36A.
///
/// When requested, the tokenizer EOS token is merged into the session stop set
/// without changing caller-provided stop tokens.
[[nodiscard]] QwenPreparedTextPrompt prepare_qwen_text_prompt(
    const Tokenizer& tokenizer,
    std::string_view prompt,
    const QwenGreedySessionOptions& session_options,
    const QwenTextTokenizerBoundaryOptions& boundary_options = {}) noexcept;

/// Decode generated token IDs from an OSM-36A result.
///
/// This function deliberately performs no model execution; it is the portable
/// boundary between the numerical token session and a concrete tokenizer.
[[nodiscard]] QwenDecodedTextResult decode_qwen_generated_text(
    const Tokenizer& tokenizer,
    const QwenGreedySessionResult& session_result,
    const QwenTextTokenizerBoundaryOptions& boundary_options = {}) noexcept;

}  // namespace orbi::streammoe
