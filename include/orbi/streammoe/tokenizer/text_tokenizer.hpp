#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace orbi::streammoe {

/// Portable text/token boundary.
///
/// Concrete tokenizer backends may wrap Hugging Face tokenizers, SentencePiece,
/// a platform tokenizer, or a deterministic test fixture. Model execution never
/// depends on a concrete tokenizer implementation.
class TextTokenizer {
 public:
  virtual ~TextTokenizer() = default;

  [[nodiscard]] virtual std::vector<std::size_t> encode(
      std::string_view text) = 0;

  [[nodiscard]] virtual std::string decode(
      std::span<const std::size_t> token_ids) = 0;

  [[nodiscard]] virtual std::size_t vocab_size() const noexcept = 0;
};

}  // namespace orbi::streammoe
