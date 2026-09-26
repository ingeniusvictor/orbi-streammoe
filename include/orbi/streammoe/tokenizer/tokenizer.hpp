#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace orbi::streammoe {

struct TokenizerEncodeResult {
  bool encoded{};
  std::vector<std::size_t> token_ids;
  std::string diagnostic;
};

struct TokenizerDecodeResult {
  bool decoded{};
  std::string text;
  std::string diagnostic;
};

/// Portable tokenizer integration contract.
///
/// Numerical model execution intentionally depends only on token IDs. Concrete
/// tokenizer implementations may live in a platform adapter, native library,
/// JNI layer, or another component without coupling tokenization to Vulkan,
/// expert streaming, or checkpoint execution.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  [[nodiscard]] virtual std::optional<std::size_t> vocab_size() const noexcept {
    return std::nullopt;
  }

  [[nodiscard]] virtual std::optional<std::size_t> eos_token_id() const noexcept {
    return std::nullopt;
  }

  [[nodiscard]] virtual TokenizerEncodeResult encode(
      std::string_view text,
      bool add_special_tokens) const = 0;

  [[nodiscard]] virtual TokenizerDecodeResult decode(
      std::span<const std::size_t> token_ids,
      bool skip_special_tokens) const = 0;
};

}  // namespace orbi::streammoe
