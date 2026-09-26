#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "orbi/streammoe/tokenizer/tokenizer.hpp"

namespace orbi::streammoe {

/// True when ORBI StreamMoE was built with the optional tokenizers-cpp backend.
[[nodiscard]] bool qwen_tokenizers_cpp_backend_available() noexcept;

/// Concrete Qwen tokenizer adapter backed by the Hugging Face Tokenizers engine
/// exposed through mlc-ai/tokenizers-cpp.
///
/// The dependency is optional so the numerical runtime remains buildable
/// without Rust/tokenizer tooling. When enabled, this adapter preserves the
/// Tokenizer contract's add_special_tokens and skip_special_tokens controls.
class QwenTokenizersCppTokenizer final : public Tokenizer {
 public:
  QwenTokenizersCppTokenizer();
  ~QwenTokenizersCppTokenizer() override;

  QwenTokenizersCppTokenizer(const QwenTokenizersCppTokenizer&) = delete;
  QwenTokenizersCppTokenizer& operator=(const QwenTokenizersCppTokenizer&) = delete;

  QwenTokenizersCppTokenizer(QwenTokenizersCppTokenizer&&) noexcept;
  QwenTokenizersCppTokenizer& operator=(QwenTokenizersCppTokenizer&&) noexcept;

  [[nodiscard]] static std::optional<QwenTokenizersCppTokenizer> create(
      const std::filesystem::path& asset_root,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;

  [[nodiscard]] std::optional<std::size_t> vocab_size() const noexcept override;
  [[nodiscard]] std::optional<std::size_t> eos_token_id() const noexcept override;

  [[nodiscard]] TokenizerEncodeResult encode(
      std::string_view text,
      bool add_special_tokens) const override;

  [[nodiscard]] TokenizerDecodeResult decode(
      std::span<const std::size_t> token_ids,
      bool skip_special_tokens) const override;

 private:
  struct Impl;
  explicit QwenTokenizersCppTokenizer(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
