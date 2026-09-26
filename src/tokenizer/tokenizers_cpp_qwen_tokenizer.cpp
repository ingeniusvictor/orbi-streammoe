#include "orbi/streammoe/tokenizer/tokenizers_cpp_qwen_tokenizer.hpp"

#include <cstdint>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "orbi/streammoe/tokenizer/qwen_tokenizer_assets.hpp"

#if defined(ORBI_STREAMMOE_HAS_TOKENIZERS_CPP)
#include <tokenizers_c.h>
#endif

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string value) {
  if (target != nullptr) *target = std::move(value);
}

std::optional<std::string> read_blob(
    const std::filesystem::path& path,
    std::string* diagnostic) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    set_diagnostic(
        diagnostic,
        "tokenizers-cpp adapter: unable to open " + path.string());
    return std::nullopt;
  }

  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
}

}  // namespace

struct QwenTokenizersCppTokenizer::Impl {
  QwenTokenizerAssetContract contract;
#if defined(ORBI_STREAMMOE_HAS_TOKENIZERS_CPP)
  TokenizerHandle handle{};

  ~Impl() {
    if (handle != nullptr) {
      tokenizers_free(handle);
      handle = nullptr;
    }
  }
#endif
};

bool qwen_tokenizers_cpp_backend_available() noexcept {
#if defined(ORBI_STREAMMOE_HAS_TOKENIZERS_CPP)
  return true;
#else
  return false;
#endif
}

QwenTokenizersCppTokenizer::QwenTokenizersCppTokenizer() = default;
QwenTokenizersCppTokenizer::~QwenTokenizersCppTokenizer() = default;
QwenTokenizersCppTokenizer::QwenTokenizersCppTokenizer(
    QwenTokenizersCppTokenizer&&) noexcept = default;
QwenTokenizersCppTokenizer& QwenTokenizersCppTokenizer::operator=(
    QwenTokenizersCppTokenizer&&) noexcept = default;

QwenTokenizersCppTokenizer::QwenTokenizersCppTokenizer(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<QwenTokenizersCppTokenizer>
QwenTokenizersCppTokenizer::create(
    const std::filesystem::path& asset_root,
    std::string* diagnostic) noexcept {
  try {
#if !defined(ORBI_STREAMMOE_HAS_TOKENIZERS_CPP)
    (void)asset_root;
    set_diagnostic(
        diagnostic,
        "tokenizers-cpp adapter unavailable: build with "
        "ORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP=ON");
    return std::nullopt;
#else
    auto contract = inspect_qwen_tokenizer_assets(asset_root);
    if (!contract.valid) {
      set_diagnostic(
          diagnostic,
          "tokenizers-cpp adapter asset validation failed: " +
              contract.diagnostic);
      return std::nullopt;
    }

    auto blob = read_blob(asset_root / "tokenizer.json", diagnostic);
    if (!blob.has_value()) return std::nullopt;

    const auto handle = tokenizers_new_from_str(blob->data(), blob->size());
    if (handle == nullptr) {
      set_diagnostic(
          diagnostic,
          "tokenizers-cpp adapter failed to create Hugging Face tokenizer");
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->contract = std::move(contract);
    impl->handle = handle;

    if (impl->contract.eos_token.has_value() &&
        impl->contract.tokenizer_eos_token_id.has_value()) {
      std::int32_t resolved = -1;
      tokenizers_token_to_id(
          impl->handle,
          impl->contract.eos_token->data(),
          impl->contract.eos_token->size(),
          &resolved);
      if (resolved < 0 ||
          static_cast<std::size_t>(resolved) !=
              *impl->contract.tokenizer_eos_token_id) {
        set_diagnostic(
            diagnostic,
            "tokenizers-cpp adapter EOS mapping disagrees with asset contract");
        return std::nullopt;
      }
    }

    if (impl->contract.pad_token.has_value() &&
        impl->contract.tokenizer_pad_token_id.has_value()) {
      std::int32_t resolved = -1;
      tokenizers_token_to_id(
          impl->handle,
          impl->contract.pad_token->data(),
          impl->contract.pad_token->size(),
          &resolved);
      if (resolved < 0 ||
          static_cast<std::size_t>(resolved) !=
              *impl->contract.tokenizer_pad_token_id) {
        set_diagnostic(
            diagnostic,
            "tokenizers-cpp adapter PAD mapping disagrees with asset contract");
        return std::nullopt;
      }
    }

    set_diagnostic(
        diagnostic,
        "tokenizers-cpp Qwen tokenizer adapter created");
    return QwenTokenizersCppTokenizer(std::move(impl));
#endif
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("tokenizers-cpp adapter creation failed: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "tokenizers-cpp adapter creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool QwenTokenizersCppTokenizer::valid() const noexcept {
#if defined(ORBI_STREAMMOE_HAS_TOKENIZERS_CPP)
  return impl_ != nullptr && impl_->handle != nullptr && impl_->contract.valid;
#else
  return false;
#endif
}

std::optional<std::size_t>
QwenTokenizersCppTokenizer::vocab_size() const noexcept {
  if (impl_ == nullptr || !impl_->contract.valid) return std::nullopt;
  return impl_->contract.tokenizer_id_domain_size;
}

std::optional<std::size_t>
QwenTokenizersCppTokenizer::eos_token_id() const noexcept {
  if (impl_ == nullptr || !impl_->contract.valid) return std::nullopt;
  return impl_->contract.tokenizer_eos_token_id;
}

TokenizerEncodeResult QwenTokenizersCppTokenizer::encode(
    std::string_view text,
    bool add_special_tokens) const {
  TokenizerEncodeResult out;

#if !defined(ORBI_STREAMMOE_HAS_TOKENIZERS_CPP)
  (void)text;
  (void)add_special_tokens;
  out.diagnostic =
      "tokenizers-cpp adapter unavailable in this build";
  return out;
#else
  if (!valid()) {
    out.diagnostic = "tokenizers-cpp adapter is not valid";
    return out;
  }

  TokenizerEncodeResult raw{};
  tokenizers_encode(
      impl_->handle,
      text.data(),
      text.size(),
      add_special_tokens ? 1 : 0,
      &raw);

  try {
    out.token_ids.reserve(raw.len);
    for (std::size_t i = 0U; i < raw.len; ++i) {
      const auto value = raw.token_ids[i];
      if (value < 0) {
        tokenizers_free_encode_results(&raw, 1U);
        out.token_ids.clear();
        out.diagnostic =
            "tokenizers-cpp adapter produced a negative token ID";
        return out;
      }
      out.token_ids.push_back(static_cast<std::size_t>(value));
    }
  } catch (...) {
    tokenizers_free_encode_results(&raw, 1U);
    throw;
  }

  tokenizers_free_encode_results(&raw, 1U);
  out.encoded = true;
  out.diagnostic = "tokenizers-cpp encode succeeded";
  return out;
#endif
}

TokenizerDecodeResult QwenTokenizersCppTokenizer::decode(
    std::span<const std::size_t> token_ids,
    bool skip_special_tokens) const {
  TokenizerDecodeResult out;

#if !defined(ORBI_STREAMMOE_HAS_TOKENIZERS_CPP)
  (void)token_ids;
  (void)skip_special_tokens;
  out.diagnostic =
      "tokenizers-cpp adapter unavailable in this build";
  return out;
#else
  if (!valid()) {
    out.diagnostic = "tokenizers-cpp adapter is not valid";
    return out;
  }

  std::vector<std::uint32_t> ids;
  ids.reserve(token_ids.size());
  for (const auto token : token_ids) {
    if (token > std::numeric_limits<std::uint32_t>::max()) {
      out.diagnostic =
          "tokenizers-cpp adapter token ID exceeds uint32 domain";
      return out;
    }
    ids.push_back(static_cast<std::uint32_t>(token));
  }

  tokenizers_decode(
      impl_->handle,
      ids.data(),
      ids.size(),
      skip_special_tokens ? 1 : 0);

  const char* data = nullptr;
  std::size_t len = 0U;
  tokenizers_get_decode_str(impl_->handle, &data, &len);
  if (data == nullptr && len != 0U) {
    out.diagnostic =
        "tokenizers-cpp adapter returned invalid decode storage";
    return out;
  }

  out.text.assign(data != nullptr ? data : "", len);
  out.decoded = true;
  out.diagnostic = "tokenizers-cpp decode succeeded";
  return out;
#endif
}

}  // namespace orbi::streammoe
