#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace orbi::streammoe {

struct QwenTokenizerAssetContract {
  bool valid{};
  std::filesystem::path root;

  std::string model_type;
  std::string tokenizer_class;
  std::string tokenizer_model_type;
  std::string normalizer_type;
  std::string decoder_type;

  bool has_split_pretokenizer{};
  bool has_byte_level_pretokenizer{};
  bool add_prefix_space{};
  bool add_bos_token{};
  bool has_chat_template{};
  bool has_vocab_json{};
  bool has_merges_txt{};

  std::size_t model_vocab_size{};
  std::size_t tokenizer_model_vocab_entries{};
  std::size_t added_token_count{};
  std::size_t tokenizer_id_domain_size{};
  std::size_t padded_model_vocab_rows{};

  std::optional<std::string> eos_token;
  std::optional<std::string> pad_token;
  std::optional<std::size_t> tokenizer_eos_token_id;
  std::optional<std::size_t> tokenizer_pad_token_id;
  std::optional<std::size_t> model_bos_token_id;
  std::optional<std::size_t> model_eos_token_id;
  std::optional<std::size_t> generation_bos_token_id;
  std::optional<std::size_t> generation_pad_token_id;
  std::vector<std::size_t> generation_stop_token_ids;

  std::string diagnostic;
};

/// Inspect and validate the tokenizer assets colocated with a Qwen3-Next model.
///
/// Required:
/// - config.json
/// - tokenizer.json
/// - tokenizer_config.json
/// - generation_config.json
///
/// tokenizer.json is the canonical executable tokenizer artifact. vocab.json
/// and merges.txt are recorded when present but are not required by this gate.
[[nodiscard]] QwenTokenizerAssetContract inspect_qwen_tokenizer_assets(
    const std::filesystem::path& root) noexcept;

}  // namespace orbi::streammoe
