#include "orbi/streammoe/tokenizer/qwen_tokenizer_assets.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

namespace orbi::streammoe {
namespace {

using json = nlohmann::json;

bool read_json(
    const std::filesystem::path& path,
    json& value,
    std::string& diagnostic) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    diagnostic = "tokenizer assets: unable to open " + path.string();
    return false;
  }

  try {
    input >> value;
    return true;
  } catch (const json::exception& e) {
    diagnostic =
        "tokenizer assets: invalid JSON in " + path.string() + ": " + e.what();
    return false;
  }
}

std::optional<std::size_t> optional_size(
    const json& object,
    const char* key) {
  if (!object.contains(key) || object.at(key).is_null()) {
    return std::nullopt;
  }
  return object.at(key).get<std::size_t>();
}

std::optional<std::string> optional_string(
    const json& object,
    const char* key) {
  if (!object.contains(key) || object.at(key).is_null()) {
    return std::nullopt;
  }
  return object.at(key).get<std::string>();
}

void collect_component_types(
    const json& component,
    bool& has_split,
    bool& has_byte_level) {
  if (!component.is_object()) return;

  const auto type = component.value("type", std::string{});
  if (type == "Split") has_split = true;
  if (type == "ByteLevel") has_byte_level = true;

  for (const auto* key : {"pretokenizers", "pre_tokenizers"}) {
    if (component.contains(key) && component.at(key).is_array()) {
      for (const auto& child : component.at(key)) {
        collect_component_types(child, has_split, has_byte_level);
      }
    }
  }
}

bool append_stop_ids(
    const json& value,
    std::vector<std::size_t>& output) {
  if (value.is_number_unsigned() || value.is_number_integer()) {
    output.push_back(value.get<std::size_t>());
    return true;
  }
  if (!value.is_array()) return false;

  for (const auto& item : value) {
    if (!item.is_number_unsigned() && !item.is_number_integer()) {
      return false;
    }
    output.push_back(item.get<std::size_t>());
  }
  return true;
}

void deduplicate(std::vector<std::size_t>& values) {
  std::vector<std::size_t> unique;
  unique.reserve(values.size());
  for (const auto value : values) {
    if (std::find(unique.begin(), unique.end(), value) == unique.end()) {
      unique.push_back(value);
    }
  }
  values = std::move(unique);
}

bool update_max_id(
    std::size_t id,
    bool& have_id,
    std::size_t& max_id) {
  if (!have_id || id > max_id) {
    max_id = id;
    have_id = true;
  }
  return true;
}

}  // namespace

QwenTokenizerAssetContract inspect_qwen_tokenizer_assets(
    const std::filesystem::path& root) noexcept {
  QwenTokenizerAssetContract out;
  out.root = root;

  try {
    json model_config;
    json tokenizer;
    json tokenizer_config;
    json generation_config;

    if (!read_json(root / "config.json", model_config, out.diagnostic) ||
        !read_json(root / "tokenizer.json", tokenizer, out.diagnostic) ||
        !read_json(
            root / "tokenizer_config.json",
            tokenizer_config,
            out.diagnostic) ||
        !read_json(
            root / "generation_config.json",
            generation_config,
            out.diagnostic)) {
      return out;
    }

    out.has_vocab_json = std::filesystem::exists(root / "vocab.json");
    out.has_merges_txt = std::filesystem::exists(root / "merges.txt");

    out.model_type = model_config.at("model_type").get<std::string>();
    out.model_vocab_size = model_config.at("vocab_size").get<std::size_t>();
    out.model_bos_token_id = optional_size(model_config, "bos_token_id");
    out.model_eos_token_id = optional_size(model_config, "eos_token_id");

    out.tokenizer_class =
        tokenizer_config.at("tokenizer_class").get<std::string>();
    out.add_prefix_space =
        tokenizer_config.value("add_prefix_space", false);
    out.add_bos_token =
        tokenizer_config.value("add_bos_token", false);
    out.has_chat_template =
        tokenizer_config.contains("chat_template") &&
        tokenizer_config.at("chat_template").is_string() &&
        !tokenizer_config.at("chat_template").get<std::string>().empty();
    out.eos_token = optional_string(tokenizer_config, "eos_token");
    out.pad_token = optional_string(tokenizer_config, "pad_token");

    const auto& model = tokenizer.at("model");
    out.tokenizer_model_type = model.at("type").get<std::string>();
    const auto& vocab = model.at("vocab");
    if (!vocab.is_object() || vocab.empty()) {
      out.diagnostic = "tokenizer assets: tokenizer model vocab is empty";
      return out;
    }
    out.tokenizer_model_vocab_entries = vocab.size();

    bool have_id = false;
    std::size_t max_id = 0U;
    std::unordered_map<std::string, std::size_t> token_ids;
    token_ids.reserve(vocab.size());

    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
      const auto id = it.value().get<std::size_t>();
      token_ids[it.key()] = id;
      update_max_id(id, have_id, max_id);
    }

    if (tokenizer.contains("added_tokens")) {
      const auto& added = tokenizer.at("added_tokens");
      if (!added.is_array()) {
        out.diagnostic =
            "tokenizer assets: tokenizer added_tokens must be an array";
        return out;
      }
      out.added_token_count = added.size();
      for (const auto& entry : added) {
        const auto id = entry.at("id").get<std::size_t>();
        const auto content = entry.at("content").get<std::string>();
        token_ids[content] = id;
        update_max_id(id, have_id, max_id);
      }
    }

    if (!have_id || max_id == std::numeric_limits<std::size_t>::max()) {
      out.diagnostic = "tokenizer assets: invalid tokenizer token-ID domain";
      return out;
    }
    out.tokenizer_id_domain_size = max_id + 1U;

    if (tokenizer.contains("normalizer") &&
        tokenizer.at("normalizer").is_object()) {
      out.normalizer_type =
          tokenizer.at("normalizer").value("type", std::string{});
    }
    if (tokenizer.contains("decoder") &&
        tokenizer.at("decoder").is_object()) {
      out.decoder_type =
          tokenizer.at("decoder").value("type", std::string{});
    }
    if (tokenizer.contains("pre_tokenizer")) {
      collect_component_types(
          tokenizer.at("pre_tokenizer"),
          out.has_split_pretokenizer,
          out.has_byte_level_pretokenizer);
    }

    if (out.eos_token.has_value()) {
      const auto it = token_ids.find(*out.eos_token);
      if (it != token_ids.end()) out.tokenizer_eos_token_id = it->second;
    }
    if (out.pad_token.has_value()) {
      const auto it = token_ids.find(*out.pad_token);
      if (it != token_ids.end()) out.tokenizer_pad_token_id = it->second;
    }

    out.generation_bos_token_id =
        optional_size(generation_config, "bos_token_id");
    out.generation_pad_token_id =
        optional_size(generation_config, "pad_token_id");
    if (!generation_config.contains("eos_token_id") ||
        !append_stop_ids(
            generation_config.at("eos_token_id"),
            out.generation_stop_token_ids)) {
      out.diagnostic =
          "tokenizer assets: generation eos_token_id must be an ID or ID array";
      return out;
    }
    deduplicate(out.generation_stop_token_ids);

    if (out.model_type != "qwen3_next") {
      out.diagnostic =
          "tokenizer assets: model_type must be qwen3_next";
      return out;
    }
    if (out.model_vocab_size == 0U) {
      out.diagnostic =
          "tokenizer assets: model vocab_size must be non-zero";
      return out;
    }
    if (out.tokenizer_class != "Qwen2Tokenizer" &&
        out.tokenizer_class != "Qwen2TokenizerFast") {
      out.diagnostic =
          "tokenizer assets: tokenizer_class must be Qwen2Tokenizer-compatible";
      return out;
    }
    if (out.tokenizer_model_type != "BPE") {
      out.diagnostic =
          "tokenizer assets: tokenizer.json must use BPE";
      return out;
    }
    if (out.normalizer_type != "NFC") {
      out.diagnostic =
          "tokenizer assets: Qwen tokenizer normalizer must be NFC";
      return out;
    }
    if (out.decoder_type != "ByteLevel") {
      out.diagnostic =
          "tokenizer assets: Qwen tokenizer decoder must be ByteLevel";
      return out;
    }
    if (!out.has_split_pretokenizer ||
        !out.has_byte_level_pretokenizer) {
      out.diagnostic =
          "tokenizer assets: Qwen tokenizer requires Split + ByteLevel pretokenization";
      return out;
    }
    if (out.tokenizer_id_domain_size > out.model_vocab_size) {
      out.diagnostic =
          "tokenizer assets: tokenizer token-ID domain exceeds model vocabulary";
      return out;
    }
    out.padded_model_vocab_rows =
        out.model_vocab_size - out.tokenizer_id_domain_size;

    auto id_in_model = [&](std::size_t id) {
      return id < out.model_vocab_size;
    };
    for (const auto id : out.generation_stop_token_ids) {
      if (!id_in_model(id)) {
        out.diagnostic =
            "tokenizer assets: generation stop token exceeds model vocabulary";
        return out;
      }
    }
    for (const auto id : {
             out.model_bos_token_id,
             out.model_eos_token_id,
             out.generation_bos_token_id,
             out.generation_pad_token_id,
             out.tokenizer_eos_token_id,
             out.tokenizer_pad_token_id}) {
      if (id.has_value() && !id_in_model(*id)) {
        out.diagnostic =
            "tokenizer assets: special token ID exceeds model vocabulary";
        return out;
      }
    }

    if (!out.tokenizer_eos_token_id.has_value()) {
      out.diagnostic =
          "tokenizer assets: tokenizer eos_token has no token-ID mapping";
      return out;
    }
    if (!out.tokenizer_pad_token_id.has_value()) {
      out.diagnostic =
          "tokenizer assets: tokenizer pad_token has no token-ID mapping";
      return out;
    }
    if (out.model_eos_token_id.has_value() &&
        *out.model_eos_token_id != *out.tokenizer_eos_token_id) {
      out.diagnostic =
          "tokenizer assets: model and tokenizer EOS IDs disagree";
      return out;
    }
    if (std::find(
            out.generation_stop_token_ids.begin(),
            out.generation_stop_token_ids.end(),
            *out.tokenizer_eos_token_id) ==
        out.generation_stop_token_ids.end()) {
      out.diagnostic =
          "tokenizer assets: tokenizer EOS is missing from generation stop IDs";
      return out;
    }
    if (out.generation_pad_token_id.has_value() &&
        *out.generation_pad_token_id != *out.tokenizer_pad_token_id) {
      out.diagnostic =
          "tokenizer assets: generation and tokenizer PAD IDs disagree";
      return out;
    }

    out.valid = true;
    out.diagnostic = "Qwen3-Next tokenizer asset contract validated";
    return out;
  } catch (const json::exception& e) {
    out.diagnostic =
        std::string("tokenizer assets: malformed JSON contract: ") + e.what();
    return out;
  } catch (const std::exception& e) {
    out.diagnostic =
        std::string("tokenizer assets: inspection failed: ") + e.what();
    return out;
  } catch (...) {
    out.diagnostic =
        "tokenizer assets: inspection encountered an unknown exception";
    return out;
  }
}

}  // namespace orbi::streammoe
