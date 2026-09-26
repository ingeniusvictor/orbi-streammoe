#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/tokenizer/qwen_tokenizer_assets.hpp"

namespace fs = std::filesystem;
using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& value) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("unable to write tokenizer fixture");
  out << value;
}

void replace_once(
    const fs::path& path,
    const std::string& from,
    const std::string& to) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("unable to read tokenizer fixture");
  std::string value(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>());
  const auto at = value.find(from);
  if (at == std::string::npos) {
    throw std::runtime_error("fixture replacement target missing");
  }
  value.replace(at, from.size(), to);
  write_text(path, value);
}

fs::path write_valid_fixture(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root);

  write_text(
      root / "config.json",
      R"({"model_type":"qwen3_next","vocab_size":16,"bos_token_id":10,"eos_token_id":12})");

  write_text(
      root / "tokenizer_config.json",
      R"({"tokenizer_class":"Qwen2Tokenizer","add_prefix_space":false,"add_bos_token":false,"eos_token":"<|im_end|>","pad_token":"<|endoftext|>","chat_template":"<|im_start|>{{ content }}<|im_end|>"})");

  write_text(
      root / "generation_config.json",
      R"({"bos_token_id":10,"eos_token_id":[12,10,12],"pad_token_id":10})");

  write_text(
      root / "tokenizer.json",
      R"({
        "version":"1.0",
        "normalizer":{"type":"NFC"},
        "pre_tokenizer":{
          "type":"Sequence",
          "pretokenizers":[
            {"type":"Split","pattern":{"Regex":"x"},"behavior":"Isolated","invert":false},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
          ]
        },
        "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "model":{
          "type":"BPE",
          "dropout":null,
          "unk_token":null,
          "continuing_subword_prefix":null,
          "end_of_word_suffix":null,
          "fuse_unk":false,
          "byte_fallback":false,
          "vocab":{"a":0,"b":1},
          "merges":[]
        },
        "added_tokens":[
          {"id":10,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
          {"id":12,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ]
      })");

  write_text(root / "vocab.json", R"({"a":0,"b":1})");
  write_text(root / "merges.txt", "#version: 0.2\n");

  return root;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm37a-tokenizer-assets";

  try {
    const auto valid_root = write_valid_fixture(root / "valid");
    const auto contract = inspect_qwen_tokenizer_assets(valid_root);

    require(contract.valid, contract.diagnostic);
    require(contract.model_type == "qwen3_next", "model type mismatch");
    require(
        contract.tokenizer_class == "Qwen2Tokenizer",
        "tokenizer class mismatch");
    require(contract.tokenizer_model_type == "BPE", "BPE type mismatch");
    require(contract.normalizer_type == "NFC", "normalizer mismatch");
    require(contract.decoder_type == "ByteLevel", "decoder mismatch");
    require(contract.has_split_pretokenizer, "Split pretokenizer missing");
    require(
        contract.has_byte_level_pretokenizer,
        "ByteLevel pretokenizer missing");
    require(!contract.add_prefix_space, "add_prefix_space mismatch");
    require(!contract.add_bos_token, "add_bos_token mismatch");
    require(contract.has_chat_template, "chat template must be detected");
    require(contract.has_vocab_json, "vocab.json must be detected");
    require(contract.has_merges_txt, "merges.txt must be detected");
    require(contract.model_vocab_size == 16U, "model vocab mismatch");
    require(
        contract.tokenizer_model_vocab_entries == 2U,
        "base tokenizer vocab count mismatch");
    require(contract.added_token_count == 2U, "added token count mismatch");
    require(
        contract.tokenizer_id_domain_size == 13U,
        "tokenizer ID domain mismatch");
    require(
        contract.padded_model_vocab_rows == 3U,
        "padded model vocab rows mismatch");
    require(
        contract.tokenizer_eos_token_id == std::optional<std::size_t>(12U),
        "tokenizer EOS ID mismatch");
    require(
        contract.tokenizer_pad_token_id == std::optional<std::size_t>(10U),
        "tokenizer PAD ID mismatch");
    require(
        contract.generation_stop_token_ids ==
            std::vector<std::size_t>({12U, 10U}),
        "generation stop IDs must preserve order and deduplicate");

    const auto no_legacy_root = write_valid_fixture(root / "no-legacy");
    fs::remove(no_legacy_root / "vocab.json");
    fs::remove(no_legacy_root / "merges.txt");
    const auto no_legacy = inspect_qwen_tokenizer_assets(no_legacy_root);
    require(no_legacy.valid, no_legacy.diagnostic);
    require(
        !no_legacy.has_vocab_json && !no_legacy.has_merges_txt,
        "tokenizer.json must be sufficient without legacy files");

    const auto overflow_root = write_valid_fixture(root / "domain-overflow");
    replace_once(
        overflow_root / "tokenizer.json",
        R"("id":12,"content":"<|im_end|>")",
        R"("id":16,"content":"<|im_end|>")");
    const auto overflow = inspect_qwen_tokenizer_assets(overflow_root);
    require(!overflow.valid, "tokenizer domain overflow must fail");

    const auto wrong_model_root = write_valid_fixture(root / "wrong-model");
    replace_once(
        wrong_model_root / "config.json",
        R"("model_type":"qwen3_next")",
        R"("model_type":"qwen2")");
    const auto wrong_model = inspect_qwen_tokenizer_assets(wrong_model_root);
    require(!wrong_model.valid, "wrong model type must fail");

    const auto wrong_class_root = write_valid_fixture(root / "wrong-class");
    replace_once(
        wrong_class_root / "tokenizer_config.json",
        R"("tokenizer_class":"Qwen2Tokenizer")",
        R"("tokenizer_class":"GPT2Tokenizer")");
    const auto wrong_class = inspect_qwen_tokenizer_assets(wrong_class_root);
    require(!wrong_class.valid, "wrong tokenizer class must fail");

    const auto wrong_bpe_root = write_valid_fixture(root / "wrong-bpe");
    replace_once(
        wrong_bpe_root / "tokenizer.json",
        R"("type":"BPE")",
        R"("type":"WordPiece")");
    const auto wrong_bpe = inspect_qwen_tokenizer_assets(wrong_bpe_root);
    require(!wrong_bpe.valid, "wrong tokenizer model type must fail");

    const auto wrong_norm_root = write_valid_fixture(root / "wrong-normalizer");
    replace_once(
        wrong_norm_root / "tokenizer.json",
        R"("normalizer":{"type":"NFC"})",
        R"("normalizer":{"type":"NFD"})");
    const auto wrong_norm = inspect_qwen_tokenizer_assets(wrong_norm_root);
    require(!wrong_norm.valid, "wrong normalizer must fail");

    const auto no_split_root = write_valid_fixture(root / "no-split");
    replace_once(
        no_split_root / "tokenizer.json",
        R"({"type":"Split","pattern":{"Regex":"x"},"behavior":"Isolated","invert":false})",
        R"({"type":"Whitespace"})");
    const auto no_split = inspect_qwen_tokenizer_assets(no_split_root);
    require(!no_split.valid, "missing Split pretokenizer must fail");

    const auto stop_oob_root = write_valid_fixture(root / "stop-oob");
    replace_once(
        stop_oob_root / "generation_config.json",
        R"("eos_token_id":[12,10,12])",
        R"("eos_token_id":[12,16])");
    const auto stop_oob = inspect_qwen_tokenizer_assets(stop_oob_root);
    require(!stop_oob.valid, "out-of-range generation stop must fail");

    const auto eos_mismatch_root = write_valid_fixture(root / "eos-mismatch");
    replace_once(
        eos_mismatch_root / "config.json",
        R"("eos_token_id":12)",
        R"("eos_token_id":11)");
    const auto eos_mismatch =
        inspect_qwen_tokenizer_assets(eos_mismatch_root);
    require(!eos_mismatch.valid, "model/tokenizer EOS mismatch must fail");

    const auto pad_mismatch_root = write_valid_fixture(root / "pad-mismatch");
    replace_once(
        pad_mismatch_root / "generation_config.json",
        R"("pad_token_id":10)",
        R"("pad_token_id":11)");
    const auto pad_mismatch =
        inspect_qwen_tokenizer_assets(pad_mismatch_root);
    require(!pad_mismatch.valid, "generation/tokenizer PAD mismatch must fail");

    const auto missing_root = write_valid_fixture(root / "missing");
    fs::remove(missing_root / "generation_config.json");
    const auto missing = inspect_qwen_tokenizer_assets(missing_root);
    require(!missing.valid, "missing generation config must fail");

    const auto malformed_root = write_valid_fixture(root / "malformed");
    write_text(malformed_root / "tokenizer.json", "{not-json");
    const auto malformed = inspect_qwen_tokenizer_assets(malformed_root);
    require(!malformed.valid, "malformed tokenizer JSON must fail");

    fs::remove_all(root);
    std::cout
        << "OSM-37A Qwen tokenizer assets: PASS\n"
        << "  qwen3_next_contract=PASS\n"
        << "  bpe_nfc_bytelevel_contract=PASS\n"
        << "  eos_pad_mapping=PASS\n"
        << "  generation_stop_normalization=PASS\n"
        << "  tokenizer_json_primary=PASS\n"
        << "  padded_model_vocab=PASS\n"
        << "  tokenizer_domain_guard=PASS\n"
        << "  structural_mismatch_guards=PASS\n"
        << "  special_id_guards=PASS\n"
        << "  missing_malformed_guards=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-37A Qwen tokenizer assets: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
