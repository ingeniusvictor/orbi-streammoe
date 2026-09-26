#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/tokenizer/tokenizers_cpp_qwen_tokenizer.hpp"

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

fs::path write_fixture(const fs::path& root) {
  fs::remove_all(root);
  fs::create_directories(root);

  write_text(
      root / "config.json",
      R"({"model_type":"qwen3_next","vocab_size":16,"bos_token_id":10,"eos_token_id":12})");

  write_text(
      root / "tokenizer_config.json",
      R"({"tokenizer_class":"Qwen2Tokenizer","add_prefix_space":false,"add_bos_token":false,"eos_token":"<|im_end|>","pad_token":"<|endoftext|>","chat_template":"{{ content }}"})");

  write_text(
      root / "generation_config.json",
      R"({"bos_token_id":10,"eos_token_id":[12,10],"pad_token_id":10})");

  write_text(
      root / "tokenizer.json",
      R"({
        "version":"1.0",
        "truncation":null,
        "padding":null,
        "added_tokens":[
          {"id":10,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
          {"id":12,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ],
        "normalizer":{"type":"NFC"},
        "pre_tokenizer":{
          "type":"Sequence",
          "pretokenizers":[
            {"type":"Split","pattern":{"Regex":"x"},"behavior":"Isolated","invert":false},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}
          ]
        },
        "post_processor":null,
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
        }
      })");

  return root;
}

}  // namespace

int main() {
  const auto root =
      fs::temp_directory_path() / "orbi-streammoe-osm37b-tokenizers-cpp";

  try {
    if (!qwen_tokenizers_cpp_backend_available()) {
      std::string diagnostic;
      const auto unavailable =
          QwenTokenizersCppTokenizer::create(root, &diagnostic);
      require(
          !unavailable.has_value(),
          "disabled tokenizers-cpp backend must not create an adapter");
      require(
          diagnostic.find("ORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP=ON") !=
              std::string::npos,
          "disabled backend must explain how to enable integration");
      std::cout
          << "OSM-37B tokenizers-cpp adapter: PASS (backend disabled)\n";
      return 0;
    }

    write_fixture(root);

    std::string diagnostic;
    auto tokenizer =
        QwenTokenizersCppTokenizer::create(root, &diagnostic);
    require(tokenizer.has_value(), diagnostic);
    require(tokenizer->valid(), "adapter must be valid");
    require(
        tokenizer->vocab_size() == std::optional<std::size_t>(13U),
        "tokenizer ID-domain size mismatch");
    require(
        tokenizer->eos_token_id() == std::optional<std::size_t>(12U),
        "tokenizer EOS mismatch");

    const auto encoded = tokenizer->encode("ab", false);
    require(encoded.encoded, encoded.diagnostic);
    require(
        encoded.token_ids == std::vector<std::size_t>({0U, 1U}),
        "tokenizers-cpp encode mismatch");

    const auto decoded = tokenizer->decode(encoded.token_ids, false);
    require(decoded.decoded, decoded.diagnostic);
    require(decoded.text == "ab", "tokenizers-cpp decode mismatch");

    const auto special = tokenizer->encode("<|im_end|>", true);
    require(special.encoded, special.diagnostic);
    require(
        special.token_ids == std::vector<std::size_t>({12U}),
        "special-token encode mismatch");

    const std::vector<std::size_t> eos_only{12U};
    const auto keep_special = tokenizer->decode(eos_only, false);
    require(keep_special.decoded, keep_special.diagnostic);
    require(
        keep_special.text == "<|im_end|>",
        "special-token decode must preserve token when requested");

    const auto skip_special = tokenizer->decode(eos_only, true);
    require(skip_special.decoded, skip_special.diagnostic);
    require(
        skip_special.text.empty(),
        "skip_special_tokens must suppress special token");

    fs::remove_all(root);
    std::cout
        << "OSM-37B tokenizers-cpp adapter: PASS\n"
        << "  hf_tokenizer_json_load=PASS\n"
        << "  qwen_asset_contract=PASS\n"
        << "  encode=PASS\n"
        << "  decode=PASS\n"
        << "  add_special_tokens=PASS\n"
        << "  skip_special_tokens=PASS\n"
        << "  eos_mapping=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(root);
    std::cerr
        << "OSM-37B tokenizers-cpp adapter: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
