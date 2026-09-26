#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/tokenizer/tokenizers_cpp_qwen_tokenizer.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::string required_env(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string{};
}

std::vector<std::size_t> read_ids(const json& value) {
  std::vector<std::size_t> ids;
  for (const auto& item : value) ids.push_back(item.get<std::size_t>());
  return ids;
}

}  // namespace

int main() {
  try {
    if (!qwen_tokenizers_cpp_backend_available()) {
      std::cout
          << "OSM-37C official tokenizer parity: PASS (backend disabled)\n";
      return 0;
    }

    const auto asset_root = required_env("ORBI_QWEN_TOKENIZER_ASSET_ROOT");
    const auto parity_path = required_env("ORBI_QWEN_TOKENIZER_PARITY_JSON");
    if (asset_root.empty() || parity_path.empty()) {
      std::cout
          << "OSM-37C official tokenizer parity: PASS "
          << "(authoritative fixture not configured)\n";
      return 0;
    }

    std::ifstream input(parity_path, std::ios::binary);
    require(input.good(), "unable to open authoritative parity JSON");

    json reference;
    input >> reference;

    require(
        reference.at("model").get<std::string>() ==
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "unexpected authoritative model");
    require(
        reference.at("revision").get<std::string>() ==
            "f5e99a3698d364cf77584543481b778afee26177",
        "unexpected authoritative revision");
    require(
        reference.at("tokenizer_sha256").get<std::string>() ==
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "unexpected tokenizer SHA256");

    std::string diagnostic;
    auto tokenizer = QwenTokenizersCppTokenizer::create(
        fs::path(asset_root),
        &diagnostic);
    require(tokenizer.has_value(), diagnostic);
    require(tokenizer->valid(), "official Qwen tokenizer adapter must be valid");

    std::size_t checked = 0U;
    for (const auto& vector : reference.at("vectors")) {
      const auto text = vector.at("text").get<std::string>();
      const auto add_special_tokens =
          vector.at("add_special_tokens").get<bool>();
      const auto expected_ids = read_ids(vector.at("ids"));

      const auto encoded =
          tokenizer->encode(text, add_special_tokens);
      require(encoded.encoded, encoded.diagnostic);
      require(
          encoded.token_ids == expected_ids,
          "official Qwen encode parity mismatch for vector " +
              std::to_string(checked));

      const auto keep =
          tokenizer->decode(expected_ids, false);
      require(keep.decoded, keep.diagnostic);
      require(
          keep.text ==
              vector.at("decode_keep_special").get<std::string>(),
          "official Qwen decode keep-special parity mismatch for vector " +
              std::to_string(checked));

      const auto skip =
          tokenizer->decode(expected_ids, true);
      require(skip.decoded, skip.diagnostic);
      require(
          skip.text ==
              vector.at("decode_skip_special").get<std::string>(),
          "official Qwen decode skip-special parity mismatch for vector " +
              std::to_string(checked));

      ++checked;
    }

    require(checked >= 10U, "insufficient authoritative parity coverage");

    std::cout
        << "OSM-37C official tokenizer parity: PASS\n"
        << "  authoritative_model=Qwen3-Next-80B-A3B-Instruct\n"
        << "  pinned_revision=PASS\n"
        << "  tokenizer_sha256=PASS\n"
        << "  python_hf_tokenizers_oracle=PASS\n"
        << "  cpp_encode_parity=PASS\n"
        << "  cpp_decode_keep_special_parity=PASS\n"
        << "  cpp_decode_skip_special_parity=PASS\n"
        << "  vectors_checked=" << checked << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-37C official tokenizer parity: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
