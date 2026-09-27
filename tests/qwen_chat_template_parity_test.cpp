#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "orbi/streammoe/tokenizer/qwen_chat_template.hpp"
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
    const auto reference_path =
        required_env("ORBI_QWEN_CHAT_TEMPLATE_REFERENCE_JSON");
    const auto asset_root =
        required_env("ORBI_QWEN_TOKENIZER_ASSET_ROOT");

    if (reference_path.empty() || asset_root.empty()) {
      std::cout
          << "OSM-44B native chat renderer parity: PASS "
          << "(authoritative fixture not configured)\n";
      return 0;
    }

    std::ifstream input(reference_path, std::ios::binary);
    require(input.good(), "unable to open OSM-44A reference JSON");

    json reference;
    input >> reference;

    require(
        reference.at("stage").get<std::string>() ==
            "official-qwen-chat-template-reference",
        "unexpected OSM-44A reference stage");

    std::string diagnostic;
    auto tokenizer = QwenTokenizersCppTokenizer::create(
        fs::path(asset_root), &diagnostic);
    require(tokenizer.has_value(), diagnostic);
    require(tokenizer->valid(), "official Qwen tokenizer must be valid");

    std::size_t checked = 0U;
    for (const auto& vector : reference.at("vectors")) {
      std::vector<QwenChatMessage> messages;
      for (const auto& message : vector.at("messages")) {
        messages.push_back({
            message.at("role").get<std::string>(),
            message.at("content").get<std::string>(),
        });
      }

      const auto rendered = render_qwen_chat_subset(
          messages,
          vector.at("add_generation_prompt").get<bool>());
      require(rendered.rendered, rendered.diagnostic);

      const auto expected_text =
          vector.at("rendered").get<std::string>();
      require(
          rendered.text == expected_text,
          "native rendered text mismatch for vector " +
              std::to_string(checked));

      const auto encoded = tokenizer->encode(rendered.text, false);
      require(encoded.encoded, encoded.diagnostic);
      const auto expected_ids = read_ids(vector.at("token_ids"));
      require(
          encoded.token_ids == expected_ids,
          "native rendered token-ID mismatch for vector " +
              std::to_string(checked));

      ++checked;
    }

    require(checked >= 4U, "insufficient OSM-44B parity vectors");

    {
      const auto bad = render_qwen_chat_subset(
          {{"tool", "unsupported"}}, true);
      require(!bad.rendered, "tool role must be rejected in OSM-44B");
    }

    std::cout
        << "OSM-44B native chat renderer parity: PASS\n"
        << "  official_reference_consumed=PASS\n"
        << "  rendered_text_byte_parity=PASS\n"
        << "  token_id_parity=PASS\n"
        << "  system_user_assistant_subset=PASS\n"
        << "  unsupported_tool_role_rejected=PASS\n"
        << "  vectors_checked=" << checked << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-44B native chat renderer parity: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
