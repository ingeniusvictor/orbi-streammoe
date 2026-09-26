#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "orbi/streammoe/session/text_tokenizer_boundary.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

class FixtureTokenizer final : public Tokenizer {
 public:
  std::optional<std::size_t> vocab{32U};
  std::optional<std::size_t> eos{2U};
  bool fail_encode{};
  bool fail_decode{};
  bool empty_encode{};
  bool oov_encode{};
  bool throw_encode{};
  bool throw_decode{};

  [[nodiscard]] std::optional<std::size_t> vocab_size() const noexcept override {
    return vocab;
  }

  [[nodiscard]] std::optional<std::size_t> eos_token_id() const noexcept override {
    return eos;
  }

  [[nodiscard]] TokenizerEncodeResult encode(
      std::string_view text,
      bool add_special_tokens) const override {
    if (throw_encode) throw std::runtime_error("encode boom");
    if (fail_encode) {
      return {false, {}, "fixture encode failure"};
    }
    if (empty_encode) {
      return {true, {}, "fixture empty encode"};
    }
    if (oov_encode) {
      return {true, {32U}, "fixture OOV encode"};
    }
    if (text != "hello") {
      return {false, {}, "unexpected fixture prompt"};
    }

    TokenizerEncodeResult result;
    result.encoded = true;
    result.token_ids = add_special_tokens
        ? std::vector<std::size_t>{1U, 5U, 7U}
        : std::vector<std::size_t>{5U, 7U};
    result.diagnostic = "fixture encoded";
    return result;
  }

  [[nodiscard]] TokenizerDecodeResult decode(
      std::span<const std::size_t> token_ids,
      bool skip_special_tokens) const override {
    if (throw_decode) throw std::runtime_error("decode boom");
    if (fail_decode) {
      return {false, {}, "fixture decode failure"};
    }
    if (token_ids != std::span<const std::size_t>(
            expected_generated_.data(),
            expected_generated_.size())) {
      return {false, {}, "unexpected fixture generated tokens"};
    }

    TokenizerDecodeResult result;
    result.decoded = true;
    result.text = skip_special_tokens ? "world" : "world<EOS>";
    result.diagnostic = "fixture decoded";
    return result;
  }

 private:
  const std::vector<std::size_t> expected_generated_{9U, 2U};
};

}  // namespace

int main() {
  try {
    FixtureTokenizer tokenizer;

    QwenGreedySessionOptions session_options;
    session_options.max_new_tokens = 8U;
    session_options.lm_head_chunk_rows = 64U;
    session_options.stop_token_ids = {11U};
    session_options.reset_before_prompt = false;

    const auto prepared = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        session_options);
    require(prepared.prepared, prepared.diagnostic);
    require(
        prepared.prompt_tokens ==
            std::vector<std::size_t>({1U, 5U, 7U}),
        "special-token encode mismatch");
    require(
        prepared.session_options.max_new_tokens == 8U,
        "max_new_tokens must pass through boundary");
    require(
        prepared.session_options.lm_head_chunk_rows == 64U,
        "LM-head chunk size must pass through boundary");
    require(
        !prepared.session_options.reset_before_prompt,
        "reset semantics must pass through boundary");
    require(
        prepared.session_options.stop_token_ids ==
            std::vector<std::size_t>({11U, 2U}),
        "tokenizer EOS must merge into stop tokens");

    QwenTextTokenizerBoundaryOptions no_special;
    no_special.add_special_tokens = false;
    no_special.use_tokenizer_eos_as_stop = false;
    const auto raw = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        session_options,
        no_special);
    require(raw.prepared, raw.diagnostic);
    require(
        raw.prompt_tokens == std::vector<std::size_t>({5U, 7U}),
        "raw encode mismatch");
    require(
        raw.session_options.stop_token_ids ==
            std::vector<std::size_t>({11U}),
        "EOS stop must remain disabled");

    QwenGreedySessionOptions already_has_eos = session_options;
    already_has_eos.stop_token_ids = {2U};
    const auto deduplicated = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        already_has_eos);
    require(deduplicated.prepared, deduplicated.diagnostic);
    require(
        deduplicated.session_options.stop_token_ids.size() == 1U,
        "EOS stop token must not be duplicated");

    QwenGreedySessionResult token_result;
    token_result.executed = true;
    token_result.stop_reason = QwenGreedySessionStopReason::stop_token;
    token_result.prompt_tokens = prepared.prompt_tokens;
    token_result.generated_tokens = {9U, 2U};
    token_result.generated_logits = {1.0F, 2.0F};
    token_result.model_steps = 4U;
    token_result.diagnostic = "fixture token session success";

    const auto decoded = decode_qwen_generated_text(
        tokenizer,
        token_result);
    require(decoded.decoded, decoded.diagnostic);
    require(decoded.text == "world", "skip-special decode mismatch");
    require(
        decoded.generated_tokens == token_result.generated_tokens,
        "decoded token echo mismatch");

    QwenTextTokenizerBoundaryOptions keep_special;
    keep_special.skip_special_tokens = false;
    const auto decoded_with_special = decode_qwen_generated_text(
        tokenizer,
        token_result,
        keep_special);
    require(decoded_with_special.decoded, decoded_with_special.diagnostic);
    require(
        decoded_with_special.text == "world<EOS>",
        "special-token decode mismatch");

    tokenizer.fail_encode = true;
    const auto encode_failure = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        session_options);
    require(!encode_failure.prepared, "encode failure must fail boundary");
    tokenizer.fail_encode = false;

    tokenizer.empty_encode = true;
    const auto empty = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        session_options);
    require(!empty.prepared, "empty encode must fail boundary");
    tokenizer.empty_encode = false;

    tokenizer.oov_encode = true;
    const auto oov_prompt = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        session_options);
    require(!oov_prompt.prepared, "OOV prompt must fail boundary");
    tokenizer.oov_encode = false;

    tokenizer.eos = 32U;
    const auto bad_eos = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        session_options);
    require(!bad_eos.prepared, "OOV EOS must fail boundary");
    tokenizer.eos = 2U;

    tokenizer.throw_encode = true;
    const auto encode_exception = prepare_qwen_text_prompt(
        tokenizer,
        "hello",
        session_options);
    require(
        !encode_exception.prepared,
        "encode exception must be contained by boundary");
    tokenizer.throw_encode = false;

    QwenGreedySessionResult failed_session;
    failed_session.executed = false;
    failed_session.diagnostic = "model failed";
    const auto no_decode = decode_qwen_generated_text(
        tokenizer,
        failed_session);
    require(!no_decode.decoded, "failed token session must not decode");

    QwenGreedySessionResult empty_generated = token_result;
    empty_generated.generated_tokens.clear();
    const auto empty_decode = decode_qwen_generated_text(
        tokenizer,
        empty_generated);
    require(!empty_decode.decoded, "empty generation must not decode");

    QwenGreedySessionResult oov_generated = token_result;
    oov_generated.generated_tokens = {32U};
    const auto oov_decode = decode_qwen_generated_text(
        tokenizer,
        oov_generated);
    require(!oov_decode.decoded, "OOV generation must not decode");

    tokenizer.fail_decode = true;
    const auto decode_failure = decode_qwen_generated_text(
        tokenizer,
        token_result);
    require(!decode_failure.decoded, "decode failure must fail boundary");
    tokenizer.fail_decode = false;

    tokenizer.throw_decode = true;
    const auto decode_exception = decode_qwen_generated_text(
        tokenizer,
        token_result);
    require(
        !decode_exception.decoded,
        "decode exception must be contained by boundary");

    std::cout
        << "OSM-36B tokenizer boundary: PASS\n"
        << "  encode_contract=PASS\n"
        << "  EOS_stop_merge=PASS\n"
        << "  option_passthrough=PASS\n"
        << "  decode_contract=PASS\n"
        << "  tokenizer_error_guards=PASS\n"
        << "  vocabulary_guards=PASS\n"
        << "  exception_boundary=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-36B tokenizer boundary: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
