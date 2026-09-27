#include "orbi/streammoe/tokenizer/qwen_chat_template.hpp"

#include <exception>
#include <string>
#include <utility>

namespace orbi::streammoe {

QwenChatRenderResult render_qwen_chat_subset(
    const std::vector<QwenChatMessage>& messages,
    bool add_generation_prompt) noexcept {
  QwenChatRenderResult out;

  try {
    if (messages.empty()) {
      out.diagnostic = "Qwen chat renderer requires at least one message";
      return out;
    }

    for (const auto& message : messages) {
      if (message.role != "system" &&
          message.role != "user" &&
          message.role != "assistant") {
        out.diagnostic =
            "OSM-44B supports only system/user/assistant roles";
        return out;
      }
    }

    std::string rendered;

    if (messages.front().role == "system") {
      rendered += "<|im_start|>system\n";
      rendered += messages.front().content;
      rendered += "<|im_end|>\n";
    }

    for (std::size_t i = 0U; i < messages.size(); ++i) {
      const auto& message = messages[i];

      if (i == 0U && message.role == "system") {
        continue;
      }

      rendered += "<|im_start|>";
      rendered += message.role;
      rendered += "\n";
      rendered += message.content;
      rendered += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
      rendered += "<|im_start|>assistant\n";
    }

    out.rendered = true;
    out.text = std::move(rendered);
    out.diagnostic = "Qwen native chat subset rendered";
    return out;
  } catch (const std::exception& e) {
    out.diagnostic =
        std::string("Qwen native chat renderer failed: ") + e.what();
    return out;
  } catch (...) {
    out.diagnostic =
        "Qwen native chat renderer encountered an unknown exception";
    return out;
  }
}

}  // namespace orbi::streammoe
