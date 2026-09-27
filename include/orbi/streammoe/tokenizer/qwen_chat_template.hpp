#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace orbi::streammoe {

struct QwenChatMessage {
  std::string role;
  std::string content;
};

struct QwenChatRenderResult {
  bool rendered{};
  std::string text;
  std::string diagnostic;
};

/// Native renderer for the OSM-44A-certified role/content subset of the
/// official Qwen3-Next chat template.
///
/// Supported:
/// - string content only;
/// - roles: system, user, assistant;
/// - optional assistant generation prompt.
///
/// Deliberately unsupported in OSM-44B:
/// - tools/tool calls/tool responses;
/// - multimodal content arrays;
/// - explicit reasoning_content fields.
[[nodiscard]] QwenChatRenderResult render_qwen_chat_subset(
    const std::vector<QwenChatMessage>& messages,
    bool add_generation_prompt) noexcept;

}  // namespace orbi::streammoe
