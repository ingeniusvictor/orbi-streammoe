#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/session/generation_metadata.hpp"
#include "orbi/streammoe/session/greedy_token_session.hpp"
#include "orbi/streammoe/tokenizer/text_tokenizer.hpp"

namespace orbi::streammoe {

struct QwenTextCompletionOptions {
  std::size_t max_new_tokens{32U};
  std::size_t lm_head_chunk_rows{1024U};
  std::vector<std::size_t> extra_stop_token_ids;
  bool reset_before_prompt{true};
};

struct QwenTextCompletionResult {
  bool executed{};
  std::size_t requested_max_new_tokens{};
  std::size_t admitted_max_new_tokens{};
  std::vector<std::size_t> prompt_tokens;
  QwenGreedySessionResult token_session;
  std::vector<std::size_t> decoded_tokens;
  std::string text;
  std::string diagnostic;
};

/// Text-facing completion boundary around the token-ID autoregressive core.
///
/// Responsibilities are intentionally narrow:
///   text -> tokenizer -> token session -> tokenizer -> text
///
/// The model graph, expert streaming and vocabulary-matrix streaming remain
/// tokenizer-agnostic.
class QwenTextCompletionSession {
 public:
  QwenTextCompletionSession();
  ~QwenTextCompletionSession();

  QwenTextCompletionSession(const QwenTextCompletionSession&) = delete;
  QwenTextCompletionSession& operator=(const QwenTextCompletionSession&) = delete;

  QwenTextCompletionSession(QwenTextCompletionSession&&) noexcept;
  QwenTextCompletionSession& operator=(QwenTextCompletionSession&&) noexcept;

  [[nodiscard]] static std::optional<QwenTextCompletionSession> create(
      VulkanComputeContext& context,
      const QpackReader& qpack,
      const QpackMlxCheckpoint& checkpoint,
      const Qwen3NextDenseConfig& config,
      std::unique_ptr<TextTokenizer> tokenizer,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t vocab_size() const noexcept;
  [[nodiscard]] std::size_t context_capacity() const noexcept;
  [[nodiscard]] const QwenGenerationMetadata* generation_metadata() const noexcept;
  [[nodiscard]] const QwenGreedyTokenSession* token_session() const noexcept;
  [[nodiscard]] TextTokenizer* tokenizer() noexcept;

  void reset() noexcept;

  [[nodiscard]] QwenTextCompletionResult complete(
      const QpackMlxCheckpoint& checkpoint,
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::string_view prompt,
      const QwenTextCompletionOptions& options) noexcept;

 private:
  struct Impl;
  explicit QwenTextCompletionSession(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
