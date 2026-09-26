#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "orbi/streammoe/session/greedy_token_session.hpp"
#include "orbi/streammoe/session/text_tokenizer_boundary.hpp"
#include "orbi/streammoe/tokenizer/tokenizer.hpp"

namespace orbi::streammoe {

struct QwenGreedyTextSessionOptions {
  QwenGreedySessionOptions generation;
  QwenTextTokenizerBoundaryOptions tokenizer;
};

enum class QwenGreedyTextSessionStatus {
  completed,
  invalid_request,
  tokenizer_error,
  model_error,
};

struct QwenGreedyTextSessionResult {
  bool completed{};
  bool token_session_executed{};
  QwenGreedyTextSessionStatus status{
      QwenGreedyTextSessionStatus::invalid_request};
  QwenGreedySessionStopReason stop_reason{
      QwenGreedySessionStopReason::invalid_request};
  std::size_t model_steps{};
  std::vector<std::size_t> prompt_tokens;
  std::vector<std::size_t> generated_tokens;
  std::vector<float> generated_logits;
  std::string text;
  std::string diagnostic;
};

/// Stateful text facade over the certified token-ID autoregressive runtime.
///
/// The facade owns the OSM-36A numerical session, while concrete tokenization
/// remains injected through the OSM-36B Tokenizer contract. This keeps text
/// handling replaceable across Windows, Android/JNI, and future native ports.
class QwenGreedyTextSession {
 public:
  QwenGreedyTextSession();
  ~QwenGreedyTextSession();

  QwenGreedyTextSession(const QwenGreedyTextSession&) = delete;
  QwenGreedyTextSession& operator=(const QwenGreedyTextSession&) = delete;

  QwenGreedyTextSession(QwenGreedyTextSession&&) noexcept;
  QwenGreedyTextSession& operator=(QwenGreedyTextSession&&) noexcept;

  [[nodiscard]] static std::optional<QwenGreedyTextSession> create(
      VulkanComputeContext& context,
      const QpackMlxCheckpoint& checkpoint,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t vocab_size() const noexcept;
  [[nodiscard]] const QwenGreedyTokenSession* token_session() const noexcept;

  void reset() noexcept;

  [[nodiscard]] QwenGreedyTextSessionResult generate(
      const QpackMlxCheckpoint& checkpoint,
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      const Tokenizer& tokenizer,
      std::string_view prompt,
      const QwenGreedyTextSessionOptions& options) noexcept;

 private:
  struct Impl;
  explicit QwenGreedyTextSession(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
