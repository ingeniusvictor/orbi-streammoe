#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/model/checkpoint_model_shell.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"

namespace orbi::streammoe {

struct QwenGreedySessionOptions {
  std::size_t max_new_tokens{1U};
  std::size_t lm_head_chunk_rows{1024U};
  std::vector<std::size_t> stop_token_ids;
  bool reset_before_prompt{true};
};

enum class QwenGreedySessionStopReason {
  max_new_tokens,
  stop_token,
  model_error,
  invalid_request,
};

struct QwenGreedySessionResult {
  bool executed{};
  QwenGreedySessionStopReason stop_reason{
      QwenGreedySessionStopReason::invalid_request};
  std::size_t model_steps{};
  std::vector<std::size_t> prompt_tokens;
  std::vector<std::size_t> generated_tokens;
  std::vector<float> generated_logits;
  std::string diagnostic;
};

/// Correctness-first token-ID autoregressive session.
///
/// Prompt tokens are consumed with teacher forcing. The prediction produced by
/// the final prompt token becomes generated token 0. Each later generated token
/// is fed back into the model until a stop token or max_new_tokens is reached.
class QwenGreedyTokenSession {
 public:
  QwenGreedyTokenSession();
  ~QwenGreedyTokenSession();

  QwenGreedyTokenSession(const QwenGreedyTokenSession&) = delete;
  QwenGreedyTokenSession& operator=(const QwenGreedyTokenSession&) = delete;

  QwenGreedyTokenSession(QwenGreedyTokenSession&&) noexcept;
  QwenGreedyTokenSession& operator=(QwenGreedyTokenSession&&) noexcept;

  [[nodiscard]] static std::optional<QwenGreedyTokenSession> create(
      VulkanComputeContext& context,
      const QpackMlxCheckpoint& checkpoint,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t vocab_size() const noexcept;
  [[nodiscard]] const QwenCheckpointModelShell* model_shell() const noexcept;

  void reset() noexcept;

  [[nodiscard]] QwenGreedySessionResult generate(
      const QpackMlxCheckpoint& checkpoint,
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::span<const std::size_t> prompt_tokens,
      const QwenGreedySessionOptions& options) noexcept;

 private:
  struct Impl;
  explicit QwenGreedyTokenSession(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
