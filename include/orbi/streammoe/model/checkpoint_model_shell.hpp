#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "orbi/streammoe/backend/vulkan_compute_context.hpp"
#include "orbi/streammoe/backend/vulkan_resident_expert_cache.hpp"
#include "orbi/streammoe/container/mlx_affine_checkpoint.hpp"
#include "orbi/streammoe/model/checkpoint_decoder_stack.hpp"
#include "orbi/streammoe/model/qwen_dense_binding.hpp"
#include "orbi/streammoe/model/qwen_global_binding.hpp"
#include "orbi/streammoe/model/qwen_global_streaming.hpp"

namespace orbi::streammoe {

struct QwenCheckpointModelStepResult {
  bool executed{};
  std::size_t input_token{};
  std::vector<float> embedding;
  std::vector<float> decoder_hidden;
  std::vector<float> final_hidden;
  QwenGreedyTokenResult greedy;
  std::string diagnostic;
};

/// Correctness-first end-to-end Qwen3-Next model shell.
///
/// One decode step:
///   token -> streamed embedding row -> decoder stack -> final RMSNorm
///         -> streamed LM-head argmax
class QwenCheckpointModelShell {
 public:
  QwenCheckpointModelShell();
  ~QwenCheckpointModelShell();

  QwenCheckpointModelShell(const QwenCheckpointModelShell&) = delete;
  QwenCheckpointModelShell& operator=(const QwenCheckpointModelShell&) = delete;

  QwenCheckpointModelShell(QwenCheckpointModelShell&&) noexcept;
  QwenCheckpointModelShell& operator=(QwenCheckpointModelShell&&) noexcept;

  [[nodiscard]] static std::optional<QwenCheckpointModelShell> create(
      VulkanComputeContext& context,
      const QpackMlxCheckpoint& checkpoint,
      const Qwen3NextDenseConfig& config,
      std::string* diagnostic = nullptr) noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::size_t hidden_size() const noexcept;
  [[nodiscard]] std::size_t vocab_size() const noexcept;
  [[nodiscard]] std::size_t layer_count() const noexcept;
  [[nodiscard]] const QwenGlobalCheckpointBinding* global_binding() const noexcept;
  [[nodiscard]] const QwenCheckpointDecoderStack* decoder_stack() const noexcept;

  void reset_state() noexcept;

  [[nodiscard]] QwenCheckpointModelStepResult step_greedy(
      const QpackMlxCheckpoint& checkpoint,
      VulkanComputeContext& context,
      VulkanResidentExpertCache& expert_cache,
      std::size_t token_id,
      std::size_t lm_head_chunk_rows) noexcept;

 private:
  struct Impl;
  explicit QwenCheckpointModelShell(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace orbi::streammoe
