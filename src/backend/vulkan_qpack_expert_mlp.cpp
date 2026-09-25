#include "orbi/streammoe/backend/vulkan_qpack_expert_mlp.hpp"

#include <exception>
#include <string>
#include <utility>

#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"
#include "orbi/streammoe/backend/vulkan_swiglu.hpp"

namespace orbi::streammoe {

VulkanQpackExpertMlpResult run_vulkan_qpack_q4_expert_mlp(
    VulkanComputeContext& context,
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::span<const float> x) noexcept {
  VulkanQpackExpertMlpResult result;

  try {
    const auto gate_view =
        bind_qpack_q4_projection(reader, entry, "gate_proj");
    const auto up_view =
        bind_qpack_q4_projection(reader, entry, "up_proj");
    const auto down_view =
        bind_qpack_q4_projection(reader, entry, "down_proj");

    const auto input_dim = gate_view.packed_cols * 8U;
    if (x.size() != input_dim) {
      result.diagnostic =
          "qpack expert MLP input size does not match gate/up projection.";
      return result;
    }

    if (up_view.packed_cols * 8U != input_dim ||
        up_view.out_dim != gate_view.out_dim) {
      result.diagnostic =
          "qpack expert MLP gate/up geometry mismatch.";
      return result;
    }

    const auto intermediate_dim = gate_view.out_dim;
    if (down_view.packed_cols * 8U != intermediate_dim) {
      result.diagnostic =
          "qpack expert MLP down projection input does not match intermediate size.";
      return result;
    }

    const auto gate = run_vulkan_qpack_q4_projection(
        context, reader, entry, "gate_proj", x);
    if (!gate.executed) {
      result.diagnostic =
          "qpack expert MLP gate projection failed: " + gate.diagnostic;
      return result;
    }

    const auto up = run_vulkan_qpack_q4_projection(
        context, reader, entry, "up_proj", x);
    if (!up.executed) {
      result.diagnostic =
          "qpack expert MLP up projection failed: " + up.diagnostic;
      return result;
    }

    if (gate.values.size() != intermediate_dim ||
        up.values.size() != intermediate_dim) {
      result.diagnostic =
          "qpack expert MLP gate/up result size mismatch.";
      return result;
    }

    const auto hidden = run_vulkan_swiglu(
        context, gate.values, up.values);
    if (!hidden.executed) {
      result.diagnostic =
          "qpack expert MLP SwiGLU failed: " + hidden.diagnostic;
      return result;
    }

    const auto down = run_vulkan_qpack_q4_projection(
        context, reader, entry, "down_proj", hidden.values);
    if (!down.executed) {
      result.diagnostic =
          "qpack expert MLP down projection failed: " + down.diagnostic;
      return result;
    }

    if (down.values.size() != down_view.out_dim) {
      result.diagnostic =
          "qpack expert MLP down result size mismatch.";
      return result;
    }

    result.executed = true;
    result.values = down.values;
    result.diagnostic =
        "Complete streamed Q4 expert MLP executed fully through Vulkan math (gate/up/SwiGLU/down).";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("qpack expert MLP exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "qpack expert MLP encountered an unknown exception.";
    return result;
  }
}

}  // namespace orbi::streammoe
