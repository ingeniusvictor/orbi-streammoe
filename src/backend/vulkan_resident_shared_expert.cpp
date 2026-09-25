#include "orbi/streammoe/backend/vulkan_resident_shared_expert.hpp"

#include <cmath>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "orbi/streammoe/backend/vulkan_q4_gemv.hpp"
#include "orbi/streammoe/backend/vulkan_q4_weights.hpp"
#include "orbi/streammoe/backend/vulkan_swiglu.hpp"
#include "orbi/streammoe/cpu/reference_ops.hpp"

namespace orbi::streammoe {
namespace {

void set_diagnostic(std::string* target, std::string message) {
  if (target != nullptr) *target = std::move(message);
}

template <typename T>
bool move_optional(
    std::optional<T> source,
    std::optional<T>& target,
    std::string* diagnostic,
    const char* label) {
  if (!source.has_value()) {
    if (diagnostic != nullptr && diagnostic->empty()) {
      *diagnostic = std::string("shared expert failed to create ") + label;
    }
    return false;
  }
  target.emplace(std::move(*source));
  return true;
}

bool valid_projection(
    const VulkanSharedExpertProjectionView& view) {
  if (view.out_dim == 0U ||
      view.packed_cols == 0U ||
      view.group_size == 0U) {
    return false;
  }

  const auto input_dim = view.packed_cols * 8U;
  if ((input_dim % view.group_size) != 0U) return false;

  const auto groups_per_row = input_dim / view.group_size;
  return
      view.packed.size() == view.out_dim * view.packed_cols &&
      view.scales.size() == view.out_dim * groups_per_row &&
      view.biases.size() == view.out_dim * groups_per_row;
}

}  // namespace

struct VulkanResidentSharedExpert::Impl {
  std::uintptr_t device{};
  std::size_t input_dim{};
  std::size_t intermediate_dim{};
  std::size_t output_dim{};

  std::optional<VulkanQ4ProjectionWeights> gate_weights;
  std::optional<VulkanQ4ProjectionWeights> up_weights;
  std::optional<VulkanQ4ProjectionWeights> down_weights;

  std::optional<VulkanFloatBuffer> gate;
  std::optional<VulkanFloatBuffer> up;
  std::optional<VulkanFloatBuffer> hidden;
  std::optional<VulkanFloatBuffer> output;

  std::vector<float> scalar_gate;
};

VulkanResidentSharedExpert::VulkanResidentSharedExpert() = default;
VulkanResidentSharedExpert::~VulkanResidentSharedExpert() = default;
VulkanResidentSharedExpert::VulkanResidentSharedExpert(
    VulkanResidentSharedExpert&&) noexcept = default;
VulkanResidentSharedExpert& VulkanResidentSharedExpert::operator=(
    VulkanResidentSharedExpert&&) noexcept = default;

VulkanResidentSharedExpert::VulkanResidentSharedExpert(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<VulkanResidentSharedExpert>
VulkanResidentSharedExpert::create(
    VulkanComputeContext& context,
    VulkanSharedExpertWeights weights,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(diagnostic, "shared expert requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    if (!valid_projection(weights.gate) ||
        !valid_projection(weights.up) ||
        !valid_projection(weights.down)) {
      set_diagnostic(
          diagnostic,
          "shared expert Q4 projection geometry is invalid");
      return std::nullopt;
    }

    const auto input_dim = weights.gate.packed_cols * 8U;
    const auto intermediate_dim = weights.gate.out_dim;
    const auto output_dim = weights.down.out_dim;

    if (weights.up.packed_cols * 8U != input_dim ||
        weights.up.out_dim != intermediate_dim ||
        weights.down.packed_cols * 8U != intermediate_dim ||
        output_dim != input_dim ||
        weights.scalar_gate.size() != input_dim) {
      set_diagnostic(
          diagnostic,
          "shared expert gate/up/down/scalar geometry mismatch");
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->device = context.native_device();
    impl->input_dim = input_dim;
    impl->intermediate_dim = intermediate_dim;
    impl->output_dim = output_dim;
    impl->scalar_gate.assign(
        weights.scalar_gate.begin(),
        weights.scalar_gate.end());

    std::string local;

    if (!move_optional(
            VulkanQ4ProjectionWeights::create(
                context,
                weights.gate.packed,
                weights.gate.out_dim,
                weights.gate.packed_cols,
                weights.gate.scales,
                weights.gate.biases,
                weights.gate.group_size,
                &local),
            impl->gate_weights,
            &local,
            "gate projection")) {
      set_diagnostic(
          diagnostic,
          "shared expert gate upload failed: " + local);
      return std::nullopt;
    }

    local.clear();
    if (!move_optional(
            VulkanQ4ProjectionWeights::create(
                context,
                weights.up.packed,
                weights.up.out_dim,
                weights.up.packed_cols,
                weights.up.scales,
                weights.up.biases,
                weights.up.group_size,
                &local),
            impl->up_weights,
            &local,
            "up projection")) {
      set_diagnostic(
          diagnostic,
          "shared expert up upload failed: " + local);
      return std::nullopt;
    }

    local.clear();
    if (!move_optional(
            VulkanQ4ProjectionWeights::create(
                context,
                weights.down.packed,
                weights.down.out_dim,
                weights.down.packed_cols,
                weights.down.scales,
                weights.down.biases,
                weights.down.group_size,
                &local),
            impl->down_weights,
            &local,
            "down projection")) {
      set_diagnostic(
          diagnostic,
          "shared expert down upload failed: " + local);
      return std::nullopt;
    }

    local.clear();
    if (!move_optional(
            VulkanFloatBuffer::create(context, intermediate_dim, &local),
            impl->gate,
            &local,
            "gate activation") ||
        !move_optional(
            VulkanFloatBuffer::create(context, intermediate_dim, &local),
            impl->up,
            &local,
            "up activation") ||
        !move_optional(
            VulkanFloatBuffer::create(context, intermediate_dim, &local),
            impl->hidden,
            &local,
            "hidden activation") ||
        !move_optional(
            VulkanFloatBuffer::create(context, output_dim, &local),
            impl->output,
            &local,
            "output activation")) {
      set_diagnostic(
          diagnostic,
          "shared expert activation allocation failed: " + local);
      return std::nullopt;
    }

    set_diagnostic(
        diagnostic,
        "Vulkan shared expert created with resident Q4 projections.");
    return VulkanResidentSharedExpert(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("shared expert creation exception: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic,
        "shared expert creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool VulkanResidentSharedExpert::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->device != 0U &&
         impl_->gate_weights.has_value() &&
         impl_->up_weights.has_value() &&
         impl_->down_weights.has_value() &&
         impl_->gate_weights->valid() &&
         impl_->up_weights->valid() &&
         impl_->down_weights->valid() &&
         impl_->gate.has_value() &&
         impl_->up.has_value() &&
         impl_->hidden.has_value() &&
         impl_->output.has_value() &&
         impl_->gate->valid() &&
         impl_->up->valid() &&
         impl_->hidden->valid() &&
         impl_->output->valid() &&
         impl_->scalar_gate.size() == impl_->input_dim;
}

std::size_t VulkanResidentSharedExpert::input_dim() const noexcept {
  return impl_ != nullptr ? impl_->input_dim : 0U;
}

std::size_t VulkanResidentSharedExpert::intermediate_dim() const noexcept {
  return impl_ != nullptr ? impl_->intermediate_dim : 0U;
}

std::size_t VulkanResidentSharedExpert::output_dim() const noexcept {
  return impl_ != nullptr ? impl_->output_dim : 0U;
}

std::size_t VulkanResidentSharedExpert::accounted_vulkan_bytes() const noexcept {
  if (!valid()) return 0U;

  const auto projection_bytes = [](const VulkanQ4ProjectionWeights& weights) {
    return weights.packed_size_bytes() +
           weights.scales_buffer().size_bytes() +
           weights.biases_buffer().size_bytes();
  };

  return
      projection_bytes(*impl_->gate_weights) +
      projection_bytes(*impl_->up_weights) +
      projection_bytes(*impl_->down_weights) +
      impl_->gate->size_bytes() +
      impl_->up->size_bytes() +
      impl_->hidden->size_bytes() +
      impl_->output->size_bytes();
}

std::uintptr_t VulkanResidentSharedExpert::native_device() const noexcept {
  return impl_ != nullptr ? impl_->device : 0U;
}

std::optional<float> VulkanResidentSharedExpert::scalar_scale(
    std::span<const float> hidden,
    std::string* diagnostic) const noexcept {
  if (!valid()) {
    set_diagnostic(diagnostic, "shared expert is not valid");
    return std::nullopt;
  }
  if (hidden.size() != impl_->input_dim) {
    set_diagnostic(diagnostic, "shared expert scalar gate input size mismatch");
    return std::nullopt;
  }

  float logit = 0.0F;
  for (std::size_t i = 0; i < hidden.size(); ++i) {
    logit += impl_->scalar_gate[i] * hidden[i];
  }

  const auto scale = cpu::sigmoid(logit);
  if (!std::isfinite(scale)) {
    set_diagnostic(diagnostic, "shared expert scalar gate is non-finite");
    return std::nullopt;
  }

  set_diagnostic(diagnostic, "shared expert scalar gate computed");
  return scale;
}

VulkanBufferDispatchResult VulkanResidentSharedExpert::run_from_buffer(
    VulkanComputeContext& context,
    const VulkanFloatBuffer& hidden) noexcept {
  VulkanBufferDispatchResult result;

  if (!valid()) {
    result.diagnostic = "shared expert is not valid";
    return result;
  }
  if (!context.valid() ||
      context.native_device() != impl_->device ||
      !hidden.valid() ||
      hidden.native_device() != impl_->device ||
      hidden.size() != impl_->input_dim) {
    result.diagnostic =
        "shared expert input buffer has wrong device or geometry";
    return result;
  }

  try {
    const auto gate = run_vulkan_q4_gemv_preloaded(
        context,
        *impl_->gate_weights,
        hidden,
        *impl_->gate);
    if (!gate.executed) {
      result.diagnostic =
          "shared expert gate projection failed: " + gate.diagnostic;
      return result;
    }

    const auto up = run_vulkan_q4_gemv_preloaded(
        context,
        *impl_->up_weights,
        hidden,
        *impl_->up);
    if (!up.executed) {
      result.diagnostic =
          "shared expert up projection failed: " + up.diagnostic;
      return result;
    }

    const auto swiglu = run_vulkan_swiglu_buffers(
        context,
        *impl_->gate,
        *impl_->up,
        *impl_->hidden);
    if (!swiglu.executed) {
      result.diagnostic =
          "shared expert SwiGLU failed: " + swiglu.diagnostic;
      return result;
    }

    const auto down = run_vulkan_q4_gemv_preloaded(
        context,
        *impl_->down_weights,
        *impl_->hidden,
        *impl_->output);
    if (!down.executed) {
      result.diagnostic =
          "shared expert down projection failed: " + down.diagnostic;
      return result;
    }

    result.executed = true;
    result.diagnostic =
        "Vulkan shared expert MLP output remains device-resident.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("shared expert Vulkan exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "shared expert Vulkan execution encountered an unknown exception";
    return result;
  }
}

const VulkanFloatBuffer*
VulkanResidentSharedExpert::output_buffer() const noexcept {
  if (!valid()) return nullptr;
  return &*impl_->output;
}

}  // namespace orbi::streammoe
