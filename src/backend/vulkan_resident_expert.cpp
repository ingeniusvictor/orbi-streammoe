#include "orbi/streammoe/backend/vulkan_resident_expert.hpp"

#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "orbi/streammoe/backend/vulkan_float_buffer.hpp"
#include "orbi/streammoe/backend/vulkan_q4_gemv.hpp"
#include "orbi/streammoe/backend/vulkan_q4_weights.hpp"
#include "orbi/streammoe/backend/vulkan_qpack_expert.hpp"
#include "orbi/streammoe/backend/vulkan_swiglu.hpp"

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
      *diagnostic = std::string("resident expert failed to create ") + label;
    }
    return false;
  }
  target.emplace(std::move(*source));
  return true;
}

}  // namespace

struct VulkanResidentExpert::Impl {
  std::uintptr_t device{};
  std::size_t input_dim{};
  std::size_t intermediate_dim{};
  std::size_t output_dim{};

  std::optional<VulkanQ4ProjectionWeights> gate_weights;
  std::optional<VulkanQ4ProjectionWeights> up_weights;
  std::optional<VulkanQ4ProjectionWeights> down_weights;

  std::optional<VulkanFloatBuffer> input;
  std::optional<VulkanFloatBuffer> gate;
  std::optional<VulkanFloatBuffer> up;
  std::optional<VulkanFloatBuffer> hidden;
  std::optional<VulkanFloatBuffer> output;
};

VulkanResidentExpert::VulkanResidentExpert() = default;
VulkanResidentExpert::~VulkanResidentExpert() = default;
VulkanResidentExpert::VulkanResidentExpert(
    VulkanResidentExpert&&) noexcept = default;
VulkanResidentExpert& VulkanResidentExpert::operator=(
    VulkanResidentExpert&&) noexcept = default;

VulkanResidentExpert::VulkanResidentExpert(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

std::optional<VulkanResidentExpert> VulkanResidentExpert::create(
    VulkanComputeContext& context,
    const QpackReader& reader,
    const ExpertCacheEntry& entry,
    std::string* diagnostic) noexcept {
  if (!context.valid()) {
    set_diagnostic(diagnostic, "resident expert requires a valid Vulkan context");
    return std::nullopt;
  }

  try {
    const auto gate =
        bind_qpack_q4_projection(reader, entry, "gate_proj");
    const auto up =
        bind_qpack_q4_projection(reader, entry, "up_proj");
    const auto down =
        bind_qpack_q4_projection(reader, entry, "down_proj");

    const auto input_dim = gate.packed_cols * 8U;
    const auto intermediate_dim = gate.out_dim;
    const auto output_dim = down.out_dim;

    if (input_dim == 0U || intermediate_dim == 0U || output_dim == 0U) {
      set_diagnostic(
          diagnostic, "resident expert dimensions must be non-zero");
      return std::nullopt;
    }

    if (up.packed_cols * 8U != input_dim ||
        up.out_dim != intermediate_dim) {
      set_diagnostic(
          diagnostic, "resident expert gate/up geometry mismatch");
      return std::nullopt;
    }

    if (down.packed_cols * 8U != intermediate_dim) {
      set_diagnostic(
          diagnostic,
          "resident expert down input does not match intermediate dimension");
      return std::nullopt;
    }

    auto impl = std::make_unique<Impl>();
    impl->device = context.native_device();
    impl->input_dim = input_dim;
    impl->intermediate_dim = intermediate_dim;
    impl->output_dim = output_dim;

    std::string local;

    if (!move_optional(
            VulkanQ4ProjectionWeights::create(
                context,
                gate.packed,
                gate.out_dim,
                gate.packed_cols,
                gate.scales,
                gate.biases,
                gate.group_size,
                &local),
            impl->gate_weights,
            &local,
            "gate projection")) {
      set_diagnostic(diagnostic, "resident expert gate upload failed: " + local);
      return std::nullopt;
    }

    local.clear();
    if (!move_optional(
            VulkanQ4ProjectionWeights::create(
                context,
                up.packed,
                up.out_dim,
                up.packed_cols,
                up.scales,
                up.biases,
                up.group_size,
                &local),
            impl->up_weights,
            &local,
            "up projection")) {
      set_diagnostic(diagnostic, "resident expert up upload failed: " + local);
      return std::nullopt;
    }

    local.clear();
    if (!move_optional(
            VulkanQ4ProjectionWeights::create(
                context,
                down.packed,
                down.out_dim,
                down.packed_cols,
                down.scales,
                down.biases,
                down.group_size,
                &local),
            impl->down_weights,
            &local,
            "down projection")) {
      set_diagnostic(diagnostic, "resident expert down upload failed: " + local);
      return std::nullopt;
    }

    local.clear();
    if (!move_optional(
            VulkanFloatBuffer::create(context, input_dim, &local),
            impl->input,
            &local,
            "input activation") ||
        !move_optional(
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
          diagnostic, "resident expert activation allocation failed: " + local);
      return std::nullopt;
    }

    set_diagnostic(
        diagnostic,
        "Vulkan resident expert created with gate/up/down weights preloaded.");
    return VulkanResidentExpert(std::move(impl));
  } catch (const std::exception& e) {
    set_diagnostic(
        diagnostic,
        std::string("resident expert creation exception: ") + e.what());
    return std::nullopt;
  } catch (...) {
    set_diagnostic(
        diagnostic, "resident expert creation encountered an unknown exception");
    return std::nullopt;
  }
}

bool VulkanResidentExpert::valid() const noexcept {
  return impl_ != nullptr &&
         impl_->device != 0U &&
         impl_->gate_weights.has_value() &&
         impl_->up_weights.has_value() &&
         impl_->down_weights.has_value() &&
         impl_->gate_weights->valid() &&
         impl_->up_weights->valid() &&
         impl_->down_weights->valid() &&
         impl_->input.has_value() &&
         impl_->gate.has_value() &&
         impl_->up.has_value() &&
         impl_->hidden.has_value() &&
         impl_->output.has_value() &&
         impl_->input->valid() &&
         impl_->gate->valid() &&
         impl_->up->valid() &&
         impl_->hidden->valid() &&
         impl_->output->valid();
}

std::size_t VulkanResidentExpert::input_dim() const noexcept {
  return impl_ != nullptr ? impl_->input_dim : 0U;
}

std::size_t VulkanResidentExpert::intermediate_dim() const noexcept {
  return impl_ != nullptr ? impl_->intermediate_dim : 0U;
}

std::size_t VulkanResidentExpert::output_dim() const noexcept {
  return impl_ != nullptr ? impl_->output_dim : 0U;
}

std::size_t VulkanResidentExpert::packed_weight_bytes() const noexcept {
  if (!valid()) return 0U;
  return impl_->gate_weights->packed_size_bytes() +
         impl_->up_weights->packed_size_bytes() +
         impl_->down_weights->packed_size_bytes();
}

std::size_t VulkanResidentExpert::accounted_bytes() const noexcept {
  if (!valid()) return 0U;

  const auto projection_bytes = [](const VulkanQ4ProjectionWeights& weights) {
    return weights.packed_size_bytes() +
           weights.scales_buffer().size_bytes() +
           weights.biases_buffer().size_bytes();
  };

  const auto weight_bytes =
      projection_bytes(*impl_->gate_weights) +
      projection_bytes(*impl_->up_weights) +
      projection_bytes(*impl_->down_weights);

  const auto activation_bytes =
      impl_->input->size_bytes() +
      impl_->gate->size_bytes() +
      impl_->up->size_bytes() +
      impl_->hidden->size_bytes() +
      impl_->output->size_bytes();

  return weight_bytes + activation_bytes;
}

std::uintptr_t VulkanResidentExpert::native_device() const noexcept {
  return impl_ != nullptr ? impl_->device : 0U;
}

VulkanBufferDispatchResult VulkanResidentExpert::run_from_buffer(
    VulkanComputeContext& context,
    const VulkanFloatBuffer& x) noexcept {
  VulkanBufferDispatchResult result;

  if (!valid()) {
    result.diagnostic = "resident expert is not valid";
    return result;
  }
  if (!context.valid() || context.native_device() != impl_->device) {
    result.diagnostic = "resident expert belongs to a different Vulkan device";
    return result;
  }
  if (!x.valid() ||
      x.native_device() != impl_->device ||
      x.size() != impl_->input_dim) {
    result.diagnostic =
        "resident expert input Vulkan buffer is invalid or has wrong shape/device";
    return result;
  }

  try {
    const auto gate = run_vulkan_q4_gemv_preloaded(
        context,
        *impl_->gate_weights,
        x,
        *impl_->gate);
    if (!gate.executed) {
      result.diagnostic =
          "resident expert gate projection failed: " + gate.diagnostic;
      return result;
    }

    const auto up = run_vulkan_q4_gemv_preloaded(
        context,
        *impl_->up_weights,
        x,
        *impl_->up);
    if (!up.executed) {
      result.diagnostic =
          "resident expert up projection failed: " + up.diagnostic;
      return result;
    }

    const auto hidden = run_vulkan_swiglu_buffers(
        context,
        *impl_->gate,
        *impl_->up,
        *impl_->hidden);
    if (!hidden.executed) {
      result.diagnostic =
          "resident expert SwiGLU failed: " + hidden.diagnostic;
      return result;
    }

    const auto down = run_vulkan_q4_gemv_preloaded(
        context,
        *impl_->down_weights,
        *impl_->hidden,
        *impl_->output);
    if (!down.executed) {
      result.diagnostic =
          "resident expert down projection failed: " + down.diagnostic;
      return result;
    }

    result.executed = true;
    result.diagnostic =
        "Vulkan resident expert output remains in reusable device buffer.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("resident expert buffer execution exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "resident expert buffer execution encountered an unknown exception";
    return result;
  }
}

const VulkanFloatBuffer* VulkanResidentExpert::output_buffer() const noexcept {
  if (!valid()) return nullptr;
  return &*impl_->output;
}

VulkanResidentExpertResult VulkanResidentExpert::run(
    VulkanComputeContext& context,
    std::span<const float> x) noexcept {
  VulkanResidentExpertResult result;

  if (!valid()) {
    result.diagnostic = "resident expert is not valid";
    return result;
  }
  if (!context.valid() || context.native_device() != impl_->device) {
    result.diagnostic = "resident expert belongs to a different Vulkan device";
    return result;
  }
  if (x.size() != impl_->input_dim) {
    result.diagnostic = "resident expert input size mismatch";
    return result;
  }

  try {
    std::string diagnostic;
    if (!impl_->input->upload(x, &diagnostic)) {
      result.diagnostic =
          "resident expert input upload failed: " + diagnostic;
      return result;
    }

    const auto dispatch =
        run_from_buffer(context, *impl_->input);
    if (!dispatch.executed) {
      result.diagnostic = dispatch.diagnostic;
      return result;
    }

    const auto values = impl_->output->download(&diagnostic);
    if (!values.has_value()) {
      result.diagnostic =
          "resident expert output download failed: " + diagnostic;
      return result;
    }

    result.executed = true;
    result.values = *values;
    result.diagnostic =
        "Vulkan resident expert executed through reusable buffer path.";
    return result;
  } catch (const std::exception& e) {
    result.diagnostic =
        std::string("resident expert execution exception: ") + e.what();
    return result;
  } catch (...) {
    result.diagnostic =
        "resident expert execution encountered an unknown exception";
    return result;
  }
}

}  // namespace orbi::streammoe
