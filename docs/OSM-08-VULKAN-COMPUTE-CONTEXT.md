# OSM-08 — Vulkan compute context

Status: **IMPLEMENTED / CI GATE**

## Goal

Move beyond capability probing and create the minimum real Vulkan compute
execution context required by later StreamMoE kernels.

OSM-08 deliberately does **not** introduce model kernels yet. It establishes
the lifetime and ownership boundary that those kernels will depend on.

## Runtime objects

A successful `VulkanComputeContext` owns:

- `VkInstance`
- selected `VkPhysicalDevice`
- `VkDevice`
- one compute-capable `VkQueue`
- one resettable/transient `VkCommandPool`

The public header does not include Vulkan headers. Native handles are exposed
only as opaque integer-sized values for diagnostics/integration.

## Device selection

Selection is intentionally simple and deterministic:

1. reject devices without a compute queue;
2. prefer a compute-only queue family when available;
3. prefer discrete GPU over integrated GPU, virtual GPU, then CPU Vulkan.

This is a bootstrap policy, not the final performance policy.

## Cross-platform rule

The context uses only core Vulkan + volk. No Win32 surface, Android surface,
presentation queue, or GUI dependency is introduced.

That keeps the same compute context usable by:

- headless Windows / L.U.M.I.A.
- Android NDK / ORBI Edge Mesh
- Linux CI and development hosts

## CI behavior

CI machines are not required to expose a physical Vulkan GPU.

The test therefore verifies two valid states:

- no Vulkan compute device: graceful diagnostic and no context;
- compute device available: real instance/device/queue/command-pool creation.

A loader/device absence is not silently treated as proof that GPU execution
works; it is only a portability-path validation.

## Exit gate

OSM-08 is green when:

- the project builds on Windows and Linux;
- context creation has deterministic ownership/cleanup;
- a real compute-capable host can create all required handles;
- no platform window/surface dependency enters the core.

## Next

OSM-09 should add the first real compute primitive:

**host-visible Vulkan buffer + command submission + tiny compute shader
round-trip**, validated against CPU output.

Only after that bridge is green should quantized GEMV work begin.
