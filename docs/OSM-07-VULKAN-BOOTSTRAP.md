# OSM-07 — Vulkan runtime bootstrap

Status: **IMPLEMENTED / CI PENDING**

## Objective

Introduce the shared GPU API intended for both Windows and Android without making the runtime depend on an installed Vulkan SDK at build time.

## Dependencies

Pinned compatibility set:

- KhronosGroup/Vulkan-Headers: `v1.4.363`
- zeux/volk: `vulkan-sdk-1.4.363`

volk dynamically loads the platform Vulkan loader at runtime. ORBI StreamMoE therefore does not directly link against `vulkan-1.dll` or `libvulkan.so`.

## Runtime probe

`probe_vulkan_runtime()`:

1. asks volk to locate the Vulkan loader;
2. determines the loader API version;
3. creates a headless Vulkan instance without window-system extensions;
4. enumerates physical devices;
5. records:
   - device name;
   - vendor/device IDs;
   - Vulkan API version;
   - device type;
   - first compute-capable queue family;
   - max compute workgroup invocations;
   - max compute shared memory;
   - max storage buffer range;
6. destroys the instance and unloads volk.

The probe is deliberately headless because inference requires compute, not presentation.

## CI semantics

A cloud CI runner may legitimately have no Vulkan loader or no physical GPU.

Therefore the CI gate validates:

- Vulkan/volk integration builds on Windows and Linux;
- runtime detection is graceful;
- absence of a loader/device is reported rather than crashing;
- any enumerated device has internally valid compute limits.

**CI success is not hardware certification.**

A later local hardware pilot must run this probe on:

- the Windows L.U.M.I.A. machine;
- POCO X7 Pro / ORBI Edge Mesh;
- Xiaomi 11T Pro where applicable.

## Why Vulkan

The same compute API is available on Windows and Android. Keeping the first accelerated backend on Vulkan avoids maintaining a DirectML implementation for Windows and a separate Vulkan implementation for Android.

## Next gate

OSM-08 will create a real Vulkan compute context:

- select a compute-capable physical device;
- create logical device + compute queue;
- allocate/bind a storage buffer;
- submit a minimal compute dispatch;
- read back and compare against the CPU oracle.

No Qwen kernel should be ported until that round-trip is certified.
