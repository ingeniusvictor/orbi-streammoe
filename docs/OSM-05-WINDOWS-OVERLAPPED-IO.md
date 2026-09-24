# OSM-05 — Windows overlapped expert I/O

Status: **IMPLEMENTED / CI PENDING**

## Objective

Replace the portable correctness reader with a Windows-native positional I/O backend while preserving the `ExpertStorage` contract used by the shared cache.

This is the first direct proof that the key Swiftlet expert-streaming mechanism can be reproduced without Apple APIs.

## Implementation

`WindowsOverlappedExpertStorage` uses:

- `CreateFileW`;
- `FILE_FLAG_OVERLAPPED`;
- `FILE_FLAG_RANDOM_ACCESS`;
- one lazily opened handle per qpack expert layer;
- explicit 64-bit byte offsets;
- one `OVERLAPPED` structure + event per requested expert;
- all reads issued before waiting for completion;
- `GetOverlappedResult` to verify each transfer completed with exactly one expert stride.

The offset contract remains:

```text
offset = expert_id * expertStride
```

## Intentional non-goals

This gate does **not** enable `FILE_FLAG_NO_BUFFERING`.

Buffered overlapped I/O is the correctness-first baseline. Direct/no-buffering behavior has stricter alignment rules and should only be enabled after benchmark evidence shows a benefit on the target SSD.

This gate also does not implement Vulkan yet.

## Cache integration

The backend implements the same platform-neutral `ExpertStorage` interface already used by OSM-04:

```text
ExpertCache
   │
   ▼
ExpertStorage
   │
   ▼
WindowsOverlappedExpertStorage
   │
   ▼
qpack layer_XX.bin
```

No cache-policy code is Windows-specific.

## Tests

The Windows-only CI test:

1. creates a synthetic qpack;
2. issues a batch containing multiple experts, including two experts from one layer;
3. verifies byte-for-byte expert contents;
4. runs the backend through the portable `ExpertCache`;
5. verifies hits/misses and cached bytes;
6. truncates a layer and verifies that a short overlapped read fails.

Linux continues to build and test the portable runtime but does not compile the Win32 backend.

## Exit gate

OSM-05 closes when:

- all existing portable tests remain green on Ubuntu;
- all portable tests remain green on Windows;
- the Windows-only overlapped I/O test passes.

After this gate, the next compute milestone is Vulkan bootstrap; the Android/POSIX positional I/O backend can reuse the same storage contract.
