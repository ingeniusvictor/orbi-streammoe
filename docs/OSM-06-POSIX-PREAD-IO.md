# OSM-06 — POSIX pread expert I/O

Status: **IMPLEMENTED / CI PENDING**

## Objective

Implement the storage primitive that maps naturally to Android NDK and other POSIX systems:

`pread(fd, destination, expertStride, expert * expertStride)`

This reproduces the central Swiftlet qpack expert-fetch idea without Swift, Foundation or Metal.

## Implementation

`PosixPreadExpertStorage` uses:

- one lazily opened file descriptor per qpack expert layer;
- explicit positional `pread`;
- one read operation per requested expert;
- concurrent reads for a miss batch;
- exact expert-stride transfer validation;
- the same portable `ExpertStorage` interface consumed by `ExpertCache`.

The current concurrency mechanism uses C++20 `std::jthread` workers as a correctness baseline. A persistent I/O pool can replace per-batch thread creation after profiling.

## Android relevance

Android's NDK exposes the POSIX file APIs this backend relies on. The core implementation intentionally has no desktop-Linux-specific dependency.

Passing this gate on Linux therefore validates the portable storage mechanism and API surface, but it is **not yet an Android device certification**. A later Android/NDK gate must cross-compile and run on ORBI Edge Mesh hardware.

## Symmetry with Windows

At this point the shared runtime has two native storage paths:

```text
ExpertCache
   │
   ├─ WindowsOverlappedExpertStorage
   │      └─ ReadFile + OVERLAPPED
   │
   └─ PosixPreadExpertStorage
          └─ pread
```

Both preserve the same fixed-stride qpack contract.

## Tests

The POSIX CI test:

1. creates a synthetic qpack;
2. requests several experts concurrently;
3. checks byte-for-byte results;
4. drives the backend through the portable expert cache;
5. validates repeat hits;
6. truncates a layer file and requires the short read to fail.

## Exit gate

OSM-06 closes when Ubuntu CI passes the POSIX storage test and Windows remains green on its own storage/backend tests.

The next major milestone is Vulkan compute bootstrap shared by Windows and Android.
