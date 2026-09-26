# OSM-37B — tokenizers-cpp Qwen adapter

Status: **IMPLEMENTED / OPTIONAL BACKEND CI CERTIFICATION PENDING**

## Goal

Bind the OSM-36B portable tokenizer interface to a real Hugging Face
`tokenizer.json` execution engine while keeping Rust and tokenizer-specific
dependencies out of the default numerical runtime build.

## Backend

OSM-37B uses `mlc-ai/tokenizers-cpp` pinned to:

```text
c586c52f93f7b060753bd2388eb96a105cb7374d
```

The upstream project is Apache-2.0 licensed and supports Windows, Linux and
Android. Its build uses the Hugging Face Rust `tokenizers` engine.

## Why the C API

The public `tokenizers_cpp::Tokenizer` virtual interface exposes only the
default `Encode(text)` / `Decode(ids)` calls. The underlying C API exposes
the exact controls required by the ORBI tokenizer contract:

- `add_special_tokens` on encode;
- `skip_special_tokens` on decode.

OSM-37B therefore binds the upstream C API directly rather than weakening the
OSM-36B semantics.

## Build isolation

The backend is disabled by default:

```bash
-DORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP=OFF
```

Enable it explicitly with:

```bash
-DORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP=ON
```

When enabled, CMake fetches the pinned upstream source and links its tokenizer
engine into `orbi_streammoe_core`.

The default Windows/Linux/Vulkan CI remains independent of Rust. A dedicated
tokenizer integration workflow builds the optional backend on Windows and
Ubuntu.

## Runtime contract

`QwenTokenizersCppTokenizer::create(...)` first executes the OSM-37A
`QwenTokenizerAssetContract` validation. It then:

1. loads `tokenizer.json`;
2. creates the Hugging Face tokenizer engine;
3. rechecks EOS and PAD token-to-ID mappings through the concrete engine;
4. exposes the OSM-36B `Tokenizer` interface.

The reported tokenizer vocabulary remains the **token-ID domain upper bound**
from OSM-37A, not merely the number of entries returned by an engine. This
preserves padded/sparse vocabulary correctness.

## Exit gate

OSM-37B is GREEN when:

- default CI still passes with the backend disabled;
- optional Windows build loads a tokenizer.json fixture;
- optional Ubuntu build loads the same fixture;
- encode/decode round-trip is exact;
- special-token encoding is exact;
- skip-special-token decoding is exact;
- EOS mapping agrees with OSM-37A;
- the dependency remains pinned and isolated from the numerical core.

## Android

The chosen upstream has explicit Android CMake/cargo target handling. Android
packaging is intentionally deferred until the desktop adapter contract is
certified.

## Next

**OSM-37C — official Qwen tokenizer parity fixture**

Run the concrete adapter against authoritative Qwen tokenizer assets and certify
known prompt/token-ID vectors before wiring it into a real Qwen3-Next checkpoint
pilot.
