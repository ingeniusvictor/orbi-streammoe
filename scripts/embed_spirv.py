#!/usr/bin/env python3
import argparse
import pathlib
import struct


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed SPIR-V words as a C++ header")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--namespace", default="orbi::streammoe::generated")
    args = parser.parse_args()

    data = args.input.read_bytes()
    if len(data) == 0 or len(data) % 4 != 0:
        raise SystemExit("SPIR-V byte length must be non-zero and divisible by 4")

    words = struct.unpack("<" + "I" * (len(data) // 4), data)

    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        f"namespace {args.namespace} {{",
        "",
        f"inline constexpr std::array<std::uint32_t, {len(words)}> {args.symbol}{{{{",
    ]

    chunk = 8
    for i in range(0, len(words), chunk):
        values = ", ".join(f"0x{word:08x}U" for word in words[i:i + chunk])
        lines.append(f"    {values},")

    lines.extend([
        "}};",
        "",
        f"}}  // namespace {args.namespace}",
        "",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
