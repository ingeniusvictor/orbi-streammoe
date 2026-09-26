#!/usr/bin/env python3
import argparse
import pathlib
import urllib.request

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"
FILES = ("config.json", "model.safetensors.index.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        url = (
            f"https://huggingface.co/{MODEL}/resolve/{SNAPSHOT}/{name}"
            "?download=true"
        )
        destination = root / name
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "orbi-streammoe-osm38b/1"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if not data:
            raise RuntimeError(f"empty download: {name}")
        destination.write_bytes(data)
        print(f"{name}: {len(data)} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
