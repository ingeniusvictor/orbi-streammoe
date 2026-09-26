#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import urllib.request

from tokenizers import Tokenizer

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
REVISION = "f5e99a3698d364cf77584543481b778afee26177"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")

CASES = [
    "Hello, world!",
    "¿Cómo estás?",
    "太阳能",
    "ORBI StreamMoE\nQwen3-Next",
    "<|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\n",
]

def download(root: pathlib.Path, name: str) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        target.write_bytes(response.read())
    return target

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.asset_dir)
    for name in FILES:
        download(root, name)

    actual = hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
    if actual != TOKENIZER_SHA256:
        raise SystemExit(
            f"tokenizer.json SHA256 mismatch: expected {TOKENIZER_SHA256}, got {actual}"
        )

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))

    vectors = []
    for text in CASES:
        for add_special_tokens in (False, True):
            encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
            ids = list(encoded.ids)
            vectors.append(
                {
                    "text": text,
                    "add_special_tokens": add_special_tokens,
                    "ids": ids,
                    "decode_keep_special": tokenizer.decode(
                        ids, skip_special_tokens=False
                    ),
                    "decode_skip_special": tokenizer.decode(
                        ids, skip_special_tokens=True
                    ),
                }
            )

    payload = {
        "model": MODEL,
        "revision": REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizers_python_version": __import__("tokenizers").__version__,
        "vectors": vectors,
    }
    pathlib.Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"OSM-37C reference vectors: {len(vectors)} cases")
    print(f"model={MODEL}")
    print(f"revision={REVISION}")
    print(f"tokenizer_sha256={TOKENIZER_SHA256}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
