#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
REVISION = "f5e99a3698d364cf77584543481b778afee26177"

CASES = [
    {
        "name": "user_only_generation",
        "messages": [{"role": "user", "content": "Hola, responde en una frase."}],
        "add_generation_prompt": True,
    },
    {
        "name": "system_user_generation",
        "messages": [
            {"role": "system", "content": "Eres un asistente técnico de ORBI."},
            {"role": "user", "content": "Explica qué es un inversor solar."},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "multiturn_generation",
        "messages": [
            {"role": "system", "content": "Responde con precisión."},
            {"role": "user", "content": "¿Qué es tensión?"},
            {"role": "assistant", "content": "Es la diferencia de potencial eléctrico."},
            {"role": "user", "content": "¿Y corriente?"},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "completed_assistant_turn",
        "messages": [
            {"role": "user", "content": "Di hola."},
            {"role": "assistant", "content": "Hola."},
        ],
        "add_generation_prompt": False,
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.asset_dir)
    config_path = root / "tokenizer_config.json"
    tokenizer_path = root / "tokenizer.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise SystemExit("OSM-44A requires tokenizer_config.json and tokenizer.json")

    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    chat_template = config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise SystemExit("official tokenizer_config.json has no non-empty chat_template")

    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
    )

    vectors = []
    for case in CASES:
        rendered = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=case["add_generation_prompt"],
        )
        ids = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=case["add_generation_prompt"],
        )
        if not isinstance(rendered, str) or not rendered:
            raise SystemExit(f"empty rendered chat template: {case['name']}")
        if not isinstance(ids, list) or not ids:
            raise SystemExit(f"empty chat token sequence: {case['name']}")
        direct_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if ids != direct_ids:
            raise SystemExit(f"render/tokenize disagreement: {case['name']}")

        vectors.append(
            {
                **case,
                "rendered": rendered,
                "rendered_sha256": sha256_bytes(rendered.encode("utf-8")),
                "token_ids": ids,
                "token_count": len(ids),
            }
        )

    payload = {
        "schema_version": 1,
        "stage": "official-qwen-chat-template-reference",
        "model": MODEL,
        "revision": REVISION,
        "tokenizer_json_sha256": sha256_bytes(tokenizer_path.read_bytes()),
        "tokenizer_config_sha256": sha256_bytes(config_bytes),
        "chat_template_sha256": sha256_bytes(chat_template.encode("utf-8")),
        "transformers_version": __import__("transformers").__version__,
        "vectors": vectors,
        "claims": {
            "official_chat_template_loaded": True,
            "system_role_covered": True,
            "user_role_covered": True,
            "assistant_role_covered": True,
            "generation_prompt_covered": True,
            "multiturn_history_covered": True,
            "native_runtime_chat_rendering_validated": False,
            "model_inference_executed": False,
        },
    }
    pathlib.Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("OSM-44A official chat-template reference: PASS")
    print(f"vectors={len(vectors)}")
    print(f"tokenizer_config_sha256={payload['tokenizer_config_sha256']}")
    print(f"chat_template_sha256={payload['chat_template_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
