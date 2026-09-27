#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
REVISION = "f5e99a3698d364cf77584543481b778afee26177"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.asset_dir)
    reference = json.loads(pathlib.Path(args.reference).read_text(encoding="utf-8"))

    if reference.get("schema_version") != 1:
        raise RuntimeError("unsupported OSM-44A schema")
    if reference.get("stage") != "official-qwen-chat-template-reference":
        raise RuntimeError("wrong OSM-44A stage")
    if reference.get("model") != MODEL or reference.get("revision") != REVISION:
        raise RuntimeError("official model/revision pin mismatch")

    tokenizer_json = (root / "tokenizer.json").read_bytes()
    tokenizer_config = (root / "tokenizer_config.json").read_bytes()
    config = json.loads(tokenizer_config.decode("utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise RuntimeError("official chat_template missing")

    if reference.get("tokenizer_json_sha256") != sha256_bytes(tokenizer_json):
        raise RuntimeError("tokenizer.json digest mismatch")
    if reference.get("tokenizer_config_sha256") != sha256_bytes(tokenizer_config):
        raise RuntimeError("tokenizer_config.json digest mismatch")
    if reference.get("chat_template_sha256") != sha256_bytes(template.encode("utf-8")):
        raise RuntimeError("chat_template digest mismatch")

    vectors = reference.get("vectors")
    if not isinstance(vectors, list) or len(vectors) < 4:
        raise RuntimeError("insufficient chat-template vectors")

    names = {vector.get("name") for vector in vectors}
    required_names = {
        "user_only_generation",
        "system_user_generation",
        "multiturn_generation",
        "completed_assistant_turn",
    }
    if not required_names.issubset(names):
        raise RuntimeError("required chat-template cases missing")

    roles_seen = set()
    generation_cases = 0
    for vector in vectors:
        rendered = vector.get("rendered")
        ids = vector.get("token_ids")
        messages = vector.get("messages")
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("empty rendered chat")
        if vector.get("rendered_sha256") != sha256_bytes(rendered.encode("utf-8")):
            raise RuntimeError("rendered chat digest mismatch")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, int) or x < 0 for x in ids):
            raise RuntimeError("invalid chat token IDs")
        if vector.get("token_count") != len(ids):
            raise RuntimeError("chat token count mismatch")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("chat messages missing")
        for message in messages:
            roles_seen.add(message.get("role"))
        if vector.get("add_generation_prompt") is True:
            generation_cases += 1

    if not {"system", "user", "assistant"}.issubset(roles_seen):
        raise RuntimeError("system/user/assistant coverage incomplete")
    if generation_cases < 3:
        raise RuntimeError("insufficient generation-prompt coverage")

    claims = reference.get("claims", {})
    for key in (
        "official_chat_template_loaded",
        "system_role_covered",
        "user_role_covered",
        "assistant_role_covered",
        "generation_prompt_covered",
        "multiturn_history_covered",
    ):
        if claims.get(key) is not True:
            raise RuntimeError(f"required OSM-44A claim missing: {key}")
    for key in ("native_runtime_chat_rendering_validated", "model_inference_executed"):
        if claims.get(key) is not False:
            raise RuntimeError(f"OSM-44A crossed claim boundary: {key}")

    print(
        "OSM-44A official chat-template contract: PASS\n"
        "  official_model_revision=PASS\n"
        "  tokenizer_config_digest=PASS\n"
        "  chat_template_digest=PASS\n"
        "  role_coverage=PASS\n"
        "  generation_prompt_coverage=PASS\n"
        "  multiturn_coverage=PASS\n"
        "  claim_boundary=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
