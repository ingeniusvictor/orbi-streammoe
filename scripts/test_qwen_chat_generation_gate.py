#!/usr/bin/env python3
import hashlib
import json
import pathlib
import stat
import tempfile

import certify_qwen_chat_generation as gate


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def make_probe(path: pathlib.Path, *, generated: int = 3, tools: bool = False) -> None:
    ids = list(range(50, 50 + generated))
    logits = [1.0 + i / 10.0 for i in range(generated)]
    path.write_text(f"""#!/usr/bin/env python3
import json, sys
ids = {ids!r}
logits = {logits!r}
print(json.dumps({{
  "stage": "official-chat-bounded-generation",
  "executed": True,
  "message_count": 2,
  "rendered_prompt": "<|im_start|>user\\nHola<|im_end|>\\n<|im_start|>assistant\\n",
  "prompt_token_count": 4,
  "prompt_token_ids": [1, 2, 3, 4],
  "generated_token_count": len(ids),
  "generated_token_ids": ids,
  "generated_logits": logits,
  "generated_text": " fixture",
  "model_steps": 4 + len(ids) - 1,
  "model": {{"hidden_size": 2048, "vocab_size": 151936, "layer_count": 48}},
  "claims": {{
    "native_chat_renderer_used": True,
    "official_tokenizer_used": True,
    "chat_formatted_prompt_executed": True,
    "checkpoint_backed_48_layer_inference_executed": True,
    "bounded_multitoken_chat_generation_validated": True,
    "tool_calling_validated": {str(tools)},
    "tokens_per_second_measured": False,
    "ram_vram_profile_measured": False
  }}
}}))
""", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm44c-") as temp:
        root = pathlib.Path(temp)
        checkpoint = root / "checkpoint"; checkpoint.mkdir()
        assets = root / "tokenizer"; assets.mkdir()
        tokenizer_json = assets / "tokenizer.json"
        tokenizer_config = assets / "tokenizer_config.json"
        tokenizer_json.write_bytes(b"fixture-tokenizer")
        tokenizer_config.write_bytes(b'{"chat_template":"fixture-template"}')

        original_sha = gate.OFFICIAL_TOKENIZER_SHA256
        gate.OFFICIAL_TOKENIZER_SHA256 = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()

        parent = root / "bounded.json"
        write_json(parent, {
            "schema_version": 1,
            "stage": "official-bounded-multitoken-generation-certification",
            "source": {"model": gate.OFFICIAL_MODEL, "snapshot": gate.OFFICIAL_SNAPSHOT},
            "package": {"checkpoint_dir": str(checkpoint), "package_digest_sha256": "e" * 64},
            "claims": {
                "official_bounded_multitoken_generation_validated": True,
                "chat_template_validated": False,
            },
        })

        chat_ref = root / "chat-template.json"
        config_hash = hashlib.sha256(tokenizer_config.read_bytes()).hexdigest()
        template_hash = hashlib.sha256(b"fixture-template").hexdigest()
        write_json(chat_ref, {
            "schema_version": 1,
            "stage": "official-qwen-chat-template-reference",
            "model": gate.OFFICIAL_MODEL,
            "revision": gate.OFFICIAL_SNAPSHOT,
            "tokenizer_json_sha256": gate.OFFICIAL_TOKENIZER_SHA256,
            "tokenizer_config_sha256": config_hash,
            "chat_template_sha256": template_hash,
        })

        messages = root / "messages.json"
        write_json(messages, {
            "messages": [
                {"role": "system", "content": "Eres ORBI."},
                {"role": "user", "content": "Hola"},
            ]
        })

        probe = root / "probe.py"; make_probe(probe)
        evidence = gate.build_evidence(
            parent, chat_ref, checkpoint, assets, messages, probe,
            8, 32, 3, 1024, 65536, 65536
        )
        if not evidence["claims"]["native_chat_formatted_generation_validated"]:
            raise RuntimeError("OSM-44C chat-generation claim missing")

        output = root / "cert.json"
        gate.write_immutable(output, evidence)
        gate.write_immutable(output, evidence)

        one = root / "one.py"; make_probe(one, generated=1)
        expect_failure(
            lambda: gate.build_evidence(
                parent, chat_ref, checkpoint, assets, messages, one,
                8, 32, 3, 1024, 65536, 65536
            ),
            "single generated token",
        )

        tool_claim = root / "tool.py"; make_probe(tool_claim, tools=True)
        expect_failure(
            lambda: gate.build_evidence(
                parent, chat_ref, checkpoint, assets, messages, tool_claim,
                8, 32, 3, 1024, 65536, 65536
            ),
            "premature tool calling claim",
        )

        tampered = json.loads(chat_ref.read_text(encoding="utf-8"))
        tampered["chat_template_sha256"] = "0" * 64
        write_json(chat_ref, tampered)
        expect_failure(
            lambda: gate.build_evidence(
                parent, chat_ref, checkpoint, assets, messages, probe,
                8, 32, 3, 1024, 65536, 65536
            ),
            "chat template digest",
        )

        tampered["chat_template_sha256"] = template_hash
        tampered["revision"] = "wrong"
        write_json(chat_ref, tampered)
        expect_failure(
            lambda: gate.build_evidence(
                parent, chat_ref, checkpoint, assets, messages, probe,
                8, 32, 3, 1024, 65536, 65536
            ),
            "chat reference snapshot pin",
        )

        gate.OFFICIAL_TOKENIZER_SHA256 = original_sha

        print(
            "OSM-44C chat-formatted bounded generation gate: PASS\n"
            "  OSM43D_parent_required=PASS\n"
            "  OSM44A_reference_required=PASS\n"
            "  native_chat_renderer_claim=PASS\n"
            "  bounded_generation_contract=PASS\n"
            "  autoregressive_step_accounting=PASS\n"
            "  immutable_evidence=PASS\n"
            "  tool_calling_claim_boundary=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
