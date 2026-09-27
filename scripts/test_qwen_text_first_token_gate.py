#!/usr/bin/env python3
import hashlib
import json
import pathlib
import stat
import tempfile

import certify_qwen_text_first_token as gate

def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")

def make_probe(path: pathlib.Path, *, multi: bool = False) -> None:
    path.write_text(f"""#!/usr/bin/env python3
import json, sys
prompt = open(sys.argv[3], "rb").read()
print(json.dumps({{
  "stage": "official-text-prompt-first-token",
  "executed": True,
  "prompt_utf8_bytes": len(prompt),
  "prompt_token_count": 3,
  "prompt_token_ids": [1, 2, 3],
  "generated_token_id": 42,
  "generated_logit": 1.5,
  "generated_text": " fixture",
  "model_steps": 3,
  "model": {{"hidden_size": 2048, "vocab_size": 151936, "layer_count": 48}},
  "limits": {{
    "max_prompt_tokens": int(sys.argv[4]),
    "lm_head_chunk_rows": int(sys.argv[5]),
    "host_cache_bytes": int(sys.argv[6]),
    "gpu_cache_bytes": int(sys.argv[7])
  }},
  "claims": {{
    "official_tokenizer_used": True,
    "text_prompt_tokenized": True,
    "teacher_forced_prompt_executed": True,
    "checkpoint_backed_48_layer_inference_executed": True,
    "first_continuation_token_generated": True,
    "multi_token_generation_validated": {str(multi)},
    "tokens_per_second_measured": False
  }}
}}))
""", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)

def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm43c-") as temp:
        root = pathlib.Path(temp)
        checkpoint = root / "checkpoint"; checkpoint.mkdir()
        assets = root / "tokenizer"; assets.mkdir()
        tokenizer = assets / "tokenizer.json"
        tokenizer.write_bytes(b"fixture")
        original_sha = gate.OFFICIAL_TOKENIZER_SHA256
        gate.OFFICIAL_TOKENIZER_SHA256 = hashlib.sha256(b"fixture").hexdigest()

        prompt = root / "prompt.txt"
        prompt.write_text("Hola ORBI", encoding="utf-8")
        parent = root / "first-token.json"
        write_json(parent, {
            "schema_version": 1,
            "stage": "official-first-token-inference-certification",
            "source": {"model": gate.OFFICIAL_MODEL, "snapshot": gate.OFFICIAL_SNAPSHOT},
            "package": {"checkpoint_dir": str(checkpoint), "package_digest_sha256": "c" * 64},
            "claims": {
                "official_checkpoint_first_token_generated": True,
                "multi_token_generation_validated": False,
            },
        })
        probe = root / "probe.py"; make_probe(probe)

        evidence = gate.build_evidence(
            parent, checkpoint, assets, prompt, probe, 8, 1024, 65536, 65536
        )
        if not evidence["claims"]["official_first_continuation_token_generated"]:
            raise RuntimeError("OSM-43C continuation-token claim missing")
        if evidence["claims"]["multi_token_generation_validated"]:
            raise RuntimeError("OSM-43C crossed multi-token boundary")

        output = root / "cert.json"
        gate.write_immutable(output, evidence)
        gate.write_immutable(output, evidence)

        bad = root / "bad.py"; make_probe(bad, multi=True)
        expect_failure(
            lambda: gate.build_evidence(parent, checkpoint, assets, prompt, bad, 8, 1024, 65536, 65536),
            "premature multi-token claim",
        )
        expect_failure(
            lambda: gate.build_evidence(parent, checkpoint, assets, prompt, probe, 2, 1024, 65536, 65536),
            "prompt bound",
        )

        tampered = json.loads(parent.read_text(encoding="utf-8"))
        tampered["source"]["snapshot"] = "wrong"
        write_json(parent, tampered)
        expect_failure(
            lambda: gate.build_evidence(parent, checkpoint, assets, prompt, probe, 8, 1024, 65536, 65536),
            "snapshot pin",
        )
        gate.OFFICIAL_TOKENIZER_SHA256 = original_sha

        print(
            "OSM-43C official text-prompt first-token gate: PASS\n"
            "  OSM43B_parent_required=PASS\n"
            "  official_tokenizer_digest_binding=PASS\n"
            "  bounded_prompt_contract=PASS\n"
            "  teacher_forcing_accounting=PASS\n"
            "  immutable_evidence=PASS\n"
            "  multi_token_claim_boundary=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
