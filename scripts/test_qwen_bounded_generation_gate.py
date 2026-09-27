#!/usr/bin/env python3
import hashlib
import json
import pathlib
import stat
import tempfile

import certify_qwen_bounded_generation as gate

def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")

def make_probe(path: pathlib.Path, *, generated: int = 3, perf_claim: bool = False) -> None:
    ids = list(range(40, 40 + generated))
    logits = [1.0 + i / 10.0 for i in range(generated)]
    path.write_text(f"""#!/usr/bin/env python3
import json, sys
max_new = int(sys.argv[5])
ids = {ids!r}
logits = {logits!r}
print(json.dumps({{
  "stage": "official-bounded-multitoken-generation",
  "executed": True,
  "prompt_token_count": 3,
  "prompt_token_ids": [1, 2, 3],
  "generated_token_count": len(ids),
  "generated_token_ids": ids,
  "generated_logits": logits,
  "generated_text": " fixture",
  "model_steps": 3 + len(ids) - 1,
  "stop_reason": "max_new_tokens",
  "model": {{"hidden_size": 2048, "vocab_size": 151936, "layer_count": 48}},
  "limits": {{"max_prompt_tokens": int(sys.argv[4]), "max_new_tokens": max_new}},
  "claims": {{
    "official_tokenizer_used": True,
    "text_prompt_tokenized": True,
    "teacher_forced_prompt_executed": True,
    "checkpoint_backed_48_layer_inference_executed": True,
    "bounded_multitoken_generation_validated": True,
    "generated_token_sequence_decoded": True,
    "chat_template_validated": False,
    "tokens_per_second_measured": {str(perf_claim)},
    "ram_vram_profile_measured": False
  }}
}}))
""", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)

def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm43d-") as temp:
        root = pathlib.Path(temp)
        checkpoint = root / "checkpoint"; checkpoint.mkdir()
        assets = root / "tokenizer"; assets.mkdir()
        tokenizer = assets / "tokenizer.json"; tokenizer.write_bytes(b"fixture")
        original_sha = gate.OFFICIAL_TOKENIZER_SHA256
        gate.OFFICIAL_TOKENIZER_SHA256 = hashlib.sha256(b"fixture").hexdigest()
        prompt = root / "prompt.txt"; prompt.write_text("Hola ORBI", encoding="utf-8")
        parent = root / "text-first-token.json"
        write_json(parent, {
            "schema_version": 1,
            "stage": "official-text-prompt-first-token-certification",
            "source": {"model": gate.OFFICIAL_MODEL, "snapshot": gate.OFFICIAL_SNAPSHOT},
            "package": {"checkpoint_dir": str(checkpoint), "package_digest_sha256": "d" * 64},
            "claims": {
                "official_first_continuation_token_generated": True,
                "multi_token_generation_validated": False,
            },
        })

        probe = root / "probe.py"; make_probe(probe)
        evidence = gate.build_evidence(
            parent, checkpoint, assets, prompt, probe, 8, 3, 1024, 65536, 65536
        )
        if not evidence["claims"]["official_bounded_multitoken_generation_validated"]:
            raise RuntimeError("OSM-43D bounded generation claim missing")
        output = root / "cert.json"
        gate.write_immutable(output, evidence)
        gate.write_immutable(output, evidence)

        one = root / "one.py"; make_probe(one, generated=1)
        expect_failure(
            lambda: gate.build_evidence(parent, checkpoint, assets, prompt, one, 8, 3, 1024, 65536, 65536),
            "single-token output",
        )
        perf = root / "perf.py"; make_probe(perf, perf_claim=True)
        expect_failure(
            lambda: gate.build_evidence(parent, checkpoint, assets, prompt, perf, 8, 3, 1024, 65536, 65536),
            "premature performance claim",
        )
        expect_failure(
            lambda: gate.build_evidence(parent, checkpoint, assets, prompt, probe, 8, 9, 1024, 65536, 65536),
            "max_new_tokens upper bound",
        )

        tampered = json.loads(parent.read_text(encoding="utf-8"))
        tampered["source"]["snapshot"] = "wrong"
        write_json(parent, tampered)
        expect_failure(
            lambda: gate.build_evidence(parent, checkpoint, assets, prompt, probe, 8, 3, 1024, 65536, 65536),
            "snapshot pin",
        )
        gate.OFFICIAL_TOKENIZER_SHA256 = original_sha

        print(
            "OSM-43D bounded multi-token generation gate: PASS\n"
            "  OSM43C_parent_required=PASS\n"
            "  official_tokenizer_digest_binding=PASS\n"
            "  generation_bound_2_to_8=PASS\n"
            "  autoregressive_step_accounting=PASS\n"
            "  immutable_evidence=PASS\n"
            "  performance_claim_boundary=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
