#!/usr/bin/env python3
import json
import pathlib
import stat
import tempfile

from certify_qwen_first_token import build_evidence, write_immutable

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def make_probe(path: pathlib.Path, *, layers: int = 48, multi: bool = False) -> None:
    script = f"""#!/usr/bin/env python3
import json
import sys
token = int(sys.argv[2])
print(json.dumps({{
  "stage": "official-first-token-inference",
  "executed": True,
  "checkpoint_dir": sys.argv[1],
  "input_token_id": token,
  "generated_token_id": 42,
  "generated_logit": 1.25,
  "model": {{"hidden_size": 2048, "vocab_size": 151936, "layer_count": {layers}}},
  "memory_limits": {{
    "lm_head_chunk_rows": int(sys.argv[3]),
    "host_cache_bytes": int(sys.argv[4]),
    "gpu_cache_bytes": int(sys.argv[5])
  }},
  "host_cache": {{"misses": 1}},
  "gpu_cache": {{"loads": 1}},
  "vulkan": {{"device_name": "fixture-gpu"}},
  "claims": {{
    "one_checkpoint_backed_step_executed": True,
    "streamed_embedding_executed": True,
    "decoder_48_layers_executed": True,
    "final_rmsnorm_executed": True,
    "streamed_lm_head_executed": True,
    "first_token_generated": True,
    "text_prompt_tokenized": False,
    "multi_token_generation_validated": {str(multi)},
    "tokens_per_second_measured": False
  }}
}}))
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-streammoe-osm43b-") as temp:
        root = pathlib.Path(temp)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        load_cert = root / "runtime-load-certification.json"
        write_json(
            load_cert,
            {
                "schema_version": 1,
                "stage": "official-full-checkpoint-runtime-load-certification",
                "source": {
                    "model": OFFICIAL_MODEL,
                    "snapshot": OFFICIAL_SNAPSHOT,
                    "execution_id": "fixture-execution",
                },
                "package": {
                    "checkpoint_dir": str(checkpoint),
                    "package_digest_sha256": "b" * 64,
                },
                "claims": {
                    "official_full_checkpoint_loaded": True,
                    "vulkan_runtime_initialized": True,
                    "decoder_stack_created": True,
                    "runtime_inference_validated": False,
                    "first_token_generated": False,
                },
            },
        )

        probe = root / "probe.py"
        make_probe(probe)
        evidence = build_evidence(
            load_cert, checkpoint, probe, 123, 1024, 65536, 65536
        )
        if not evidence["claims"]["official_checkpoint_first_token_generated"]:
            raise RuntimeError("first-token certification claim missing")
        if evidence["claims"]["multi_token_generation_validated"]:
            raise RuntimeError("OSM-43B crossed multi-token boundary")

        output = root / "first-token-certification.json"
        write_immutable(output, evidence)
        write_immutable(output, evidence)

        bad_layers = root / "bad-layers.py"
        make_probe(bad_layers, layers=47)
        expect_failure(
            lambda: build_evidence(
                load_cert, checkpoint, bad_layers, 123, 1024, 65536, 65536
            ),
            "wrong layer count",
        )

        bad_multi = root / "bad-multi.py"
        make_probe(bad_multi, multi=True)
        expect_failure(
            lambda: build_evidence(
                load_cert, checkpoint, bad_multi, 123, 1024, 65536, 65536
            ),
            "premature multi-token claim",
        )

        wrong_checkpoint = root / "wrong"
        wrong_checkpoint.mkdir()
        expect_failure(
            lambda: build_evidence(
                load_cert, wrong_checkpoint, probe, 123, 1024, 65536, 65536
            ),
            "checkpoint binding",
        )

        tampered = json.loads(load_cert.read_text(encoding="utf-8"))
        tampered["source"]["snapshot"] = "wrong"
        write_json(load_cert, tampered)
        expect_failure(
            lambda: build_evidence(
                load_cert, checkpoint, probe, 123, 1024, 65536, 65536
            ),
            "snapshot pin",
        )

        print(
            "OSM-43B official first-token inference gate: PASS\n"
            "  OSM43A_certification_required=PASS\n"
            "  official_model_snapshot_binding=PASS\n"
            "  explicit_memory_limits=PASS\n"
            "  shell_free_probe_execution=PASS\n"
            "  48_layer_first_token_claim=PASS\n"
            "  immutable_evidence=PASS\n"
            "  multi_token_claim_boundary=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
