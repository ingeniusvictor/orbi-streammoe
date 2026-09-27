#!/usr/bin/env python3
import json
import pathlib
import stat
import tempfile

from certify_qwen_runtime_load import build_evidence, write_immutable

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_failure(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise RuntimeError(f"expected failure: {label}")


def make_probe(path: pathlib.Path, *, layers: int = 48, token: bool = False) -> None:
    script = f"""#!/usr/bin/env python3
import json
import sys
print(json.dumps({{
  "stage": "official-full-checkpoint-runtime-load",
  "loaded": True,
  "checkpoint_dir": sys.argv[1],
  "model_name": "qwen3_next",
  "source_checkpoint": "{OFFICIAL_MODEL}",
  "hidden_size": 2048,
  "vocab_size": 151936,
  "layer_count": {layers},
  "vulkan": {{
    "device_name": "fixture-gpu",
    "vendor_id": 1,
    "device_id": 2,
    "api_version": 3,
    "queue_family_index": 0
  }},
  "claims": {{
    "checkpoint_opened": True,
    "dense_inventory_bound": True,
    "decoder_stack_created": True,
    "vulkan_context_created": True,
    "inference_executed": False,
    "token_generated": {str(token)}
  }}
}}))
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-streammoe-osm43a-") as temp:
        root = pathlib.Path(temp)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()

        certification = root / "windows-conversion-certification.json"
        write_json(
            certification,
            {
                "schema_version": 1,
                "stage": "windows-full-conversion-certification",
                "certification_id": "fixture-cert",
                "source": {
                    "model": OFFICIAL_MODEL,
                    "snapshot": OFFICIAL_SNAPSHOT,
                    "execution_id": "fixture-execution",
                },
                "package": {
                    "output_dir": str(checkpoint),
                    "package_digest_sha256": "a" * 64,
                    "declared_file_count": 8,
                    "declared_file_bytes": 123,
                },
                "claims": {
                    "full_official_checkpoint_conversion_completed": True,
                    "full_checkpoint_package_deep_audited": True,
                    "runtime_inference_validated": False,
                    "tokens_per_second_measured": False,
                    "windows_hardware_inference_certified": False,
                },
            },
        )

        probe = root / "probe.py"
        make_probe(probe)
        evidence = build_evidence(certification, checkpoint, probe)
        if not evidence["claims"]["official_full_checkpoint_loaded"]:
            raise RuntimeError("official load claim missing")
        if evidence["claims"]["runtime_inference_validated"]:
            raise RuntimeError("OSM-43A crossed inference claim boundary")

        output = root / "runtime-load-certification.json"
        write_immutable(output, evidence)
        write_immutable(output, evidence)

        wrong_checkpoint = root / "wrong"
        wrong_checkpoint.mkdir()
        expect_failure(
            lambda: build_evidence(certification, wrong_checkpoint, probe),
            "checkpoint path mismatch",
        )

        bad_layers = root / "bad-layers.py"
        make_probe(bad_layers, layers=47)
        expect_failure(
            lambda: build_evidence(certification, checkpoint, bad_layers),
            "wrong official layer count",
        )

        bad_claim = root / "bad-claim.py"
        make_probe(bad_claim, token=True)
        expect_failure(
            lambda: build_evidence(certification, checkpoint, bad_claim),
            "premature token claim",
        )

        tampered = json.loads(certification.read_text(encoding="utf-8"))
        tampered["source"]["snapshot"] = "wrong"
        write_json(certification, tampered)
        expect_failure(
            lambda: build_evidence(certification, checkpoint, probe),
            "snapshot pin mismatch",
        )

        print(
            "OSM-43A official full-checkpoint runtime load gate: PASS\n"
            "  OSM42D_certification_required=PASS\n"
            "  checkpoint_path_binding=PASS\n"
            "  shell_free_probe_execution=PASS\n"
            "  official_48_layer_gate=PASS\n"
            "  Vulkan_load_claim=PASS\n"
            "  immutable_evidence=PASS\n"
            "  inference_claim_boundary=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
