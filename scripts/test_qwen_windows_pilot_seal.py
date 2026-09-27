#!/usr/bin/env python3
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

OFFICIAL_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_SNAPSHOT = "f5e99a3698d364cf77584543481b778afee26177"

PHASES = (
    ("windows_conversion_certification", "windows-full-conversion-certification"),
    ("runtime_load", "official-full-checkpoint-runtime-load-certification"),
    ("first_token", "official-first-token-inference-certification"),
    ("text_first_token", "official-text-prompt-first-token-certification"),
    ("bounded_generation", "official-bounded-multitoken-generation-certification"),
    ("chat_generation", "official-chat-bounded-generation-certification"),
    ("performance", "official-chat-performance-instrumentation"),
    ("phase_latency", "official-chat-phase-latency-certification"),
    ("cache_phase", "official-chat-cache-phase-attribution-certification"),
)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm46b-") as td:
        root = pathlib.Path(td)
        checkpoint = root / "checkpoint"; checkpoint.mkdir()
        evidence = root / "evidence"
        digest = "a" * 64
        specs = []

        for idx, (name, stage) in enumerate(PHASES):
            path = evidence / f"{idx:02d}-{name}.json"
            payload = {
                "schema_version": 1,
                "stage": stage,
                "source": {"model": OFFICIAL_MODEL, "snapshot": OFFICIAL_SNAPSHOT},
                "package": {
                    "checkpoint_dir": str(checkpoint.resolve(strict=False)),
                    "package_digest_sha256": digest,
                },
            }
            if name == "windows_conversion_certification":
                payload["package"] = {
                    "output_dir": str(checkpoint.resolve(strict=False)),
                    "package_digest_sha256": digest,
                }
            elif name == "performance":
                payload["measurement"] = {
                    "wall_time_ns": 2_000_000_000,
                    "peak_sampled_process_rss_bytes": 123456,
                    "generated_token_count": 4,
                    "prompt_token_count": 7,
                    "cold_end_to_end_generated_tokens_per_second": 2.0,
                    "cold_end_to_end_ms_per_generated_token": 500.0,
                    "gpu_cache_resident_bytes": 1000,
                    "gpu_cache_budget_bytes": 2000,
                    "host_cache_budget_bytes": 3000,
                }
            elif name == "phase_latency":
                payload["phase_latency"] = {
                    "prompt_prefill_ns": 700,
                    "decode_ns": 300,
                    "average_prompt_step_ns": 100.0,
                    "average_decode_step_ns": 100.0,
                }
            elif name == "cache_phase":
                payload["prefill"] = {"host_hits": 1, "host_misses": 2, "gpu_hits": 3, "gpu_misses": 4, "gpu_loads": 4, "gpu_evictions": 1}
                payload["decode"] = {"host_hits": 5, "host_misses": 6, "gpu_hits": 7, "gpu_misses": 8, "gpu_loads": 8, "gpu_evictions": 2}
                payload["totals"] = {"host_hits": 6, "host_misses": 8, "gpu_hits": 10, "gpu_misses": 12, "gpu_loads": 12, "gpu_evictions": 3}
            write(path, payload)
            specs.append({"name": name, "expected_stage": stage, "output": str(path), "argv": [sys.executable, "fixture.py"]})

        pack = {
            "schema_version": 1,
            "stage": "windows-official-pilot-execution-pack",
            "source": {"model": OFFICIAL_MODEL, "snapshot": OFFICIAL_SNAPSHOT},
            "pack_id": "fixture-pack",
            "checkpoint_dir": str(checkpoint),
            "evidence_dir": str(evidence),
            "phases": specs,
            "claims": {
                "real_windows_pilot_execution_planned": True,
                "shell_string_execution_used": False,
                "manual_intermediate_json_editing_required": False,
                "real_windows_pilot_executed": False,
                "performance_target_met": False,
            },
        }
        pack_path = root / "pack.json"; write(pack_path, pack)

        state = {
            "schema_version": 1,
            "stage": "windows-official-pilot-run-state",
            "pack_path": str(pack_path.resolve(strict=False)),
            "pack_sha256": sha(pack_path),
            "completed_phases": [name for name, _ in PHASES],
            "completed": True,
            "final_evidence": specs[-1]["output"],
            "claims": {
                "all_planned_phases_completed": True,
                "real_windows_pilot_executed": True,
                "performance_target_met": False,
            },
        }
        state_path = root / "state.json"; write(state_path, state)
        seal_path = root / "seal.json"
        cert = pathlib.Path(__file__).with_name("certify_qwen_windows_pilot.py")

        subprocess.run([sys.executable, str(cert), "--pack", str(pack_path), "--state", str(state_path), "--output", str(seal_path)], check=True)
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if seal.get("stage") != "windows-official-pilot-evidence-seal":
            raise RuntimeError("OSM-46B seal stage mismatch")
        if seal.get("claims", {}).get("all_nine_evidence_phases_sealed") is not True:
            raise RuntimeError("OSM-46B evidence chain claim missing")
        if len(seal.get("evidence_chain", [])) != 9:
            raise RuntimeError("OSM-46B did not seal nine phases")
        if seal.get("claims", {}).get("performance_target_met") is not False:
            raise RuntimeError("OSM-46B crossed performance claim boundary")

        # Idempotence.
        subprocess.run([sys.executable, str(cert), "--pack", str(pack_path), "--state", str(state_path), "--output", str(seal_path)], check=True)

        # Tampering in any sealed evidence must be detected by regenerated seal conflict.
        tampered = pathlib.Path(specs[4]["output"])
        value = json.loads(tampered.read_text(encoding="utf-8"))
        value["tampered"] = True
        write(tampered, value)
        bad = subprocess.run([sys.executable, str(cert), "--pack", str(pack_path), "--state", str(state_path), "--output", str(seal_path)])
        if bad.returncode == 0:
            raise RuntimeError("OSM-46B accepted changed evidence against immutable seal")

        # Incomplete state must be rejected.
        value = json.loads(state_path.read_text(encoding="utf-8"))
        value["completed"] = False
        write(state_path, value)
        incomplete = subprocess.run([sys.executable, str(cert), "--pack", str(pack_path), "--state", str(state_path), "--output", str(root / "other.json")])
        if incomplete.returncode == 0:
            raise RuntimeError("OSM-46B accepted incomplete pilot state")

        print(
            "OSM-46B Windows pilot evidence seal: PASS\n"
            "  complete_state_required=PASS\n"
            "  nine_phase_chain_sealed=PASS\n"
            "  model_snapshot_binding=PASS\n"
            "  package_digest_binding=PASS\n"
            "  metrics_summary=PASS\n"
            "  immutable_replay=PASS\n"
            "  tamper_rejection=PASS\n"
            "  claim_boundary=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
