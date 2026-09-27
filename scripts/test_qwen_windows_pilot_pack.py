#!/usr/bin/env python3
import hashlib
import json
import pathlib
import stat
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orbi-osm46a-") as td:
        root = pathlib.Path(td)
        fake = root / "fake_phase.py"
        fake.write_text(
            """#!/usr/bin/env python3
import argparse, json, pathlib
p=argparse.ArgumentParser(); p.add_argument('--stage', required=True); p.add_argument('--output', required=True)
a=p.parse_args(); pathlib.Path(a.output).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.output).write_text(json.dumps({'stage':a.stage})+'\\n', encoding='utf-8')
""",
            encoding="utf-8",
        )
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

        inp = root / "input.json"; inp.write_text("{}\n", encoding="utf-8")
        evidence = root / "evidence"
        names = [
            ("windows_conversion_certification", "windows-full-conversion-certification"),
            ("runtime_load", "official-full-checkpoint-runtime-load-certification"),
            ("first_token", "official-first-token-inference-certification"),
            ("text_first_token", "official-text-prompt-first-token-certification"),
            ("bounded_generation", "official-bounded-multitoken-generation-certification"),
            ("chat_generation", "official-chat-bounded-generation-certification"),
            ("performance", "official-chat-performance-instrumentation"),
            ("phase_latency", "official-chat-phase-latency-certification"),
            ("cache_phase", "official-chat-cache-phase-attribution-certification"),
        ]
        phases=[]
        for i,(name,stage) in enumerate(names):
            out=evidence/f"{i}.json"
            phases.append({
                "name":name, "expected_stage":stage, "output":str(out),
                "argv":[sys.executable,str(fake),"--stage",stage,"--output",str(out)]
            })
        pack={
            "schema_version":1,
            "stage":"windows-official-pilot-execution-pack",
            "evidence_dir":str(evidence),
            "inputs":{"fixture":{"path":str(inp),"sha256":sha(inp)}},
            "phases":phases,
            "claims":{
                "real_windows_pilot_execution_planned":True,
                "shell_string_execution_used":False,
                "manual_intermediate_json_editing_required":False,
                "real_windows_pilot_executed":False,
                "performance_target_met":False,
            },
        }
        pack_path=root/"pack.json"; pack_path.write_text(json.dumps(pack)+"\n", encoding="utf-8")
        state=root/"state.json"
        runner=pathlib.Path(__file__).with_name("run_qwen_windows_pilot_pack.py")

        subprocess.run([sys.executable,str(runner),"--pack",str(pack_path),"--state",str(state),"--verify-only"],check=True)
        subprocess.run([sys.executable,str(runner),"--pack",str(pack_path),"--state",str(state)],check=True)
        result=json.loads(state.read_text(encoding="utf-8"))
        if not result.get("completed") or len(result.get("completed_phases",[])) != 9:
            raise RuntimeError("OSM-46A runner did not complete all phases")

        # Restart-safe replay must perform no destructive reset and retain completion.
        subprocess.run([sys.executable,str(runner),"--pack",str(pack_path),"--state",str(state)],check=True)
        result2=json.loads(state.read_text(encoding="utf-8"))
        if result2.get("completed_phases") != result.get("completed_phases"):
            raise RuntimeError("OSM-46A restart changed completed phase set")

        # Tamper rejection.
        inp.write_text('{"tampered":true}\n', encoding="utf-8")
        bad=subprocess.run([sys.executable,str(runner),"--pack",str(pack_path),"--state",str(state),"--verify-only"])
        if bad.returncode == 0:
            raise RuntimeError("OSM-46A accepted tampered bound input")

        print(
            "OSM-46A Windows pilot execution pack: PASS\n"
            "  argv_only_execution=PASS\n"
            "  ordered_phase_chain=PASS\n"
            "  restart_safe_resume=PASS\n"
            "  evidence_stage_validation=PASS\n"
            "  input_hash_binding=PASS\n"
            "  tamper_rejection=PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
