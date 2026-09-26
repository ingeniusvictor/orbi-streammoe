#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile


def run(script, *args):
    process = subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "scripts" / "plan_qwen_missing_experts.py"

    with tempfile.TemporaryDirectory() as tmp:
        journal = pathlib.Path(tmp) / "layer.progress.json"

        fresh = run(
            script,
            "--journal", str(journal),
            "--first-expert", "0",
            "--end-expert", "4",
            "--expert-count", "8",
            "--layer", "0",
        )
        assert fresh["missing_experts"] == [0, 1, 2, 3]
        assert fresh["already_complete"] == []
        assert fresh["journal_present"] is False

        journal.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "layer_index": 0,
                    "expert_count": 8,
                    "expert_stride": 16,
                    "completed": [
                        {"expert": 1, "fnv1a64": "1111111111111111"},
                        {"expert": 3, "fnv1a64": "3333333333333333"},
                        {"expert": 6, "fnv1a64": "6666666666666666"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        resumed = run(
            script,
            "--journal", str(journal),
            "--first-expert", "0",
            "--end-expert", "4",
            "--expert-count", "8",
            "--layer", "0",
        )
        assert resumed["missing_experts"] == [0, 2]
        assert resumed["already_complete"] == [1, 3]
        assert resumed["missing_experts_csv"] == "0,2"
        assert resumed["journal_source"] == str(journal)

        backup = pathlib.Path(str(journal) + ".bak")
        journal.rename(backup)
        recovered = run(
            script,
            "--journal", str(journal),
            "--first-expert", "0",
            "--end-expert", "4",
            "--expert-count", "8",
            "--layer", "0",
        )
        assert recovered["missing_experts"] == [0, 2]
        assert recovered["already_complete"] == [1, 3]
        assert recovered["journal_source"] == str(backup)

        invalid = json.loads(backup.read_text(encoding="utf-8"))
        invalid["completed"].append(
            {"expert": 8, "fnv1a64": "8888888888888888"}
        )
        backup.write_text(json.dumps(invalid), encoding="utf-8")
        failed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--journal", str(journal),
                "--first-expert", "0",
                "--end-expert", "4",
                "--expert-count", "8",
                "--layer", "0",
            ],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "outside expert_count" in failed.stderr

    print("OSM-39F missing-expert fetch planner: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
