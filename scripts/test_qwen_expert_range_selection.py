#!/usr/bin/env python3
import json
import pathlib
import sys
import tempfile

import fetch_qwen_expert_range as target


def require_raises(fn):
    try:
        fn()
    except Exception:
        return
    raise AssertionError("expected exception")


def main() -> int:
    assert target.resolve_expert_ids("0,2,511", 0, 2) == [0, 2, 511]
    assert target.resolve_expert_ids(None, 3, 2) == [3, 4]
    require_raises(lambda: target.resolve_expert_ids("1,1", 0, 2))
    require_raises(lambda: target.resolve_expert_ids("-1", 0, 2))
    require_raises(lambda: target.resolve_expert_ids("", -1, 2))

    calls = []
    original = target.fetch_expert
    original_argv = sys.argv[:]
    try:
        def fake_fetch(range_manifest, output_root, layer, expert, hidden, intermediate):
            calls.append(expert)
            return 100 + expert

        target.fetch_expert = fake_fetch
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = root / "range.json"
            manifest.write_text(
                json.dumps({"model": "fixture", "snapshot": "fixture", "shards": []}),
                encoding="utf-8",
            )
            output = root / "out"
            sys.argv = [
                "fetch_qwen_expert_range.py",
                "--range-manifest", str(manifest),
                "--output-dir", str(output),
                "--layer", "0",
                "--experts", "0,2",
            ]
            assert target.main() == 0
            assert calls == [0, 2]
            summary = json.loads((output / "expert-range.json").read_text())
            assert summary["experts"] == [0, 2]
            assert summary["count"] == 2
            assert summary["total_fetched_bytes"] == 202
    finally:
        target.fetch_expert = original
        sys.argv = original_argv

    print("OSM-39F explicit expert selection: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
