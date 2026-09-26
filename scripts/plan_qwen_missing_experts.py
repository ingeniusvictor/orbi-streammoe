#!/usr/bin/env python3
import argparse
import json
import pathlib


def parse_journal(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    root = json.loads(path.read_text(encoding="utf-8"))
    if root.get("schema_version") != 1:
        raise RuntimeError("unsupported QPACK layer journal schema")
    completed = root.get("completed")
    if not isinstance(completed, list):
        raise RuntimeError("journal completed must be an array")
    seen = set()
    for item in completed:
        expert = item.get("expert")
        checksum = item.get("fnv1a64")
        if not isinstance(expert, int) or expert < 0:
            raise RuntimeError("journal contains invalid expert id")
        if expert in seen:
            raise RuntimeError("journal contains duplicate expert id")
        if not isinstance(checksum, str) or len(checksum) != 16:
            raise RuntimeError("journal contains invalid expert checksum")
        int(checksum, 16)
        seen.add(expert)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    parser.add_argument("--first-expert", type=int, required=True)
    parser.add_argument("--end-expert", type=int, required=True)
    parser.add_argument("--expert-count", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    if (
        args.expert_count <= 0
        or args.layer < 0
        or args.first_expert < 0
        or args.first_expert >= args.end_expert
        or args.end_expert > args.expert_count
    ):
        raise RuntimeError("invalid requested expert interval")

    journal_path = pathlib.Path(args.journal)
    journal = parse_journal(journal_path)
    completed = set()

    if journal:
        if int(journal.get("layer_index", -1)) != args.layer:
            raise RuntimeError("journal layer disagrees with requested layer")
        if int(journal.get("expert_count", -1)) != args.expert_count:
            raise RuntimeError("journal expert_count disagrees with request")
        completed = {int(item["expert"]) for item in journal["completed"]}
        if any(expert >= args.expert_count for expert in completed):
            raise RuntimeError("journal contains expert id outside expert_count")

    requested = list(range(args.first_expert, args.end_expert))
    missing = [expert for expert in requested if expert not in completed]
    already_complete = [expert for expert in requested if expert in completed]

    plan = {
        "schema_version": 1,
        "layer_index": args.layer,
        "expert_count": args.expert_count,
        "first_expert": args.first_expert,
        "end_expert_exclusive": args.end_expert,
        "requested_experts": requested,
        "already_complete": already_complete,
        "missing_experts": missing,
        "missing_experts_csv": ",".join(str(v) for v in missing),
        "journal_present": bool(journal),
    }

    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = pathlib.Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
