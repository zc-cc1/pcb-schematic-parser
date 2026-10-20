"""Why are pin fields withheld? Pulls every reason counter from the stage logs."""
import ast
import json
import sys
from collections import Counter
from pathlib import Path

LOGS = Path(r"D:\pcb_v6_check\out_v6_full\logs")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]


def docs(path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        for parse in (json.loads, ast.literal_eval):
            try:
                doc = parse(line)
            except Exception:
                continue
            if isinstance(doc, dict):
                yield doc
            break


def walk(node, path="", out=None):
    if out is None:
        out = []
    if isinstance(node, dict):
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else key, out)
    elif isinstance(node, int) and any(token in path for token in
                                       ("reason_counts", "withheld", "unresolved", "status_counts")):
        out.append((path, node))
    return out


def main():
    merged = Counter()
    for case in CASES:
        path = LOGS / f"{case}.log"
        if not path.exists():
            continue
        rows = Counter()
        for doc in docs(path):
            for key, value in walk(doc):
                compact = key.replace("conservative.", "").replace("candidate.", "")
                rows[compact] += value
        print(f"\n=== {case}")
        for key, value in sorted(rows.items(), key=lambda kv: -kv[1])[:12]:
            print(f"   {value:5d}  {key}")
            merged[key] += value
    print("\n==== TOTAL over the six pages ===")
    for key, value in sorted(merged.items(), key=lambda kv: -kv[1])[:18]:
        print(f"   {value:5d}  {key}")


if __name__ == "__main__":
    sys.exit(main())
