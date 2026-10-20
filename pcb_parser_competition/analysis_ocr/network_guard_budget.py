"""How many nets does the pipeline compute, and what blocks them from export?

Each case log contains one JSON line per stage. This pulls the network guard,
completion and export counts so the connectivity gap can be attributed.
"""
import ast
import json
import sys
from pathlib import Path

LOGS = Path(r"D:\pcb_v6_check\out_v6_full\logs")
CASES = ["0001", "0002", "0018", "0029", "0080", "0102"]


def json_lines(path):
    """The stages print Python dict literals (single quotes), not JSON."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            try:
                doc = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
        if isinstance(doc, dict):
            yield doc


def main():
    rows = []
    for case in CASES:
        path = LOGS / f"{case}.log"
        if not path.exists():
            continue
        guard = completion = proposal = None
        for doc in json_lines(path):
            if "network_guard" in doc:
                guard = doc["network_guard"]
            if "completion" in doc:
                completion = doc["completion"]
            if "proposal_export" in doc:
                proposal = doc["proposal_export"]
        rows.append((case, guard or {}, completion or {}, proposal or {}))
    print(f"{'case':6} {'multi-pin':>9} {'field-only':>10} {'blocked':>8} {'exported nets':>13}  blockers")
    for case, guard, completion, proposal in rows:
        cons = (completion.get("conservative") or {})
        multi = cons.get("multi_pin_networks")
        field_only = cons.get("field_only_incomplete_networks")
        blockers = cons.get("non_field_export_blocker_counts") or {}
        exp = ((proposal.get("conservative") or {}).get("exported_nets"))
        print(f"{case:6} {str(multi):>9} {str(field_only):>10} {str(guard.get('blocked_networks')):>8} "
              f"{str(exp):>13}  {blockers}")
    total = {}
    for _, guard, completion, proposal in rows:
        cons = (completion.get("conservative") or {})
        for key in ("multi_pin_networks", "field_only_incomplete_networks"):
            total[key] = total.get(key, 0) + (cons.get(key) or 0)
        total["blocked"] = total.get("blocked", 0) + (guard.get("blocked_networks") or 0)
        total["exported"] = total.get("exported", 0) + ((proposal.get("conservative") or {}).get("exported_nets") or 0)
    print("\nTOTAL:", total)


if __name__ == "__main__":
    sys.exit(main())
