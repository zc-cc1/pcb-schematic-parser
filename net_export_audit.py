"""Parse the 5th-eval per-case logs: how many nets/pins were built vs actually exported."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

LOGS = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第五次评测-34.65/scoring_result/logs")

rows = []
for path in sorted(LOGS.glob("*.log"), key=lambda p: int(p.stem)):
    text = path.read_text(encoding="utf-8", errors="replace")
    rec = {"case": path.stem}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(obj, dict):
            continue
        flat = json.dumps(obj)
        if "engineering_export" in obj:
            ee = obj["engineering_export"]
            rec["exp_comp"] = ee.get("exported_components")
            rec["exp_pins"] = ee.get("exported_pins")
            rec["exp_nets"] = ee.get("exported_nets")
        if "proposal_export" in obj:
            pe = obj["proposal_export"]
            for mode in ("conservative", "candidate"):
                if mode in pe:
                    rec[f"prop_{mode}_pins"] = pe[mode].get("exported_pins")
                    rec[f"prop_{mode}_nets"] = pe[mode].get("exported_nets")
        if "candidate_count" in obj.get("summary", {}):
            s = obj["summary"]
            rec["pin_candidates"] = s.get("candidate_count")
            rec["owned"] = s.get("owned_candidates")
            rec["num_observed"] = s.get("number_observed")
        if "completion" in obj:
            comp = obj["completion"]["conservative"]
            rec["multi_pin_networks"] = comp.get("multi_pin_networks")
            rec["field_incomplete"] = comp.get("field_only_incomplete_networks")
            rec["nonfield_blockers"] = comp.get("non_field_export_blocker_counts")
        if "network_guard" in obj:
            rec["blocked_networks"] = obj["network_guard"].get("blocked_networks")
        if "power" in obj and isinstance(obj.get("power"), dict) and "network_proposals_after" in obj["power"]:
            rec["net_proposals_after_power"] = obj["power"]["network_proposals_after"]
        if "conservative" in obj and isinstance(obj.get("conservative"), dict) and "pin_reason_counts" in obj["conservative"]:
            rec["pin_reasons"] = obj["conservative"]["pin_reason_counts"]
    rows.append(rec)

cols = ["case", "pin_candidates", "owned", "num_observed", "multi_pin_networks", "field_incomplete",
        "blocked_networks", "exp_comp", "exp_pins", "exp_nets"]
w = {c: len(c) for c in cols}
for r in rows:
    for c in cols:
        w[c] = max(w[c], len(str(r.get(c, "-"))))
print(" ".join(c.ljust(w[c]) for c in cols))
print("-" * (sum(w.values()) + len(cols)))
for r in rows:
    print(" ".join(str(r.get(c, "-")).ljust(w[c]) for c in cols))

print()
print("pin export blockers (sum over cases):")
tot = {}
for r in rows:
    for k, v in (r.get("pin_reasons") or {}).items():
        tot[k] = tot.get(k, 0) + v
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<32}{v}")
print()
print("non-field net blockers (sum over cases):")
tot = {}
for r in rows:
    for k, v in (r.get("nonfield_blockers") or {}).items():
        tot[k] = tot.get(k, 0) + v
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<48}{v}")
