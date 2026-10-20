"""Audit the 8th eval: what the net/pin stages built vs exported now."""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

LOG7 = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第七次评测-33.525/scoring_result/logs")
LOG8 = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第8次测评-54.92/scoring_result/logs")


def parse(folder: Path) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*.log"), key=lambda p: int(p.stem)):
        rec = {"case": path.stem}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
            if "engineering_export" in obj:
                ee = obj["engineering_export"]
                rec["exp_pins"] = ee.get("exported_pins")
                rec["exp_nets"] = ee.get("exported_nets")
            if "completion" in obj:
                c = obj["completion"].get("conservative", {})
                rec["multi_pin"] = c.get("multi_pin_networks")
                rec["field_incomplete"] = c.get("field_only_incomplete_networks")
                rec["nonfield"] = c.get("non_field_export_blocker_counts")
            if "network_guard" in obj:
                rec["blocked"] = obj["network_guard"].get("blocked_networks")
            if "pin_lead_snap" in obj or "lead_snap" in obj:
                rec["lead_snap"] = obj.get("pin_lead_snap") or obj.get("lead_snap")
            if "whole_network" in obj or "export" in obj:
                pass
            if isinstance(obj.get("conservative"), dict) and "pin_reason_counts" in obj["conservative"]:
                rec["pin_reasons"] = obj["conservative"]["pin_reason_counts"]
        out.append(rec)
    return out


cols = ["case", "multi_pin", "field_incomplete", "blocked", "exp_pins", "exp_nets"]
for tag, folder in (("7th", LOG7), ("8th", LOG8)):
    rows = parse(folder)
    print(f"===== {tag} =====")
    w = {c: max([len(c)] + [len(str(r.get(c, "-"))) for r in rows]) for c in cols}
    print(" ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print(" ".join(str(r.get(c, "-")).ljust(w[c]) for c in cols))
    tot = Counter()
    for r in rows:
        for k, v in (r.get("pin_reasons") or {}).items():
            tot[k] += v
    print("pin export blockers:", dict(tot))
    nf = Counter()
    for r in rows:
        for k, v in (r.get("nonfield") or {}).items():
            nf[k] += v
    print("non-field net blockers:", dict(nf))
    print("sum exported pins:", sum(r.get("exp_pins") or 0 for r in rows),
          " nets:", sum(r.get("exp_nets") or 0 for r in rows))
    print()
