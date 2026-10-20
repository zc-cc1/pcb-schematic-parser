"""Where do the net_input / vbus detections go in case 12?"""
import json
from pathlib import Path

LOGS = Path(r"D:/pcb_open_gt_v20/logs")

for name in ("12_component_export_audit.json", "12_engineering_export_coverage.json", "12_summary.json"):
    p = LOGS / name
    if not p.is_file():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    print("==", name, "keys:", list(d)[:12])
    s = json.dumps(d, ensure_ascii=False)
    for kw in ("vbus", "net_input", "VIN", "pin_port", "skipped"):
        idx = s.find(kw)
        if idx >= 0:
            print(f"   {kw}: ...{s[max(0, idx-160):idx+220]}...")
        else:
            print(f"   {kw}: absent")
    print()
