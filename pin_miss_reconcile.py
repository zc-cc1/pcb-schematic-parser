"""对账：把「图里那张 33 个漏检」逐个放到我们自己的产物里看状态。

对每个没得分的 GT IC 引脚，同时找两个邻居：
  * 同元件下最近的导出引脚（判断：我们在这个元件上到底有没有检到这一条）
  * 全图最近的导出引脚（判断：附近有没有别的东西，容易把"元件没检到"误读成"尖端偏"）
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import is_ic  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v33angle120/final_results")

# 图里点名的那 33 个
CLAIM = {
    ("1", "USB1"): ["1", "5", "6", "7", "9"],
    ("1", "USB2"): ["6", "5", "2"],
    ("11", "USB1"): ["7", "A5", "B5"],
    ("11", "USB2"): ["5", "4", "3", "2", "1"],
    ("13", "H1"): ["1", "2", "3"],
    ("8", "U1"): ["2", "6", "16"],
    ("10", "U1"): ["4"],
    ("1", "U2"): ["4"],
    ("6", "U1"): ["1"],
    ("10", "U2"): ["3"],
    ("11", "U1"): ["1", "2", "6", "8"],
}

rows = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads((folder / f"{case}.json").read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    flat = [(o, k, q["point"]["x"], q["point"]["y"], str(q.get("pinname") or ""))
            for o, pins in (pred.get("pins") or {}).items() for k, q in pins.items()]
    for component, reference_pins in truth["pins"].items():
        if not is_ic(truth["components"][component]["type"]):
            continue
        ours = (pred.get("pins") or {}).get(component) or {}
        for key, reference in reference_pins.items():
            gx, gy = reference["point"]["x"], reference["point"]["y"]
            want = str(reference.get("pinname") or "")
            candidate = ours.get(key)
            if candidate is not None:
                distance = math.hypot(candidate["point"]["x"] - gx, candidate["point"]["y"] - gy)
                same = str(candidate.get("pinname") or "") == want
                if distance <= 5 and same:
                    continue
                status = "offset" if distance > 5 else "name"
                rows.append({"case": case, "component": component, "key": key.replace("pin_", ""),
                             "name": want, "status": status, "distance": round(distance, 1),
                             "same_component_near": round(distance, 1),
                             "any_owner": component, "any_distance": round(distance, 1)})
                continue
            same = [(math.hypot(px - gx, py - gy), k, nm) for o, k, px, py, nm in flat if o == component]
            anyp = [(math.hypot(px - gx, py - gy), o, k) for o, k, px, py, nm in flat]
            sd = min(same)[0] if same else None
            ad, ao, ak = min(anyp) if anyp else (None, None, None)
            status = "key_missing_same_near" if sd is not None and sd <= 40 else "key_missing_no_pin"
            rows.append({"case": case, "component": component, "key": key.replace("pin_", ""),
                         "name": want, "status": status,
                         "distance": None, "same_component_near": None if sd is None else round(sd, 1),
                         "any_owner": ao, "any_distance": round(ad, 1)})

counts = Counter(r["status"] for r in rows)
print("没得分的 GT IC 引脚共", len(rows), dict(counts))
print()
print("=== 图里点名的 33 个，逐个对账 ===")
print(f"{'case':>4} {'comp':6s} {'pin':4s} {'GT名':6s} {'我们键':4s} {'状态':22s} {'同元件最近':>10s} {'全图最近':>18s}")
seen = set()
for r in rows:
    tag = (r["case"], r["component"])
    claimed = tag in CLAIM and r["key"] in CLAIM[tag]
    if not claimed:
        continue
    seen.add((tag, r["key"]))
    exported = "在" if r["status"] in ("offset", "name") else "无"
    print(f"{r['case']:>4} {r['component']:6s} {r['key']:4s} {r['name'][:6]:6s} {exported:4s} {r['status']:22s}"
          f" {str(r['same_component_near']):>10s} {str(r['any_owner']) + ' ' + str(r['any_distance']):>18s}")
missing_in_fact = [(tag, k) for tag, keys in CLAIM.items() for k in keys if (tag, k) not in seen]
print("\n图里点名但我们的数据对不上的:", missing_in_fact)

print("\n=== 我们统计里的全部 key 缺失（同元件 40px 内有没有引脚）===")
for r in rows:
    if r["status"].startswith("key_missing"):
        print(f"  case {r['case']:>2} {r['component']:6s} pin_{r['key']:4s} {r['name'][:8]:8s} "
              f"同元件最近 {str(r['same_component_near']):>6s}  全图最近 {r['any_owner']} {r['any_distance']}")
