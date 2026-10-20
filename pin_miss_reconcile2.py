"""修正版归因：先看键在不在、名字对不对，再看同元件有没有引脚，最后才谈尖端。

上一版把「键缺失」的引脚按"全图最近引脚"归到尖端偏移，结果 USB1 旁边的 R13/R14
引脚（20–40px）被当成了 USB1 自己的尖端偏差。这里按同元件口径重算。
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

rows = []
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads((folder / f"{case}.json").read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    for component, reference_pins in truth["pins"].items():
        if not is_ic(truth["components"][component]["type"]):
            continue
        ours = (pred.get("pins") or {}).get(component) or {}
        same = [(k, q["point"]["x"], q["point"]["y"], str(q.get("pinname") or ""))
                for k, q in ours.items()]
        for key, reference in reference_pins.items():
            gx, gy = reference["point"]["x"], reference["point"]["y"]
            want = str(reference.get("pinname") or "")
            candidate = ours.get(key)
            if candidate is not None:
                distance = math.hypot(candidate["point"]["x"] - gx, candidate["point"]["y"] - gy)
                same_name = str(candidate.get("pinname") or "") == want
                if distance <= 5 and same_name:
                    continue
                if not same_name:
                    kind = "编号错位(键在/名不同)"
                elif distance <= 20:
                    kind = "尖端偏 5–20px(键名都对)"
                else:
                    kind = "尖端/键偏 >20px(键名都对)"
                rows.append({"case": case, "component": component, "key": key, "name": want,
                             "kind": kind, "distance": round(distance, 1), "near": round(distance, 1),
                             "near_key": key})
                continue
            near = min(same, key=lambda q: math.hypot(q[1] - gx, q[2] - gy)) if same else None
            sd = math.hypot(near[1] - gx, near[2] - gy) if near else None
            if sd is not None and sd <= 5:
                kind = "编号错位(脚在 5px 内，键不对)"
            elif sd is not None and sd <= 45:
                kind = "键缺失(同元件 5–45px 有脚)"
            else:
                kind = "真漏检(同元件 45px 内无脚)"
            rows.append({"case": case, "component": component, "key": key, "name": want,
                         "kind": kind, "distance": None,
                         "near": None if sd is None else round(sd, 1),
                         "near_key": near[0] if near else None})

counts = Counter(r["kind"] for r in rows)
print("没得分的 GT IC 引脚", len(rows))
for kind, n in counts.most_common():
    print(f"  {kind:28s}{n:3d}")
print()
print("按 case × 原因：")
by_case = defaultdict(Counter)
for r in rows:
    by_case[r["case"]][r["kind"]] += 1
kinds = [k for k, _ in counts.most_common()]
print(f"{'case':>4}  " + "  ".join(f"{k[:10]:>12s}" for k in kinds))
for case in sorted(by_case, key=int):
    print(f"{case:>4}  " + "  ".join(f"{by_case[case][k]:>12d}" for k in kinds))
print()
print("明细（键缺失/编号错位，按距离）：")
for r in sorted(rows, key=lambda r: (r["case"], r["component"], r["key"])):
    if r["kind"].startswith("真漏检"):
        continue
    print(f"  case {r['case']:>2} {r['component']:6s} {r['key']:9s} {r['name'][:9]:9s} "
          f"{r['kind']:26s} 最近 {str(r['near']):>6s} ({r['near_key']})")

out = Path(r"C:/Users/Pc/Desktop/新建文件夹/pin_miss_v33_corrected.json")
out.write_text(json.dumps({"gt_ic_pins": 191, "tp": 140, "counts": dict(counts), "rows": rows},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwritten:", out)
