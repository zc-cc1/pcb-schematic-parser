"""补充统计：键在但点偏的偏移量分布，以及键未导出但 40px 内已有引脚的数量。"""
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, r"C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition/scripts")
from platform_metrics import is_ic  # noqa: E402

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:/pcb_open_gt_v33angle120/final_results")

offsets, inward, gross = [], 0, []
near40 = collections.Counter()
for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    truth = json.loads((folder / f"{case}.json").read_text(encoding="utf-8"))
    pred = json.loads((PRED / f"{case}_Pred.json").read_text(encoding="utf-8"))
    flat = [(o, k, q["point"]["x"], q["point"]["y"])
            for o, pins in (pred.get("pins") or {}).items() for k, q in pins.items()]
    for component, reference_pins in truth["pins"].items():
        if not is_ic(truth["components"][component]["type"]):
            continue
        ours = (pred.get("pins") or {}).get(component) or {}
        bb = truth["components"][component]["bbox"]
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        for key, reference in reference_pins.items():
            gx, gy = reference["point"]["x"], reference["point"]["y"]
            candidate = ours.get(key)
            if candidate is not None:
                distance = math.hypot(candidate["point"]["x"] - gx, candidate["point"]["y"] - gy)
                if distance > 5:
                    offsets.append(distance)
                    ux, uy = gx - cx, gy - cy          # 本体中心 -> GT 尖端
                    ox, oy = gx - candidate["point"]["x"], gy - candidate["point"]["y"]  # 我们的点 -> GT 尖端
                    if ox * ux + oy * uy > 0:
                        inward += 1                     # GT 尖端比我们更靠外：尖端没伸到位
                    if distance >= 60:
                        gross.append((case, component, key, round(distance)))
                continue
            best = min(flat, key=lambda q: math.hypot(q[2] - gx, q[3] - gy))
            if math.hypot(best[2] - gx, best[3] - gy) <= 40:
                near40[case] += 1

print(f"键在但点偏 {len(offsets)} 个：偏移中位 {st.median(offsets):.1f} px，最大 {max(offsets):.1f} px，"
      f"方向朝外（尖端没伸到位）{inward}/{len(offsets)}")
print(f"  其中偏移 >=60 px 的（基本等于没定位到）：{len(gross)} 个 {gross}")
print(f"键未导出、但 40 px 内已有引脚的：{sum(near40.values())} 个，分布 {dict(sorted(near40.items(), key=lambda kv: int(kv[0])))}")
