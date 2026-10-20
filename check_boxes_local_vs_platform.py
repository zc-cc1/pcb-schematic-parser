import json
from pathlib import Path

B = Path(r"C:/Users/Pc/Desktop/新建文件夹/评测结果/第8次测评-54.92/intermediate_files/final_results")
V13 = Path(r"D:/pcb_open_gt_v13/final_results")
GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

for case in ("13", "12"):
    p = json.loads((B / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    q = json.loads((V13 / f"{case}_Pred.json").read_text(encoding="utf-8"))["components"]
    gt = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))["components"]
    print(f"=== case {case} ===")
    for cid in list(gt.keys()):
        if gt[cid]["type"] not in {"r", "c", "d", "led", "l"}:
            continue
        gb, pb, qb = gt[cid].get("bbox"), (p.get(cid) or {}).get("bbox"), (q.get(cid) or {}).get("bbox")

        def wh(b):
            return None if not b else (round(b[2] - b[0], 1), round(b[3] - b[1], 1))

        print(f"  {cid:<8} GT={wh(gb)!s:<16} platform={wh(pb)!s:<16} local_v13={wh(qb)!s}")
