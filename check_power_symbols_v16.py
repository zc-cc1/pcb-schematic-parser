"""Do gnd/v symbols count as components, and where do we lose them?"""
import json
from pathlib import Path

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
PRED = Path(r"D:/pcb_open_gt_v16/final_results")
POWER = {"gnd", "ground", "v", "vcc", "vdd"}


def bbox_ok(a, b, tol=20.0):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


rows = []
gt_total = pred_total = matched = 0
gt_ord = pred_ord = ord_matched = 0
fail = {"type_or_value": 0, "no_prediction": 0, "geometry": 0}
gt_vals, pred_vals = {}, {}

for folder in sorted(GT.iterdir(), key=lambda p: int(p.name)):
    case = folder.name
    gt = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))["components"]
    p_path = PRED / f"{case}_Pred.json"
    pred = json.loads(p_path.read_text(encoding="utf-8"))["components"] if p_path.is_file() else {}
    gp = {k: v for k, v in gt.items() if str(v.get("type", "")).lower() in POWER}
    pp = {k: v for k, v in pred.items() if str(v.get("type", "")).lower() in POWER}
    gt_total += len(gp)
    pred_total += len(pp)
    gt_ord += len(gt) - len(gp)
    ord_pred = {k: v for k, v in pred.items() if str(v.get("type", "")).lower() not in POWER}
    pred_ord += len(ord_pred)
    # greedy one-to-one geometric match with same type+value
    free = [k for k in pp]
    m = 0
    for gk, gv in gp.items():
        gt_vals[gv.get("value")] = gt_vals.get(gv.get("value"), 0) + 1
        hit = None
        for pk in free:
            pv = pp[pk]
            if str(pv.get("type", "")).lower() != str(gv.get("type", "")).lower():
                continue
            if (pv.get("value") or "") != (gv.get("value") or ""):
                continue
            if bbox_ok(pv["bbox"], gv["bbox"]):
                hit = pk
                break
        if hit:
            m += 1
            free.remove(hit)
    for pk in free:
        pred_vals[pp[pk].get("value")] = pred_vals.get(pp[pk].get("value"), 0) + 1
    matched += m
    rows.append((case, len(gp), len(pp), m, len(gt) - len(gp)))
    # ordinary side
    om = 0
    for gk, gv in gt.items():
        if str(gv.get("type", "")).lower() in POWER:
            continue
        if gk in pred and pred[gk].get("type") == gv.get("type"):
            assert gv is not None
            om += 1
    ord_matched += om

print(f"{'case':>4} {'GT power':>9} {'pred power':>11} {'geom+type+value match':>22} {'GT ordinary':>12}")
for r in rows:
    print(f"{r[0]:>4} {r[1]:>9} {r[2]:>11} {r[3]:>22} {r[4]:>12}")
print(f"{'ALL':>4} {gt_total:>9} {pred_total:>11} {matched:>22} {gt_ord:>12}")
print(f"\npower share of GT components: {gt_total}/{gt_total+gt_ord} = {gt_total/(gt_total+gt_ord):.1%}")
print(f"power matched: {matched}/{gt_total} = {matched/gt_total:.1%}")
print("\nGT power values:", gt_vals)
print("unmatched PRED power values:", pred_vals)

