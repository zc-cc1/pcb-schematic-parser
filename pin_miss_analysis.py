"""引脚漏检归因：GT 的每个 IC 引脚，到底丢在哪一步。

平台的判对条件是「元件 key + pin_<number> 键都对上 + pinname 严格相等 + 点距 <= 5 px」。
所以一个引脚没得分可能完全是四件不同的事，本脚本把 191 个 GT IC 引脚逐个走一遍：

  TP            正式计分
  identity      5 px 内、名字也对，但挂错了元件 key / 引脚 key（位号或编号问题）
  name          5 px 内，但 pinname 不一致（文字问题）
  tip_5_20      最近引脚在 5–20 px（尖端定位偏差）
  tip_20_40     最近引脚在 20–40 px（定位明显错）
  miss          > 40 px 内没有任何引脚（真·漏检）
  owner_miss    所属元件整个没导出（这条只作为附加标注，不替代上面的几何判断）

同时把 FP 也拆开：identity_fp / name_fp / spurious_near / orphan。

    & $py pin_miss_analysis.py --predictions D:/pcb_open_gt_v33angle120/final_results
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹")
sys.path.insert(0, str(ROOT / "pcb_parser_competition/scripts"))
from platform_metrics import is_ic, ntype, score_pins  # noqa: E402

GT = ROOT / "赛题六公开数据集/赛题六公开数据集/open_GT/open_GT"

OUTCOMES = ["TP", "identity", "name", "tip_5_20", "tip_20_40", "miss"]
LABELS = {"TP": "TP 计分", "identity": "身份错位", "name": "名称不符",
          "tip_5_20": "尖端偏 5–20px", "tip_20_40": "尖端偏 20–40px", "miss": "真漏检"}


def flat_pins(doc):
    out = []
    for owner, pins in (doc.get("pins") or {}).items():
        for key, pin in (pins or {}).items():
            point = pin.get("point") or {}
            try:
                out.append((owner, key, float(point["x"]), float(point["y"]),
                            str(pin.get("pinname") if pin.get("pinname") is not None else ""), pin))
            except (KeyError, TypeError):
                continue
    return out


def nearest(pins, x, y):
    best = None
    for owner, key, px, py, name, pin in pins:
        d = math.hypot(px - x, py - y)
        if best is None or d < best[0]:
            best = (d, owner, key, name, pin)
    return best


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path,
                    default=Path(r"D:/pcb_open_gt_v33angle120/final_results"))
    ap.add_argument("--gt-root", type=Path, default=GT)
    ap.add_argument("--tag", default="v33")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows, buckets = [], Counter()
    fp_buckets = Counter()
    levels = Counter()
    missing_keys = defaultdict(list)
    per_component = defaultdict(Counter)
    totals = Counter()

    for folder in sorted((p for p in args.gt_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        path = args.predictions / f"{case}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}}

        pred_all = flat_pins(pred)
        pred_ic = {k for k, v in (pred.get("components") or {}).items() if is_ic(v.get("type"))}
        gt_pins = truth.get("pins") or {}
        case_counts = Counter()

        for component, reference_pins in gt_pins.items():
            if not is_ic((truth.get("components", {}).get(component) or {}).get("type")):
                continue
            ours = (pred.get("pins") or {}).get(component) or {}
            owner_present = component in (pred.get("components") or {})
            for pin_key, reference in (reference_pins or {}).items():
                x = float(reference["point"]["x"])
                y = float(reference["point"]["y"])
                want_name = str(reference.get("pinname") if reference.get("pinname") is not None else "")
                candidate = ours.get(pin_key)
                near = nearest(pred_all, x, y)
                d = near[0] if near else float("inf")
                cd, cname = None, None
                if candidate is not None:
                    cd = math.hypot(float(candidate["point"]["x"]) - x, float(candidate["point"]["y"]) - y)
                    cname = str(candidate.get("pinname") if candidate.get("pinname") is not None else "")

                if candidate is not None:
                    if cd > 5.0:
                        levels["C_键在_点偏>5px"] += 1
                    elif cname != want_name:
                        levels["D_键在_名字错"] += 1
                    else:
                        levels["TP"] += 1
                elif owner_present:
                    levels["B_键未导出"] += 1
                    missing_keys[(case, component)].append(pin_key.replace("pin_", ""))
                else:
                    levels["A_元件未导出"] += 1

                if candidate is not None:
                    if cd > 5.0:
                        label = "tip_5_20" if cd <= 20 else ("tip_20_40" if cd <= 40 else "miss")
                    elif cname != want_name:
                        label = "name"
                    else:
                        label = "TP"
                elif d <= 5.0:
                    label = "identity" if near[3] == want_name else "name"
                elif d <= 20.0:
                    label = "tip_5_20"
                elif d <= 40.0:
                    label = "tip_20_40"
                else:
                    label = "miss"

                buckets[label] += 1
                case_counts[label] += 1
                per_component[(case, component)][label] += 1
                if label != "TP":
                    per_component[(case, component)]["owner_missing" if not owner_present else "owner_ok"] += 1
                if label == "identity" and near is not None:
                    same_component = near[1] == component
                    per_component[(case, component)]["id_same_component" if same_component
                                  else "id_other_component"] += 1
                totals["gt"] += 1
                if label == "TP":
                    totals["tp"] += 1

        # ---- FP 侧：pred 的 IC 引脚里没得分的那些，是什么 ----
        gt_flat = []
        for component, reference_pins in gt_pins.items():
            if not is_ic((truth.get("components", {}).get(component) or {}).get("type")):
                continue
            for pin_key, reference in (reference_pins or {}).items():
                gt_flat.append((component, pin_key, float(reference["point"]["x"]),
                                float(reference["point"]["y"]),
                                str(reference.get("pinname") if reference.get("pinname") is not None else "")))
        for owner, key, px, py, name, pin in pred_all:
            if owner not in pred_ic:
                continue
            reference = (gt_pins.get(owner) or {}).get(key)
            if reference is not None:
                rd = math.hypot(px - float(reference["point"]["x"]), py - float(reference["point"]["y"]))
                rname = str(reference.get("pinname") if reference.get("pinname") is not None else "")
                if rd <= 5.0 and rname == name:
                    continue  # 这是 TP，不是 FP
                if rd <= 5.0 and rname != name:
                    fp_buckets["name_fp"] += 1
                    continue
            near_gt = None
            for gc, gk, gx, gy, gname in gt_flat:
                dd = math.hypot(px - gx, py - gy)
                if near_gt is None or dd < near_gt[0]:
                    near_gt = (dd, gc, gk, gname)
            if near_gt and near_gt[0] <= 5.0:
                fp_buckets["identity_fp" if near_gt[3] == name else "name_fp"] += 1
            elif near_gt and near_gt[0] <= 20.0:
                fp_buckets["spurious_near"] += 1
            else:
                fp_buckets["orphan"] += 1
        totals["fp"] += sum(v for k, v in fp_buckets.items()) if False else 0

        score, detail = score_pins(pred, truth)
        rows.append({"case": case, "gt": sum(case_counts.values()), "tp": case_counts["TP"],
                     "identity": case_counts["identity"], "name": case_counts["name"],
                     "tip_5_20": case_counts["tip_5_20"], "tip_20_40": case_counts["tip_20_40"],
                     "miss": case_counts["miss"], "f1": score["f1"],
                     "platform_tp": score["tp"], "platform_fn": score["fn"], "platform_fp": score["fp"]})

    gt_total = sum(r["gt"] for r in rows)
    tp_total = sum(r["tp"] for r in rows)
    recall = tp_total / gt_total if gt_total else 0.0
    # 三种口径下的上限：检测到位 / 检测+命名到位 / 正式计分
    detect = tp_total + buckets["identity"] + buckets["name"]
    detect_name = tp_total + buckets["identity"]
    f_cur, p_cur, r_cur = f1(tp_total, 22, gt_total - tp_total)
    f_id, _, _ = f1(detect_name, 22, gt_total - detect_name)
    f_det, _, _ = f1(detect, 22, gt_total - detect)

    print(f"预测目录 {args.predictions}")
    print(f"GT IC 引脚 {gt_total}   计分 TP {tp_total}   FN {gt_total - tp_total}   "
          f"FP {sum(fp_buckets.values())}   recall {recall:.1%}   PinF1 {f_cur:.4f}")
    print()
    print("=== 漏检去向（FN = %d）===" % (gt_total - tp_total))
    for label in OUTCOMES[1:]:
        n = buckets[label]
        if n:
            print(f"  {LABELS[label]:14s} {n:4d}   {n / max(gt_total - tp_total, 1):6.1%}  (占全部 GT {n / gt_total:5.1%})")
    print()
    print(f"=== 三层上限 ===")
    print(f"  现状计分            TP {tp_total:3d}   PinF1 {f_cur:.4f}   recall {r_cur:.1%}")
    print(f"  身份修好（键归位）  TP {detect_name:3d}   PinF1 {f_id:.4f}   recall {detect_name / gt_total:.1%}  (+{detect_name - tp_total})")
    print(f"  命名也修好          TP {detect:3d}   PinF1 {f_det:.4f}   recall {detect / gt_total:.1%}  (+{detect - tp_total})")
    print(f"  检测/尖端全修好     TP {gt_total:3d}   PinF1 1.0000   (+{gt_total - detect})")
    print()
    print("=== 导出层级（FN 卡在哪一步）===")
    for key in ("A_元件未导出", "B_键未导出", "C_键在_点偏>5px", "D_键在_名字错"):
        print(f"  {key:16s} {levels[key]:3d}")
    print()
    print("=== 每修好一类，PinF1 走到哪（FP 固定 22）===")
    fp_total = sum(fp_buckets.values())
    cum = tp_total
    print(f"  {'现状':22s} TP {cum:3d}  PinF1 {f1(cum, fp_total, gt_total - cum)[0]:.4f}")
    for label in ("identity", "name", "tip_5_20", "tip_20_40", "miss"):
        cum += buckets[label]
        print(f"  {'再修好 ' + LABELS[label]:22s} TP {cum:3d}  PinF1 {f1(cum, fp_total, gt_total - cum)[0]:.4f}"
              f"  (+{(f1(cum, fp_total, gt_total - cum)[0] - f_cur) * 0.25 * 100:.2f} 总分)")
    print()
    worst_keys = sorted(missing_keys.items(), key=lambda kv: -len(kv[1]))[:10]
    if worst_keys:
        print("=== 未导出的引脚键（按元件，前 10）===")
        for (case, component), keys in worst_keys:
            print(f"  case {case:>2} {component:6s} 缺 {len(keys):2d}: {','.join(keys)}")
        print()
    print()
    print("=== 逐例 ===")
    print(f"{'case':>4}{'GT':>5}{'TP':>5}{'身份':>6}{'名称':>6}{'5-20':>6}{'20-40':>7}{'漏检':>6}{'PinF1':>8}")
    for r in rows:
        print(f"{r['case']:>4}{r['gt']:>5}{r['tp']:>5}{r['identity']:>6}{r['name']:>6}"
              f"{r['tip_5_20']:>6}{r['tip_20_40']:>7}{r['miss']:>6}{r['f1']:>8.3f}")
    print()
    print("=== FP 去向（共 %d）===" % sum(fp_buckets.values()))
    for key in ("identity_fp", "name_fp", "spurious_near", "orphan"):
        if fp_buckets[key]:
            print(f"  {key:14s} {fp_buckets[key]:3d}")
    print()
    print("=== 丢分最狠的元件（非 TP 数倒序，前 20）===")
    ranked = sorted(((k, v) for k, v in per_component.items() if v.get("TP", 0) < sum(v[x] for x in OUTCOMES)),
                    key=lambda kv: -(sum(kv[1][x] for x in OUTCOMES) - kv[1].get("TP", 0)))
    for (case, component), counts in ranked[:20]:
        lost = sum(counts[x] for x in OUTCOMES) - counts.get("TP", 0)
        detail = ", ".join(f"{LABELS[x]}={counts[x]}" for x in OUTCOMES[1:] if counts[x])
        extra = "整块未导出" if counts.get("owner_missing") else ""
        print(f"  case {case:>2} {component:6s} TP {counts.get('TP', 0):2d}/{sum(counts[x] for x in OUTCOMES):2d} 丢 {lost:2d}  {detail} {extra}")

    payload = {"predictions": str(args.predictions), "gt_ic_pins": gt_total, "tp": tp_total,
               "fn": gt_total - tp_total, "fp": dict(fp_buckets), "buckets": dict(buckets),
               "levels": dict(levels),
               "missing_keys": {f"{c}/{k}": v for (c, k), v in missing_keys.items()},
               "pin_f1_current": round(f_cur, 4), "pin_f1_identity_fixed": round(f_id, 4),
               "pin_f1_detection_fixed": round(f_det, 4), "per_case": rows}
    out = ROOT / f"pin_miss_{args.tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
