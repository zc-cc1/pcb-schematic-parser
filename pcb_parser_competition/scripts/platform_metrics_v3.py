"""Local aggregate scorer with approximate legacy connectivity.

Wire matching now uses platform_wire_metrics through old.score_wire (2026-10-20).
Its counts are validated against all 26 archived eval8/eval9 cases. Connectivity
below remains a local proxy; the aggregate is not an official platform score.
Its singleton filter was a historical heuristic. Archived reports confirm that
the official logical-net matching includes singleton nets; that heuristic must
not be reused in the corrected wire scorer.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import platform_metrics as old  # noqa: E402
import platform_metrics_v2 as v2  # noqa: E402

COMPONENT_TOLERANCE_PX = v2.COMPONENT_TOLERANCE_PX
PIN_TOLERANCE_PX = v2.PIN_TOLERANCE_PX
WIRE_TOLERANCE_PX = v2.WIRE_TOLERANCE_PX
WEIGHTS = dict(v2.WEIGHTS)
f1_block = old.f1_block


def score_connectivity(pred, truth, minimum_members=2):
    truth_nets = [n for n in v2.nets_of(truth) if len(n) >= minimum_members]
    pred_nets = [n for n in v2.nets_of(pred) if len(n) >= minimum_members]
    if not truth_nets or not pred_nets:
        return f1_block(0, len(pred_nets), len(truth_nets))
    import numpy as np
    score = np.zeros((len(pred_nets), len(truth_nets)))
    for i, p in enumerate(pred_nets):
        for j, t in enumerate(truth_nets):
            inter = len(p & t)
            if inter:
                score[i, j] = 2 * inter / (len(p) + len(t))
    rows, cols = old.linear_sum_assignment(-score)
    tp = fp = fn = 0
    matched_pred, matched_truth = set(), set()
    for i, j in zip(rows, cols):
        if score[i, j] <= 0:
            continue
        inter = len(pred_nets[i] & truth_nets[j])
        tp += inter
        fp += len(pred_nets[i]) - inter
        fn += len(truth_nets[j]) - inter
        matched_pred.add(i)
        matched_truth.add(j)
    for i in range(len(pred_nets)):
        if i not in matched_pred:
            fp += len(pred_nets[i])
    for j in range(len(truth_nets)):
        if j not in matched_truth:
            fn += len(truth_nets[j])
    return f1_block(tp, fp, fn)


def score_case(pred, truth, height=None):
    component, _detail = old.score_components(pred, truth)
    pin = old.score_pins(pred, truth, PIN_TOLERANCE_PX)[0]
    connectivity = score_connectivity(pred, truth)
    wire = old.score_wire(pred, truth, WIRE_TOLERANCE_PX)
    total = round(WEIGHTS["component_f1"] * component["f1"] + WEIGHTS["pin_f1"] * pin["f1"]
                  + WEIGHTS["connectivity_f1"] * connectivity["f1"]
                  + WEIGHTS["wire_f1"] * wire["f1"], 4)
    return {"component_f1": component, "pin_f1": pin, "connectivity_f1": connectivity,
            "wire_f1": wire, "final_score": total}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gt_root = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
    predictions = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:/pcb_final_check/final_results")
    totals = {"component_f1": 0.0, "pin_f1": 0.0, "connectivity_f1": 0.0, "wire_f1": 0.0}
    scores, count, rows = [], 0, []
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        pred = json.loads((predictions / f"{folder.name}_Pred.json").read_text(encoding="utf-8"))
        result = score_case(pred, truth)
        for key in totals:
            totals[key] += result[key]["f1"]
        scores.append(result["final_score"])
        rows.append((folder.name, result))
        count += 1
    header = f"{'case':>4}{'comp':>8}{'pin':>8}{'conn':>8}{'wire':>8}{'score':>8}"
    print(header)
    for case, result in rows:
        print(f"{case:>4}{result['component_f1']['f1']:>8.3f}{result['pin_f1']['f1']:>8.3f}"
              f"{result['connectivity_f1']['f1']:>8.3f}{result['wire_f1']['f1']:>8.3f}"
              f"{result['final_score']*100:>8.2f}")
    print({key: round(value / count, 4) for key, value in totals.items()},
          "total", round(sum(scores) / count * 100, 2))
