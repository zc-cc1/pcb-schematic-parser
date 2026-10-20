"""Score a submission directory under the NEW rules (2026-10-18), macro-averaged.

The platform reports per-case scores plus a `macro_avg` of each item; its final score is the
mean of the per-case scores.  This script reproduces that shape (the older
`evaluate_open_gt_platform.py` micro-aggregates tp/fp/fn and therefore drifts a few tenths).

    python scripts/evaluate_open_gt_v2.py --predictions D:/pcb_open_gt_v22/final_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import platform_metrics_v2 as metrics  # noqa: E402

GT_ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")
ITEMS = ("component_f1", "pin_f1", "connectivity_f1", "wire_f1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows, totals, scores = [], {key: 0.0 for key in ITEMS}, []
    for folder in sorted((p for p in args.gt_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        path = args.predictions / f"{folder.name}_Pred.json"
        pred = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"components": {}, "pins": {}, "nets": {}}
        result = metrics.score_case(pred, truth)
        rows.append((folder.name, result))
        scores.append(result["final_score"])
        for key in ITEMS:
            totals[key] += result[key]["f1"]

    header = f"{'case':>4}{'comp':>8}{'pin':>8}{'conn':>8}{'wire':>8}{'score':>8}"
    print(header)
    print("-" * len(header))
    for case, result in rows:
        print(f"{case:>4}{result['component_f1']['f1']:>8.3f}{result['pin_f1']['f1']:>8.3f}"
              f"{result['connectivity_f1']['f1']:>8.3f}{result['wire_f1']['f1']:>8.3f}"
              f"{result['final_score']*100:>8.2f}")
    count = max(len(rows), 1)
    macro = {key: totals[key] / count for key in ITEMS}
    print("-" * len(header))
    print(f"{'ALL':>4}{macro['component_f1']:>8.3f}{macro['pin_f1']:>8.3f}"
          f"{macro['connectivity_f1']:>8.3f}{macro['wire_f1']:>8.3f}"
          f"{sum(scores)/count*100:>8.2f}")
    print("\nfinal_score "
          f"{0.30*macro['component_f1'] + 0.25*macro['pin_f1'] + 0.35*macro['connectivity_f1'] + 0.10*macro['wire_f1']:.4f}"
          " (new rules: 20 px for components, IC pins and wire endpoints)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"macro": macro, "per_case": [
            {"case": case, **{key: result[key] for key in ITEMS}, "final_score": result["final_score"]}
            for case, result in rows]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", args.output)


if __name__ == "__main__":
    main()
