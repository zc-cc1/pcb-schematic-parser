"""Score a submission directory against the released open_GT (13 cases).

The 13 pages in `open_GT/open_GT/<n>/<n>.json` are the official evaluation set, so this
script is the yardstick that matters: it replaces the 10-page Golden proxy that earlier
reports used.  Everything it prints is still the *local reference implementation* of the
contest metrics (`contest_metrics.py`), not the platform's score.

Cases with no prediction file are scored as empty and stay in the denominator (R17).

    python scripts/evaluate_open_gt.py --predictions D:/pcb_open_gt_out/final_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))
from open_gt_metrics import evaluate_cases  # noqa: E402  pinned copy, see open_gt_metrics.py

GT_ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


def load_frozen_metrics(path: Path):
    """Load a bytecode snapshot of contest_metrics so a number stays reproducible.

    `contest_metrics.py` is being revised while this runs; a `.pyc` taken at a known time
    pins the rules that produced a reported score.  Diagnosis only - a score quoted from a
    snapshot must name the snapshot.
    """
    import importlib.util
    from importlib.machinery import SourcelessFileLoader
    loader = SourcelessFileLoader("frozen_contest_metrics", str(path))
    spec = importlib.util.spec_from_loader("frozen_contest_metrics", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.evaluate_cases


def image_height(path: Path) -> float:
    from PIL import Image
    with Image.open(path) as handle:
        return float(handle.height)


def load_cases(gt_root: Path):
    cases = []
    for folder in sorted((p for p in gt_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        truth_path = next(folder.glob("*.json"))
        image_path = next(folder.glob("*.png"))
        cases.append({"case": folder.name,
                      "truth": json.loads(truth_path.read_text(encoding="utf-8")),
                      "height": image_height(image_path)})
    return cases


def find_prediction(prediction_dir: Path, case: str):
    for pattern in (f"{case}_Pred.json", f"{case}.json"):
        candidate = prediction_dir / pattern
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {"components": {}, "pins": {}, "nets": {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prediction-is-top-left", action="store_true",
                        help="set when the prediction file keeps bboxes in image coordinates")
    parser.add_argument("--flip-reference", action="store_true",
                        help="set when the reference boxes are bottom-left origin")
    parser.add_argument("--frozen-metrics", type=Path, default=None,
                        help="bytecode snapshot of contest_metrics to score with instead of the live module")
    args = parser.parse_args()

    evaluate = evaluate_cases if args.frozen_metrics is None else load_frozen_metrics(args.frozen_metrics)
    cases = load_cases(args.gt_root)
    pairs, inventory = [], []
    for case in cases:
        prediction = find_prediction(args.predictions, case["case"])
        pairs.append((case["case"], prediction, case["truth"], case["height"]))
        inventory.append({"case": case["case"],
                          "predicted_components": len(prediction.get("components") or {}),
                          "predicted_pins": sum(len(v) for v in (prediction.get("pins") or {}).values()),
                          "predicted_nets": len(prediction.get("nets") or {}),
                          "truth_components": len(case["truth"].get("components") or {}),
                          "truth_pins": sum(len(v) for v in (case["truth"].get("pins") or {}).values()),
                          "truth_nets": len(case["truth"].get("nets") or {})})

    report = evaluate(pairs, prediction_is_top_left=args.prediction_is_top_left,
                      flip_reference=args.flip_reference)
    report["status"] = "local_reference_metrics_not_official"
    report["predictions"] = str(args.predictions)
    report["inventory"] = inventory

    header = (f"{'case':>4}{'predC':>7}{'C-F1':>7}{'TP':>4}{'FP':>4}{'FN':>4}{'P-F1':>7}{'TP':>4}"
              f"{'FP':>4}{'FN':>4}{'N-F1':>7}{'TP':>4}{'FP':>4}{'FN':>4}")
    print(header)
    print("-" * len(header))
    for entry in report["per_case"]:
        inv = next(i for i in inventory if i["case"] == entry["case_id"])
        c, p, n = entry["component"], entry["pin"], entry["net_hypergraph"]
        print(f"{entry['case_id']:>4}{inv['predicted_components']:>7}{c['f1']:>7.3f}"
              f"{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{p['f1']:>7.3f}{p['tp']:>4}{p['fp']:>4}{p['fn']:>4}"
              f"{n['f1']:>7.3f}{n['tp']:>4}{n['fp']:>4}{n['fn']:>4}")
    print("-" * len(header))
    totals = report["metrics"]
    print(f"{'ALL':>4}{sum(i['predicted_components'] for i in inventory):>7}"
          f"{totals['component']['f1']:>7.3f}{totals['component']['tp']:>4}{totals['component']['fp']:>4}"
          f"{totals['component']['fn']:>4}{totals['pin']['f1']:>7.3f}{totals['pin']['tp']:>4}"
          f"{totals['pin']['fp']:>4}{totals['pin']['fn']:>4}{totals['net_hypergraph']['f1']:>7.3f}"
          f"{totals['net_hypergraph']['tp']:>4}{totals['net_hypergraph']['fp']:>4}"
          f"{totals['net_hypergraph']['fn']:>4}")
    print(f"\nnet_line F1 {totals['net_line']['f1']:.4f} "
          f"(tp {totals['net_line']['tp']} fp {totals['net_line']['fp']} fn {totals['net_line']['fn']})")
    print(f"weighted total {report['weighted_total']:.4f}   [local reference, not official]")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", args.output)


if __name__ == "__main__":
    main()
