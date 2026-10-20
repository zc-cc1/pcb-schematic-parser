"""Score a submission directory on open_GT with the PLATFORM protocol.

Unlike `evaluate_open_gt.py` (our old key-only reference), this follows the protocol block
the platform prints in its own report: power symbols match geometrically ignoring key and
Name, PinF1 is IC-only, connectivity merges global power rails and matches nets one-to-one
with a soft vertex-set F1.  On the 7th-eval pipeline it reproduces the platform's component
figure to within 0.005 (0.7064 here vs 0.7108 reported), which is why it is the yardstick
to use from now on.

    python scripts/evaluate_open_gt_platform.py --predictions D:/pcb_open_gt_out/final_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from platform_metrics import evaluate_cases  # noqa: E402

GT_ROOT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")


def image_height(path: Path) -> float:
    from PIL import Image
    with Image.open(path) as handle:
        return float(handle.height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    pairs, inventory = [], []
    for folder in sorted((p for p in args.gt_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        case = folder.name
        truth = json.loads(next(folder.glob("*.json")).read_text(encoding="utf-8"))
        candidates = [args.predictions / f"{case}_Pred.json", args.predictions / f"{case}.json"]
        path = next((p for p in candidates if p.is_file()), candidates[0])
        pred = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
            {"components": {}, "pins": {}, "nets": {}}
        pairs.append((case, pred, truth, image_height(next(folder.glob("*.png")))))
        inventory.append({"case": case,
                          "predicted_components": len(pred.get("components") or {}),
                          "predicted_pins": sum(len(v) for v in (pred.get("pins") or {}).values()),
                          "predicted_nets": len(pred.get("nets") or {}),
                          "truth_components": len(truth.get("components") or {}),
                          "truth_nets": len(truth.get("nets") or {})})

    report = evaluate_cases(pairs)
    report["predictions"] = str(args.predictions)
    report["inventory"] = inventory

    header = (f"{'case':>4}{'C tp':>6}{'C fp':>6}{'C fn':>6}{'C-F1':>7}{'P tp':>6}{'P fp':>6}{'P fn':>6}"
              f"{'P-F1':>7}{'K tp':>6}{'K fp':>6}{'K fn':>6}{'K-F1':>7}{'W-F1':>7}{'score':>8}")
    print(header)
    print("-" * len(header))
    for entry in report["per_case"]:
        c, p, k, w = entry["component_f1"], entry["pin_f1"], entry["connectivity_f1"], entry["wire_f1"]
        print(f"{entry['case_id']:>4}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}{c['f1']:>7.3f}"
              f"{p['tp']:>6}{p['fp']:>6}{p['fn']:>6}{p['f1']:>7.3f}"
              f"{k['tp']:>6}{k['fp']:>6}{k['fn']:>6}{k['f1']:>7.3f}{w['f1']:>7.3f}{entry['final_score']:>8.3f}")
    m = report["metrics"]
    print("-" * len(header))
    print(f"{'ALL':>4}{m['component_f1']['tp']:>6}{m['component_f1']['fp']:>6}{m['component_f1']['fn']:>6}"
          f"{m['component_f1']['f1']:>7.3f}{m['pin_f1']['tp']:>6}{m['pin_f1']['fp']:>6}{m['pin_f1']['fn']:>6}"
          f"{m['pin_f1']['f1']:>7.3f}{m['connectivity_f1']['tp']:>6}{m['connectivity_f1']['fp']:>6}"
          f"{m['connectivity_f1']['fn']:>6}{m['connectivity_f1']['f1']:>7.3f}{m['wire_f1']['f1']:>7.3f}"
          f"{report['final_score']:>8.3f}")
    print(f"\nfinal_score {report['final_score']:.4f} on a 0-1 scale "
          f"(= {report['final_score'] * 100:.2f} points), platform protocol, not the platform itself")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", args.output)


if __name__ == "__main__":
    main()
