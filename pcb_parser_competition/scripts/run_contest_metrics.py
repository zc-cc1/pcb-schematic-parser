"""CLI for the local contest-metric reference implementation.

Two jobs:

  self-test   score the Golden files against themselves; every metric must be 1.0
              or the implementation (not the data) is wrong
  score       score pipeline output against Golden / reference JSON

Never claim the printed numbers as official.  `--official` does not exist on
purpose.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.contest_metrics import evaluate_cases  # noqa: E402

DATASET300 = Path(r"C:/Users/Pc/Desktop/新建文件夹/Dataset300")
GT_DIR = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/10_GTcase")


def golden_cases(use_dataset300_reference=False):
    """Yield (case_id, truth_payload, height)."""
    if use_dataset300_reference:
        for folder in sorted((ROOT / "data/standardized").iterdir()):
            if not folder.is_dir():
                continue
            payload = folder / f"{folder.name}.json"
            if not payload.exists():
                continue
            data = json.loads(payload.read_text(encoding="utf-8"))
            height = max((component.get("bbox") or [0, 0, 0, 0])[3] for component in data.get("components", {}).values()) \
                if data.get("components") else 0
            yield folder.name, data, height
        return
    if not GT_DIR.exists():
        raise SystemExit(f"Golden directory not found: {GT_DIR}")
    for folder in sorted(GT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        case_id = folder.name.split("-")[0]
        target = next((p for p in folder.glob("*_target.json")), None)
        if target is None:
            continue
        data = json.loads(target.read_text(encoding="utf-8"))
        image = next(folder.glob("*.png"))
        from PIL import Image
        with Image.open(image) as rendered:
            height = rendered.height
        yield case_id, data, height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "score"), default="self-test")
    parser.add_argument("--predictions", type=Path, help="directory of <case>.json predictions, or a single file")
    parser.add_argument("--predictions-are-top-left", action="store_true",
                        help="convert predicted bboxes from top-left to bottom-left origin")
    parser.add_argument("--cases", nargs="*", help="restrict to these case ids")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/contest_metrics.json"))
    args = parser.parse_args()

    pairs = []
    for case_id, truth, height in golden_cases():
        if args.cases and case_id not in set(args.cases):
            continue
        if args.mode == "self-test":
            # deep copy: the scorer mutates/reads the same structure and self-scoring
            # must not share objects between prediction and truth
            pairs.append((case_id, json.loads(json.dumps(truth)), truth, height))
        else:
            if not args.predictions:
                raise SystemExit("--predictions is required unless --mode self-test")
            candidate = args.predictions if args.predictions.is_file() else args.predictions / f"{case_id}.json"
            if not candidate.exists():
                pairs.append((case_id, {"components": {}, "pins": {}, "nets": {}}, truth, height))
                continue
            prediction = json.loads(candidate.read_text(encoding="utf-8"))
            pairs.append((case_id, prediction, truth, height))

    report = evaluate_cases(pairs)  # Golden and pipeline outputs share top-left pixel coordinates
    report["mode"] = args.mode
    report["cases"] = [pair[0] for pair in pairs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print(f"mode={args.mode}  cases={report['cases'] and len(report['cases'])}")
    for key in ("component", "pin", "net_hypergraph", "net_line"):
        value = metrics[key]
        print(f"  {key:<15} P {value['precision']:.4f}  R {value['recall']:.4f}  F1 {value['f1']:.4f}"
              f"   (tp {value['tp']} fp {value['fp']} fn {value['fn']})")
    print(f"  weighted total  {report['weighted_total']:.4f}   [LOCAL REFERENCE, NOT OFFICIAL]")
    if args.mode == "self-test":
        worst = min(metrics[key]["f1"] for key in metrics)
        print(f"  self-test verdict: {'PASS' if worst > 0.999 else 'FAIL'} (worst metric {worst:.4f})")
    print("written:", args.output)


if __name__ == "__main__":
    main()
