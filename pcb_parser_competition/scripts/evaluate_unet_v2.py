"""Score any U-Net checkpoint on the v2 validation split under the accepted protocol.

Used for the A2 decision: same pages, same thresholds, same decoder as
`scripts/evaluate_unet_baseline.py`, so an A2 number is directly comparable to
the E0 baseline (wire 0.7439 / pin 0.6661 on the 59 pages).

Differences between the two checkpoints are the only variable: the loader, the
tile geometry, the peak decoder and the ground truth all come from one place.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.unet_model import WirePinUNet  # noqa: E402
from pcb_parser.unet_split_model import SplitWirePinUNet  # noqa: E402
from evaluate_unet_baseline import LEGACY_26, match, peaks, prf, tile_starts  # noqa: E402

MODELS = {"shared": WirePinUNet, "split": SplitWirePinUNet}
V1_DATA = ROOT / "data/segmentation_unet_v1"
V2_DATA = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")


@torch.inference_mode()
def predict(model, image, device, tile, stride):
    height, width = image.shape[:2]
    output = np.zeros((2, height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    model.eval()
    for y in tile_starts(height, tile, stride):
        for x in tile_starts(width, tile, stride):
            h, w = min(tile, height - y), min(tile, width - x)
            crop = np.full((tile, tile, 3), 255, np.uint8)
            crop[:h, :w] = image[y:y + h, x:x + w]
            batch = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output[:, y:y + h, x:x + w] += model(batch).sigmoid()[0, :, :h, :w].float().cpu().numpy()
            count[y:y + h, x:x + w] += 1
    return output / count[None]


def evaluate(checkpoint_path, rows, device, wire_threshold, pin_threshold, pin_tolerance):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_class = MODELS.get(checkpoint.get("model_variant", "shared"), WirePinUNet)
    model = model_class(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    per_case = []
    for row in rows:
        case_id = row["case_id"]
        with Image.open(row["image"]) as source:
            image = np.asarray(source.convert("RGB"))
        probability = predict(model, image, device, 512, 384)
        predicted = probability[0] >= wire_threshold
        truth = np.asarray(Image.open(row["wire_mask"])) > 0
        entry = {"case_id": case_id,
                 "legacy_26": case_id in LEGACY_26,
                 "wire": prf(int((predicted & truth).sum()), int((predicted & ~truth).sum()),
                             int((truth & ~predicted).sum()))}
        structure = ndimage.generate_binary_structure(2, 2)
        truth_d = ndimage.binary_dilation(truth, structure, iterations=2)
        pred_d = ndimage.binary_dilation(predicted, structure, iterations=2)
        entry["wire_tol2"] = prf(int((predicted & truth_d).sum()), int((predicted & ~truth_d).sum()),
                                 int((truth & ~pred_d).sum()))
        before = V1_DATA / case_id / "wire_mask.png"
        if before.exists():
            baseline = np.asarray(Image.open(before)) > 0
            entry["mask_changed"] = bool(int((truth & ~baseline).sum()))
        else:
            entry["mask_changed"] = None
        with np.load(row["ignore_masks"]) as ignore:
            pin_ignore = ignore["pin_ignore"] > 0
        candidates = []
        for x, y in peaks(probability[1], pin_threshold, 5):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < pin_ignore.shape[0] and 0 <= xi < pin_ignore.shape[1] and pin_ignore[yi, xi]:
                continue
            candidates.append((x, y))
        tips = [tuple(p["point"]) for p in json.loads(Path(row["pin_points"]).read_text(encoding="utf-8"))]
        tp, unmatched_pred, unmatched_truth = match(candidates, tips, pin_tolerance)
        entry["pin"] = prf(tp, len(unmatched_pred), len(unmatched_truth))
        per_case.append(entry)
    return {"checkpoint": str(checkpoint_path), "epoch": checkpoint["epoch"], "kind": checkpoint["kind"],
            "init": checkpoint.get("init"), "per_case": per_case}


def aggregate(per_case, key, predicate=lambda row: True, field=None):
    total = {"tp": 0, "fp": 0, "fn": 0}
    rows = [r for r in per_case if predicate(r)]
    for row in rows:
        value = row[key] if field is None else row[key][field]
        for metric in total:
            total[metric] += value[metric]
    if not sum(total.values()):
        return None
    return {**prf(total["tp"], total["fp"], total["fn"]), "pages": len(rows)}


def summarize(result):
    cases = result["per_case"]
    import statistics
    return {
        "wire": aggregate(cases, "wire"),
        "wire_tol2": aggregate(cases, "wire_tol2"),
        "pin": aggregate(cases, "pin"),
        "median_page_f1": {
            "wire": round(statistics.median([r["wire"]["f1"] for r in cases]), 4),
            "pin": round(statistics.median([r["pin"]["f1"] for r in cases]), 4),
            "note": "unweighted per-page median; the micro average is dominated by the "
                    "top-10 pages, which carry 43.6% of the validation wire pixels",
        },
        "legacy_26": {"wire": aggregate(cases, "wire", lambda r: r["legacy_26"]),
                      "pin": aggregate(cases, "pin", lambda r: r["legacy_26"])},
        "changed_pages": {"pages": sum(1 for r in cases if r["mask_changed"]),
                          "wire": aggregate(cases, "wire", lambda r: r["mask_changed"]),
                          "pin": aggregate(cases, "pin", lambda r: r["mask_changed"])},
        "unchanged_pages": {"pages": sum(1 for r in cases if r["mask_changed"] is False),
                            "wire": aggregate(cases, "wire", lambda r: r["mask_changed"] is False),
                            "pin": aggregate(cases, "pin", lambda r: r["mask_changed"] is False)},
    }


def run_evaluation(checkpoint, rows, device, wire_threshold, pin_threshold, pin_tolerance, baseline=None):
    result = evaluate(checkpoint, rows, device, wire_threshold, pin_threshold, pin_tolerance)
    summary = summarize(result)
    report = {"status": "v2_validation_evaluation", "pages": len(rows),
              "config": {"wire_threshold": wire_threshold, "pin_threshold": pin_threshold,
                         "pin_tolerance_px": pin_tolerance, "tile": 512, "stride": 384},
              "checkpoint": result["checkpoint"], "epoch": result["epoch"], "kind": result["kind"],
              "init": result["init"], "metrics": summary, "per_case": result["per_case"]}
    if baseline is not None:
        report["baseline_comparison"] = {
            "baseline_checkpoint": baseline["checkpoint"],
            "wire_before": baseline["wire"]["v2"], "wire_after": summary["wire"],
            "pin_before": baseline["pin"]["v2"], "pin_after": summary["pin"],
            "wire_delta": round(summary["wire"]["f1"] - baseline["wire"]["v2"]["f1"], 4),
            "pin_delta": round(summary["pin"]["f1"] - baseline["pin"]["v2"]["f1"], 4),
            "changed_pages_wire_before": baseline["pages_with_new_supervision"]["wire_v2"],
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(r"D:/pcb_unet_v2/reports/baseline_e0.json"))
    parser.add_argument("--manifest", type=Path, default=V2_DATA / "manifest.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wire-threshold", type=float, default=0.5)
    parser.add_argument("--pin-threshold", type=float, default=0.8)
    parser.add_argument("--pin-tolerance", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == "val"]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else None
    report = run_evaluation(args.checkpoint, rows, device, args.wire_threshold, args.pin_threshold,
                            args.pin_tolerance, baseline)
    output = args.output or Path(r"D:/pcb_unet_v2/reports") / f"eval_{args.checkpoint.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["metrics"]
    print(f"checkpoint epoch {report['epoch']} init {report['init']}")
    print(f"  wire     P {summary['wire']['precision']:.4f} R {summary['wire']['recall']:.4f} F1 {summary['wire']['f1']:.4f}")
    print(f"  wire t2  P {summary['wire_tol2']['precision']:.4f} R {summary['wire_tol2']['recall']:.4f} F1 {summary['wire_tol2']['f1']:.4f}")
    print(f"  pin      P {summary['pin']['precision']:.4f} R {summary['pin']['recall']:.4f} F1 {summary['pin']['f1']:.4f}")
    print(f"  median per-page F1: wire {summary['median_page_f1']['wire']:.4f}  pin {summary['median_page_f1']['pin']:.4f}")
    for name in ("legacy_26", "changed_pages", "unchanged_pages"):
        block = summary[name]
        print(f"  {name:<15} wire F1 {block['wire']['f1']:.4f}  pin F1 {block['pin']['f1']:.4f}  ({block['wire']['pages']} pages)")
    if "baseline_comparison" in report:
        comparison = report["baseline_comparison"]
        print(f"  vs E0: wire {comparison['wire_before']['f1']:.4f} -> {comparison['wire_after']['f1']:.4f} "
              f"({comparison['wire_delta']:+.4f}), pin {comparison['pin_before']['f1']:.4f} -> "
              f"{comparison['pin_after']['f1']:.4f} ({comparison['pin_delta']:+.4f})")
    print("written:", output)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
