"""Zero-training baseline: score the existing v1 checkpoint on v1 and v2 labels.

No gradient step, no checkpoint write, no label write.  Produces the comparison
table needed before any retraining decision:

  * wire pixel P/R/F1 at 0.5 on v1 labels vs v2 labels (v2 = +692 recovered edges)
  * pin point-set P/R/F1 at 5 px tolerance, decoded exactly like the accepted v2
    decoder (0.8 threshold, 5x5 local maxima, quadratic sub-pixel)
  * pin peaks that land in an ignore region are dropped and counted, not blamed
  * per-page rows so the dense pages stay visible instead of being averaged away

The same probability maps feed every column, so differences are label-driven.
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
from pcb_parser.unet_model import WirePinUNet  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "runs/training/unet_wire_pin_v1/best.pt"
V1_DATA = ROOT / "data/segmentation_unet_v1"
V2_DATA = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def tile_starts(length, size, stride):
    return [0] if length <= size else sorted(set(list(range(0, length - size + 1, stride)) + [length - size]))


@torch.inference_mode()
def predict(model, image, device, tile=512, stride=384):
    height, width = image.shape[:2]
    output = np.zeros((2, height, width), np.float32)
    count = np.zeros((height, width), np.float32)
    model.eval()
    for y in tile_starts(height, tile, stride):
        for x in tile_starts(width, tile, stride):
            h, w = min(tile, height - y), min(tile, width - x)
            crop = image[y:y + h, x:x + w]
            padded = np.full((tile, tile, 3), 255, np.uint8)
            padded[:h, :w] = crop
            batch = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probability = model(batch).sigmoid()[0, :, :h, :w].float().cpu().numpy()
            output[:, y:y + h, x:x + w] += probability
            count[y:y + h, x:x + w] += 1
    return output / count[None]


def peaks(heat, threshold, kernel=5):
    maxima = ndimage.maximum_filter(heat, size=kernel, mode="constant", cval=0)
    candidates = (heat == maxima) & (heat >= threshold)
    labels, count = ndimage.label(candidates, structure=np.ones((3, 3), bool))
    points = []
    pad = kernel // 2
    for y, x in ndimage.center_of_mass(candidates, labels, range(1, count + 1)):
        yi, xi = int(round(y)), int(round(x))
        if not (pad <= yi < heat.shape[0] - pad and pad <= xi < heat.shape[1] - pad):
            points.append((float(x), float(y)))
            continue
        logp = np.log(np.clip(heat[yi - 1:yi + 2, xi - 1:xi + 2], 1e-9, None))
        denominator_x = logp[1, 0] - 2 * logp[1, 1] + logp[1, 2]
        denominator_y = logp[0, 1] - 2 * logp[1, 1] + logp[2, 1]
        dx = 0.0 if abs(denominator_x) < 1e-9 else 0.5 * (logp[1, 0] - logp[1, 2]) / denominator_x
        dy = 0.0 if abs(denominator_y) < 1e-9 else 0.5 * (logp[0, 1] - logp[2, 1]) / denominator_y
        points.append((x + dx, y + dy))
    return points


def match(pred, truth, tolerance):
    if not pred or not truth:
        return 0, list(range(len(pred))), list(range(len(truth)))
    distances, indices = cKDTree(np.asarray(truth)).query(np.asarray(pred), k=1)
    used_truth, used_pred, tp = set(), set(), 0
    for i in np.argsort(distances):
        j = int(indices[i])
        if distances[i] <= tolerance and j not in used_truth:
            used_truth.add(j)
            used_pred.add(int(i))
            tp += 1
    return (tp,
            [i for i in range(len(pred)) if i not in used_pred],
            [j for j in range(len(truth)) if j not in used_truth])


def truth_points(case_id, dataset):
    return [tuple(p["point"]) for p in json.loads((dataset / case_id / "pin_points.json").read_text(encoding="utf-8"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--v2-manifest", type=Path, default=V2_DATA / "manifest.jsonl")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wire-threshold", type=float, default=0.5)
    parser.add_argument("--pin-threshold", type=float, default=0.8)
    parser.add_argument("--pin-tolerance", type=float, default=5.0)
    parser.add_argument("--cases", nargs="*", help="optional case list (e.g. the 26 legacy pages)")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/baseline_e0.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"checkpoint epoch {checkpoint['epoch']} kind {checkpoint['kind']} -> {device}", flush=True)

    rows = [json.loads(line) for line in args.v2_manifest.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == args.split]
    if args.cases:
        rows = [r for r in rows if r["case_id"] in set(args.cases)]
    legacy = {r["case_id"] for r in rows if (V1_DATA / r["case_id"] / "wire_mask.png").exists()}

    per_case = []
    for index, row in enumerate(rows, 1):
        case_id = row["case_id"]
        with Image.open(row["image"]) as source:
            image = np.asarray(source.convert("RGB"))
        probability = predict(model, image, device)
        entry = {"case_id": case_id, "legacy_v1_page": case_id in legacy,
                 "in_legacy_26": case_id in LEGACY_26}

        wire_pred = probability[0] >= args.wire_threshold
        structure = ndimage.generate_binary_structure(2, 2)
        for tag, mask_path in (("v1", V1_DATA / case_id / "wire_mask.png"),
                               ("v2", Path(row["wire_mask"]))):
            if not mask_path.exists():
                entry[f"wire_{tag}"] = None
                continue
            truth = np.asarray(Image.open(mask_path)) > 0
            entry[f"wire_{tag}"] = prf(int((wire_pred & truth).sum()),
                                       int((wire_pred & ~truth).sum()),
                                       int((truth & ~wire_pred).sum()))
            # same prediction, 2 px slack: separates stroke-width error from model error
            truth_d = ndimage.binary_dilation(truth, structure, iterations=2)
            pred_d = ndimage.binary_dilation(wire_pred, structure, iterations=2)
            entry[f"wire_{tag}_tol2"] = prf(int((wire_pred & truth_d).sum()),
                                            int((wire_pred & ~truth_d).sum()),
                                            int((truth & ~pred_d).sum()))

        # Did this page's supervision actually change between v1 and v2?
        v1_mask_path = V1_DATA / case_id / "wire_mask.png"
        if v1_mask_path.exists() and Path(row["wire_mask"]).exists():
            before = np.asarray(Image.open(v1_mask_path)) > 0
            after = np.asarray(Image.open(row["wire_mask"])) > 0
            entry["mask_added_px"] = int((after & ~before).sum())
            entry["mask_lost_px"] = int((before & ~after).sum())
            entry["mask_changed"] = bool(entry["mask_added_px"] or entry["mask_lost_px"])
        else:
            entry["mask_added_px"] = entry["mask_lost_px"] = None
            entry["mask_changed"] = None

        ignore = np.load(row["ignore_masks"])
        pin_ignore = ignore["pin_ignore"] > 0
        candidates = peaks(probability[1], args.pin_threshold)
        kept, ignored_peaks = [], 0
        for x, y in candidates:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < pin_ignore.shape[0] and 0 <= xi < pin_ignore.shape[1] and pin_ignore[yi, xi]:
                ignored_peaks += 1
                continue
            kept.append((x, y))
        entry["pin_peaks_ignored"] = ignored_peaks

        for tag, dataset in (("v1", V1_DATA), ("v2", V2_DATA)):
            path = dataset / case_id / "pin_points.json"
            if not path.exists():
                entry[f"pin_{tag}"] = None
                continue
            # v2 ground truth already excludes unowned root pins; v1 counted them as
            # real pins, so scoring against the v2 ignore list is the fair pairing
            truth = [tuple(p["point"]) for p in json.loads(path.read_text(encoding="utf-8"))]
            tp, fp, fn = match(kept, truth, args.pin_tolerance)
            entry[f"pin_{tag}"] = prf(tp, len(fp), len(fn))
        per_case.append(entry)
        if index % 10 == 0:
            print(f"  {index}/{len(rows)} pages", flush=True)

    def aggregate(key):
        total = {"tp": 0, "fp": 0, "fn": 0}
        for row in per_case:
            if row.get(key):
                for metric in total:
                    total[metric] += row[key][metric]
        return prf(total["tp"], total["fp"], total["fn"]) if total["tp"] + total["fp"] + total["fn"] else None

    def subset(predicate, key):
        rows_ = [r for r in per_case if predicate(r)]
        total = {"tp": 0, "fp": 0, "fn": 0}
        for row in rows_:
            if row.get(key):
                for metric in total:
                    total[metric] += row[key][metric]
        return (prf(total["tp"], total["fp"], total["fn"]), len(rows_)) if total["tp"] + total["fp"] + total["fn"] else (None, len(rows_))

    report = {
        "status": "zero_training_baseline",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_kind": checkpoint["kind"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "device": str(device),
        "split": args.split,
        "pages": len(per_case),
        "config": {"wire_threshold": args.wire_threshold, "pin_threshold": args.pin_threshold,
                   "pin_tolerance_px": args.pin_tolerance, "tile": 512, "stride": 384},
        "wire": {key: aggregate(f"wire_{key}") for key in ("v1", "v1_tol2", "v2", "v2_tol2")},
        "pin": {key: aggregate(f"pin_{key}") for key in ("v1", "v2")},
        "pin_peaks_ignored_total": sum(r["pin_peaks_ignored"] for r in per_case),
        "label_change": {
            "pages_with_changed_wire_supervision": sum(1 for r in per_case if r["mask_changed"]),
            "pages_unchanged": sum(1 for r in per_case if r["mask_changed"] is False),
            "added_px": sum(r["mask_added_px"] or 0 for r in per_case),
            "lost_px": sum(r["mask_lost_px"] or 0 for r in per_case),
        },
        "legacy_26_pages": {
            "count": sum(1 for r in per_case if r["in_legacy_26"]),
            "wire_v1": subset(lambda r: r["in_legacy_26"], "wire_v1"),
            "wire_v2": subset(lambda r: r["in_legacy_26"], "wire_v2"),
            "pin_v1": subset(lambda r: r["in_legacy_26"], "pin_v1"),
            "pin_v2": subset(lambda r: r["in_legacy_26"], "pin_v2"),
        },
        "pages_with_new_supervision": {
            "count": sum(1 for r in per_case if r["mask_changed"]),
            "wire_v2": subset(lambda r: r["mask_changed"], "wire_v2"),
            "pin_v2": subset(lambda r: r["mask_changed"], "pin_v2"),
        },
        "pages_unchanged": {
            "count": sum(1 for r in per_case if r["mask_changed"] is False),
            "wire_v1": subset(lambda r: r["mask_changed"] is False, "wire_v1"),
            "wire_v2": subset(lambda r: r["mask_changed"] is False, "wire_v2"),
            "pin_v2": subset(lambda r: r["mask_changed"] is False, "pin_v2"),
        },
        "per_case": per_case,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== wire (same predictions, two label sets) ==")
    for key, value in report["wire"].items():
        print(f"  {key:<10} P {value['precision']:.4f}  R {value['recall']:.4f}  F1 {value['f1']:.4f}")
    print("== pin point set @ %.0f px ==" % args.pin_tolerance)
    for key, value in report["pin"].items():
        print(f"  {key:<10} P {value['precision']:.4f}  R {value['recall']:.4f}  F1 {value['f1']:.4f}"
              f"  TP {value['tp']} FP {value['fp']} FN {value['fn']}")
    print(f"  peaks dropped inside ignore regions: {report['pin_peaks_ignored_total']}")
    change = report["label_change"]
    print(f"== label change impact: {change['pages_with_changed_wire_supervision']} pages gained supervision "
          f"(+{change['added_px']} px, -{change['lost_px']} px), {change['pages_unchanged']} unchanged ==")
    print(f"== the 26 legacy pages ({report['legacy_26_pages']['count']}) ==")
    for key in ("wire_v1", "wire_v2", "pin_v1", "pin_v2"):
        value, count = report["legacy_26_pages"][key]
        if value:
            print(f"  {key:<10} F1 {value['f1']:.4f}  (P {value['precision']:.4f} R {value['recall']:.4f})")
    print(f"== pages that gained supervision ({report['pages_with_new_supervision']['count']}) ==")
    for key in ("wire_v2", "pin_v2"):
        value, count = report["pages_with_new_supervision"][key]
        if value:
            print(f"  {key:<10} F1 {value['f1']:.4f}  (P {value['precision']:.4f} R {value['recall']:.4f}) over {count} pages")
    print(f"== pages with unchanged supervision ({report['pages_unchanged']['count']}) ==")
    for key in ("wire_v1", "wire_v2", "pin_v2"):
        value, count = report["pages_unchanged"][key]
        if value:
            print(f"  {key:<10} F1 {value['f1']:.4f}  (P {value['precision']:.4f} R {value['recall']:.4f}) over {count} pages")
    print("written:", args.output)


LEGACY_26 = {"0002", "0007", "0012", "0018", "0022", "0026", "0029", "0033", "0036", "0042", "0049",
             "0096", "0105", "0131", "0148", "0149", "0170", "0208", "0213", "0227", "0236", "0240",
             "0264", "0267", "0286", "0287"}

if __name__ == "__main__":
    main()
