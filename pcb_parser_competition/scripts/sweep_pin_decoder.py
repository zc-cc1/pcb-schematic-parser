"""Sweep pin decoder settings (local-max kernel x threshold) without retraining.

The peak count is ~1.8x the true tip count, so the decoder may be splitting one
real pin into several nearby peaks.  A larger local-maximum kernel enforces
spacing but can also merge two genuinely adjacent pins, so both directions are
measured before any default changes.

Probability maps are cached per page so the sweep costs one inference pass.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.unet_model import WirePinUNet  # noqa: E402
from evaluate_unet_baseline import match, peaks  # noqa: E402
from evaluate_unet_v2 import predict  # noqa: E402

V2 = Path(r"D:/pcb_unet_v2/data/segmentation_unet_v2")
KERNELS = (5, 7, 9, 11)
THRESHOLDS = (0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:/pcb_unet_v2/runs/A2_full60/best.pt"))
    parser.add_argument("--pages", type=int, default=12)
    parser.add_argument("--cache", type=Path, default=Path(r"D:/pcb_unet_v2/reports/pin_probability_cache.npz"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/pin_decoder_sweep.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in (V2 / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    val = sorted([r for r in rows if r["split"] == "val"], key=lambda r: -r["stats"]["pin_tips"])[: args.pages]

    if args.cache.exists():
        cached = np.load(args.cache)
        maps = {case_id: cached[case_id] for case_id in cached.files}
        print(f"loaded {len(maps)} cached probability maps")
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = WirePinUNet(checkpoint["base_channels"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        maps = {}
        for row in val:
            with Image.open(row["image"]) as source:
                image = np.asarray(source.convert("RGB"))
            maps[row["case_id"]] = predict(model, image, device, 512, 384)[1].astype(np.float16)
            print(f"  predicted {row['case_id']}", flush=True)
        np.savez_compressed(args.cache, **maps)

    truth = {row["case_id"]: [tuple(p["point"]) for p in
                              json.loads(Path(row["pin_points"]).read_text(encoding="utf-8"))] for row in val}

    table = {}
    for kernel in KERNELS:
        for threshold in THRESHOLDS:
            tp = fp = fn = 0
            peaks_total = 0
            for case_id, heat in maps.items():
                heat32 = heat.astype(np.float32)
                found = peaks(heat32, threshold, kernel)
                peaks_total += len(found)
                hit, unmatched_pred, unmatched_truth = match(found, truth[case_id], 5.0)
                tp += hit
                fp += len(unmatched_pred)
                fn += len(unmatched_truth)
            table[f"k{kernel}_t{threshold}"] = {"kernel": kernel, "threshold": threshold,
                                                "peaks": peaks_total,
                                                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                                                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                                                "f1": round(f1(tp, fp, fn), 4), "tp": tp, "fp": fp, "fn": fn}

    best = max(table.items(), key=lambda kv: kv[1]["f1"])
    report = {"status": "pin_decoder_sweep", "pages": [r["case_id"] for r in val],
              "truth_tips": sum(len(v) for v in truth.values()), "table": table,
              "best": {"setting": best[0], **best[1]},
              "current_default": table.get("k5_t0.8")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ntrue tips {report['truth_tips']}; current default k5 t0.8 -> "
          f"P {report['current_default']['precision']:.4f} R {report['current_default']['recall']:.4f} "
          f"F1 {report['current_default']['f1']:.4f} ({report['current_default']['peaks']} peaks)")
    print("best:", best[0], json.dumps(best[1], ensure_ascii=False))
    print("\nF1 by kernel/threshold:")
    print("        " + "".join(f"t{threshold:<7.2f}" for threshold in THRESHOLDS))
    for kernel in KERNELS:
        line = f"  k{kernel:<4d}"
        for threshold in THRESHOLDS:
            value = table[f"k{kernel}_t{threshold}"]
            line += f"{value['f1']:.4f}   "
        print(line)
    print("written:", args.output)


if __name__ == "__main__":
    main()
