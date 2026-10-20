"""Full-split validation of pin decoder settings (kernel x threshold).

The 12 densest pages are the ones the decoder sweep already covered.  Memory of
20/20/20 hindsight is the risk here: the same 12 pages chose k9/t0.9, so the
choice has to be re-checked on pages that did not take part, and the whole
59-page split must be recomputed before any default changes.

Probability maps are cached for every page, so re-running is cheap.
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
DENSE12 = {"0107", "0080", "0170", "0001", "0141", "0016", "0117", "0278", "0101", "0124", "0251", "0075"}
SETTINGS = [(5, 0.80), (5, 0.90), (7, 0.90), (9, 0.85), (9, 0.90), (9, 0.92), (11, 0.90)]
BEST = (9, 0.90)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:/pcb_unet_v2/runs/A2_full60/best.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--cache", type=Path, default=Path(r"D:/pcb_unet_v2/reports/pin_probability_cache_all.npz"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/pin_decoder_full.json"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in (V2 / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["split"] == args.split]
    seeded = Path(r"D:/pcb_unet_v2/reports/pin_probability_cache.npz")
    maps = {}
    if seeded.exists():
        with np.load(seeded) as loaded:
            maps = {key: loaded[key] for key in loaded.files}
    if args.cache.exists():
        with np.load(args.cache) as loaded:
            maps.update({key: loaded[key] for key in loaded.files})

    missing = [r for r in rows if r["case_id"] not in maps]
    if missing:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = WirePinUNet(checkpoint["base_channels"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        for index, row in enumerate(missing, 1):
            with Image.open(row["image"]) as source:
                image = np.asarray(source.convert("RGB"))
            maps[row["case_id"]] = predict(model, image, device, 512, 384)[1].astype(np.float16)
            if index % 10 == 0:
                print(f"  predicted {index}/{len(missing)}", flush=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **maps)

    truth = {r["case_id"]: [tuple(p["point"]) for p in
                            json.loads(Path(r["pin_points"]).read_text(encoding="utf-8"))] for r in rows}

    def run(subset, kernel, threshold):
        tp = fp = fn = 0
        for case_id in subset:
            found = peaks(maps[case_id].astype(np.float32), threshold, kernel)
            hit, unmatched_pred, unmatched_truth = match(found, truth[case_id], 5.0)
            tp += hit
            fp += len(unmatched_pred)
            fn += len(unmatched_truth)
        return {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                "f1": round(f1(tp, fp, fn), 4), "tp": tp, "fp": fp, "fn": fn}

    all_ids = [r["case_id"] for r in rows]
    heldout = [c for c in all_ids if c not in DENSE12]
    report = {"status": "pin_decoder_full_validation", "pages": len(all_ids),
              "dense12": sorted(c for c in all_ids if c in DENSE12),
              "heldout_pages": len(heldout), "settings": {}}
    for kernel, threshold in SETTINGS:
        key = f"k{kernel}_t{threshold:.2f}"
        report["settings"][key] = {"all_pages": run(all_ids, kernel, threshold),
                                   "dense12": run([c for c in all_ids if c in DENSE12], kernel, threshold),
                                   "heldout": run(heldout, kernel, threshold)}
    current = "k5_t0.80"
    chosen = f"k{BEST[0]}_t{BEST[1]:.2f}"
    report["current_default"] = current
    report["proposed_default"] = chosen
    report["delta"] = {scope: round(report["settings"][chosen][scope]["f1"] - report["settings"][current][scope]["f1"], 4)
                       for scope in ("all_pages", "dense12", "heldout")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pages: {len(all_ids)} (dense12 {len(DENSE12 & set(all_ids))}, heldout {len(heldout)})")
    print(f"{'setting':<12}{'all P':>8}{'all R':>8}{'all F1':>8}   {'dense F1':>9}{'heldout F1':>11}")
    for key, value in report["settings"].items():
        print(f"{key:<12}{value['all_pages']['precision']:>8.4f}{value['all_pages']['recall']:>8.4f}"
              f"{value['all_pages']['f1']:>8.4f}   {value['dense12']['f1']:>9.4f}{value['heldout']['f1']:>11.4f}")
    print(f"\ndelta {chosen} vs {current}: " + ", ".join(f"{k} {v:+.4f}" for k, v in report["delta"].items()))
    print("written:", args.output)


if __name__ == "__main__":
    main()
