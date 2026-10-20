"""Side-by-side evaluation of the U-Net heads across A2/A3 variants.

Each checkpoint is scored on the same 59 validation pages with the same decoder,
so the table isolates what changed (wire supervision, data, loss weight, target
shape, decoder).  Two decoders are reported for the pin channel: the pre-A3
default (k5/t0.80) and the adopted one (k9/t0.90), because a model trained with a
different pin objective may prefer a different decoding.

Reads checkpoints only; writes one JSON + a printed table.
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
from evaluate_unet_baseline import match, peaks  # noqa: E402
from evaluate_unet_v2 import MODELS, V2_DATA, predict  # noqa: E402

V2 = V2_DATA

DECODERS = {"k5_t0.80": (5, 0.80), "k9_t0.90": (9, 0.90)}


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/reports/a3_comparison.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:/pcb_unet_v2/reports/prob_cache"))
    args = parser.parse_args()

    entries = {
        "A2 (v2 data, pin w0.5)": Path(r"D:/pcb_unet_v2/runs/A2_full60/best.pt"),
        "A3-pin_l1 (pin w1.0)": Path(r"D:/pcb_unet_v2/runs/A3_pin_l1/best.pt"),
        "A3-disk (disk target, w1.0)": Path(r"D:/pcb_unet_v2/runs/A3_pin_disk/best.pt"),
        "A3-split (split decoder)": Path(r"D:/pcb_unet_v2/runs/A3_pin_split/best.pt"),
        "A3-split+w2 (split, pin w2.0)": Path(r"D:/pcb_unet_v2/runs/A3_split_w2/best.pt"),
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = [json.loads(line) for line in (V2 / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    val = [r for r in rows if r["split"] == "val"]
    truth_pin = {r["case_id"]: [tuple(p["point"]) for p in
                                json.loads(Path(r["pin_points"]).read_text(encoding="utf-8"))] for r in val}
    truth_wire = {r["case_id"]: np.asarray(Image.open(r["wire_mask"])) > 0 for r in val}
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    report = {"status": "a3_comparison", "pages": len(val), "decoders": DECODERS, "models": {}}
    for label, path in entries.items():
        if not path.exists():
            print(f"{label:<30} missing ({path})")
            continue
        cache_path = args.cache_dir / f"{path.parent.name}.npz"
        if cache_path.exists():
            with np.load(cache_path) as loaded:
                maps = {key: loaded[key].astype(np.float32) for key in loaded.files}
            checkpoint_epoch = None
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model_class = MODELS.get(checkpoint.get("model_variant", "shared"))
            model = model_class(checkpoint["base_channels"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            checkpoint_epoch = checkpoint["epoch"]
            maps = {}
            for row in val:
                with Image.open(row["image"]) as source:
                    image = np.asarray(source.convert("RGB"))
                maps[row["case_id"]] = predict(model, image, device, 512, 384)
            np.savez_compressed(cache_path, **{k: v.astype(np.float16) for k, v in maps.items()})

        wire_tp = wire_fp = wire_fn = 0
        for case_id, probability in maps.items():
            predicted = probability[0] >= 0.5
            truth = truth_wire[case_id]
            wire_tp += int((predicted & truth).sum())
            wire_fp += int((predicted & ~truth).sum())
            wire_fn += int((truth & ~predicted).sum())
        entry = {"checkpoint": str(path), "epoch": checkpoint_epoch,
                 "wire": {"precision": round(wire_tp / (wire_tp + wire_fp), 4),
                          "recall": round(wire_tp / (wire_tp + wire_fn), 4),
                          "f1": round(f1(wire_tp, wire_fp, wire_fn), 4)},
                 "pin": {}}
        for decoder, (kernel, threshold) in DECODERS.items():
            tp = fp = fn = 0
            for case_id, probability in maps.items():
                hit, unmatched_pred, unmatched_truth = match(
                    peaks(probability[1], threshold, kernel), truth_pin[case_id], 5.0)
                tp += hit
                fp += len(unmatched_pred)
                fn += len(unmatched_truth)
            entry["pin"][decoder] = {"precision": round(tp / (tp + fp), 4),
                                     "recall": round(tp / (tp + fn), 4),
                                     "f1": round(f1(tp, fp, fn), 4)}
        report["models"][label] = entry
        print(f"{label:<30} wire F1 {entry['wire']['f1']:.4f}  " +
              "  ".join(f"pin[{d}] {v['f1']:.4f}" for d, v in entry["pin"].items()), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwritten:", args.output)


if __name__ == "__main__":
    main()
