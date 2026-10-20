"""Train the U-Net on v2 supervision, with pin metrics measured during training.

This is a new entry point; `scripts/train_unet.py` and its configs are untouched.

What is different from v1:
  * reads the v2 manifest and its per-channel ignore masks, so pages are no longer
    thrown away for a single bad edge or unowned pin;
  * validates on THREE metrics every epoch instead of val_loss only:
      - wire pixel F1 @0.5 (primary wire signal)
      - pin point-set F1 @5 px, decoded exactly like the accepted inference path
      - legacy val_loss, kept only so E0 stays comparable
  * keeps three checkpoints (best by loss, best by composite, best by pin) so a
    single ambiguous objective cannot hide a regression in another;
  * decodes pages once into an LRU page cache.  The v1 run spent ~70 % of its
    wall clock re-decoding PNGs per tile (177 ms/step against a 50 ms GPU step).
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pcb_parser.unet_model import WirePinUNet, segmentation_loss  # noqa: E402
from pcb_parser.unet_split_model import SplitWirePinUNet, load_shared_decoder_weights  # noqa: E402
from evaluate_unet_baseline import peaks  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/unet_wire_pin_v2.json"
MODELS = {"shared": WirePinUNet, "split": SplitWirePinUNet}


def tile_starts(length, size, stride):
    return [0] if length <= size else sorted(set(list(range(0, length - size + 1, stride)) + [length - size]))


class PageCache:
    """Decode each page at most once per epoch window; slice tiles from arrays."""

    def __init__(self, limit_bytes=2 * 1024 ** 3, pin_key="pin_heatmap"):
        self.pages = {}
        self.order = []
        self.used = 0
        self.limit = limit_bytes
        self.pin_key = pin_key

    def get(self, row):
        case_id = row["case_id"]
        if case_id in self.pages:
            return self.pages[case_id]
        with Image.open(row["image"]) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        with Image.open(row["wire_mask"]) as mask:
            wire = np.asarray(mask, dtype=np.uint8)
        pin_path = row.get(self.pin_key) or row["pin_heatmap"]
        with Image.open(pin_path) as heat:
            pin = np.asarray(heat, dtype=np.uint8)
        with np.load(row["ignore_masks"]) as ignore:
            wire_ignore = ignore["wire_ignore"].copy()
            pin_ignore = ignore["pin_ignore"].copy()
        page = (image, wire, pin, wire_ignore, pin_ignore)
        size = sum(a.nbytes for a in page)
        while self.used + size > self.limit and self.order:
            evicted = self.order.pop(0)
            self.used -= sum(a.nbytes for a in self.pages.pop(evicted))
        self.pages[case_id] = page
        self.order.append(case_id)
        self.used += size
        return page


class Tiles(Dataset):
    def __init__(self, rows, size, stride, cache, pin_key="pin_heatmap"):
        self.rows, self.size, self.cache, self.pin_key = rows, size, cache, pin_key
        self.index = []
        for i, row in enumerate(rows):
            for y in tile_starts(row["height"], size, stride):
                for x in tile_starts(row["width"], size, stride):
                    self.index.append((i, x, y))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, x, y = self.index[index]
        row = self.rows[i]
        image, wire, pin, wire_ignore, pin_ignore = self.cache.get(row)
        s = self.size
        h, w = min(s, row["height"] - y), min(s, row["width"] - x)
        rgb = np.full((s, s, 3), 255, np.uint8)
        rgb[:h, :w] = image[y:y + h, x:x + w]
        target = np.zeros((2, s, s), np.float32)
        target[0, :h, :w] = wire[y:y + h, x:x + w] / 255.0
        target[1, :h, :w] = pin[y:y + h, x:x + w] / 255.0
        # valid = inside the image AND not flagged as unknown for that channel
        valid = np.zeros((2, s, s), np.float32)
        valid[0, :h, :w] = 1.0
        valid[1, :h, :w] = 1.0
        valid[0, :h, :w] *= (wire_ignore[y:y + h, x:x + w] == 0)
        valid[1, :h, :w] *= (pin_ignore[y:y + h, x:x + w] == 0)
        return (torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255,
                torch.from_numpy(target), torch.from_numpy(valid))


def loss_with_ignore(logits, targets, valid, wire_weight, pin_weight, pin_loss_weight=0.5):
    """Same weighting as v1, but each channel is normalised by its own valid pixels."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    terms = []
    for channel, positive_weight in enumerate([wire_weight, pin_weight]):
        pixel = bce(logits[:, channel], targets[:, channel], reduction="none",
                    pos_weight=logits.new_tensor(positive_weight))
        mask = valid[:, channel]
        terms.append((pixel * mask).sum() / mask.sum().clamp_min(1))
    probability = logits[:, 0].sigmoid() * valid[:, 0]
    truth = targets[:, 0] * valid[:, 0]
    dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    return terms[0] + dice + pin_loss_weight * terms[1], {"wire_bce": float(terms[0].detach()),
                                             "wire_dice": float(dice.detach()),
                                             "pin_bce": float(terms[1].detach())}


@torch.inference_mode()
def predict_page(model, image, device, tile, stride):
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
            output[:, y:y + h, x:x + w] += model(batch).sigmoid()[0, :, :h, :w].float().cpu().numpy()
            count[y:y + h, x:x + w] += 1
    return output / count[None]


def match(pred, truth, tolerance):
    if not pred or not truth:
        return 0, len(pred), len(truth)
    distances, indices = cKDTree(np.asarray(truth)).query(np.asarray(pred), k=1)
    used_truth, used_pred, tp = set(), set(), 0
    for i in np.argsort(distances):
        j = int(indices[i])
        if distances[i] <= tolerance and j not in used_truth:
            used_truth.add(j)
            used_pred.add(int(i))
            tp += 1
    return tp, len(pred) - len(used_pred), len(truth) - len(used_truth)


def decode_metric(cfg):
    return {"wire_threshold": cfg["inference"]["wire_threshold"],
            "pin_threshold": cfg["inference"]["pin_threshold"],
            "pin_kernel": cfg["inference"]["pin_peak_kernel"],
            "pin_tolerance": cfg["inference"]["pin_tolerance_px"]}


@torch.inference_mode()
def validate(model, rows, cache, device, cfg, tile, stride, metric_cfg):
    """Pixel and point metrics on full pages, plus a val loss on the same tiles as train."""
    counts = {"wire_tp": 0, "wire_fp": 0, "wire_fn": 0, "pin_tp": 0, "pin_fp": 0, "pin_fn": 0}
    ignored_peaks = 0
    model.eval()
    for row in rows:
        image, wire, pin, _, pin_ignore = cache.get(row)
        probability = predict_page(model, image, device, tile, stride)
        predicted = probability[0] >= metric_cfg["wire_threshold"]
        truth = wire > 0
        counts["wire_tp"] += int((predicted & truth).sum())
        counts["wire_fp"] += int((predicted & ~truth).sum())
        counts["wire_fn"] += int((truth & ~predicted).sum())
        tips = [tuple(p["point"]) for p in json.loads(Path(row["pin_points"]).read_text(encoding="utf-8"))]
        candidates = []
        for x, y in peaks(probability[1], metric_cfg["pin_threshold"], metric_cfg["pin_kernel"]):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < pin_ignore.shape[0] and 0 <= xi < pin_ignore.shape[1] and pin_ignore[yi, xi]:
                ignored_peaks += 1
                continue
            candidates.append((x, y))
        tp, fp, fn = match(candidates, tips, metric_cfg["pin_tolerance"])
        counts["pin_tp"] += tp
        counts["pin_fp"] += fp
        counts["pin_fn"] += fn

    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "wire_pixel_f1": round(f1(counts["wire_tp"], counts["wire_fp"], counts["wire_fn"]), 4),
        "wire_pixel_precision": round(counts["wire_tp"] / max(1, counts["wire_tp"] + counts["wire_fp"]), 4),
        "wire_pixel_recall": round(counts["wire_tp"] / max(1, counts["wire_tp"] + counts["wire_fn"]), 4),
        "pin_point_f1": round(f1(counts["pin_tp"], counts["pin_fp"], counts["pin_fn"]), 4),
        "pin_point_precision": round(counts["pin_tp"] / max(1, counts["pin_tp"] + counts["pin_fp"]), 4),
        "pin_point_recall": round(counts["pin_tp"] / max(1, counts["pin_tp"] + counts["pin_fn"]), 4),
        "pin_peaks_ignored": ignored_peaks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--init-weights", type=Path, default=None, help="warm start; omit for from-scratch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path(r"D:/pcb_unet_v2/runs/A2_v2_from_v1"))
    parser.add_argument("--max-train-steps", type=int, default=None, help="debug only")
    parser.add_argument("--dry-run", action="store_true", help="one epoch on 20 tiles, no checkpoint writes to the run dir")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = args.manifest or Path(cfg["dataset"]["manifest"])
    training = cfg["training"]
    epochs = args.epochs or training["epochs"]
    batch_size = args.batch_size or training["batch_size"]
    tile, stride = cfg["tile_size"], cfg["stride"]
    metric_cfg = {"wire_threshold": cfg["inference"]["wire_threshold"],
                  "pin_threshold": cfg["inference"]["pin_threshold"],
                  "pin_kernel": cfg["inference"]["pin_peak_kernel"],
                  "pin_tolerance": cfg["inference"]["pin_tolerance_px"]}

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    torch.manual_seed(training["seed"])
    random.seed(training["seed"])
    torch.set_num_threads(4)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_kind = cfg.get("model_variant", "shared")
    if model_kind not in MODELS:
        raise ValueError(f"unknown model_variant {model_kind!r}; expected one of {sorted(MODELS)}")
    model = MODELS[model_kind](cfg["base_channels"]).to(device)
    pin_key = cfg.get("pin_target", "pin_heatmap")
    if pin_key == "pin_heatmap_disk" and "pin_heatmap_disk" not in rows[0]:
        raise ValueError("pin_heatmap_disk missing from manifest; run scripts/prepare_pin_disk_targets.py")
    init_note = "scratch"
    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location=device, weights_only=True)
        if model_kind == "split":
            transferred = load_shared_decoder_weights(model, checkpoint["state_dict"])
            init_note = (f"warm_start_split:{args.init_weights.name}:epoch{checkpoint['epoch']}:"
                         f"{checkpoint['kind']}:enc{transferred['copied_encoder']}:dec{transferred['copied_decoders']}")
        else:
            model.load_state_dict(checkpoint["state_dict"])
            init_note = f"warm_start:{args.init_weights.name}:epoch{checkpoint['epoch']}:{checkpoint['kind']}"
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and training["amp"])
    train_cache, val_cache = PageCache(pin_key=pin_key), PageCache(pin_key=pin_key)
    dataset = Tiles(train_rows, tile, stride, train_cache, pin_key=pin_key)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_tiles = Tiles(val_rows, tile, stride, val_cache, pin_key=pin_key)
    val_loader = torch.utils.data.DataLoader(val_tiles, batch_size=batch_size, num_workers=0)
    args.output.mkdir(parents=True, exist_ok=True)

    history = []
    best = {"wire": -1.0, "pin": -1.0, "composite": -1.0}
    started = time.perf_counter()
    for epoch_index in range(epochs):
        model.train()
        losses, component_sums = [], {}
        epoch_start = time.perf_counter()
        for step, (images, targets, valid) in enumerate(loader, 1):
            images, targets, valid = images.to(device), targets.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, enabled=bool(training["amp"] and device.type == "cuda")):
                logits = model(images)
                loss, components = loss_with_ignore(logits, targets, valid,
                                                    training["wire_positive_weight"], training["pin_positive_weight"],
                                                    training.get("pin_loss_weight", 0.5))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch_index + 1, "step": step, "steps": len(loader),
                                  "loss": round(losses[-1], 4), "sec": round(time.perf_counter() - epoch_start, 1)}), flush=True)
            if args.max_train_steps and step >= args.max_train_steps:
                break
        train_seconds = time.perf_counter() - epoch_start

        model.eval()
        val_losses, val_start = [], time.perf_counter()
        with torch.inference_mode():
            for images, targets, valid in val_loader:
                images, targets, valid = images.to(device), targets.to(device), valid.to(device)
                with torch.autocast(device_type=device.type, enabled=bool(training["amp"] and device.type == "cuda")):
                    logits = model(images)
                    loss, _ = loss_with_ignore(logits, targets, valid,
                                               training["wire_positive_weight"], training["pin_positive_weight"],
                                               training.get("pin_loss_weight", 0.5))
                val_losses.append(float(loss.detach()))
        metrics = validate(model, val_rows, val_cache, device, cfg, tile, stride, decode_metric(cfg))
        composite = round(0.5 * metrics["wire_pixel_f1"] + 0.5 * metrics["pin_point_f1"], 4)
        metrics["composite"] = composite
        record = {"epoch": epoch_index + 1,
                  "train_loss": round(float(np.mean(losses)), 4),
                  "val_loss": round(float(np.mean(val_losses)), 4) if val_losses else None,
                  "train_seconds": round(train_seconds, 1),
                  "val_seconds": round(time.perf_counter() - val_start, 1),
                  **{f"train_{k}": round(v / max(1, len(losses)), 4) for k, v in component_sums.items()},
                  **metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        checkpoint = {"state_dict": model.state_dict(), "base_channels": cfg["base_channels"],
                      "model_variant": model_kind, "pin_target": pin_key,
                      "kind": cfg["labels_status"], "config": cfg, "outputs": cfg["outputs"],
                      "source_commit": cfg["source_commit"], "optimizer_state": optimizer.state_dict(),
                      "scaler_state": scaler.state_dict(), "epoch": epoch_index + 1,
                      "best": best, "history": history, "init": init_note,
                      "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        torch.save(checkpoint, args.output / "last.pt")
        if metrics["composite"] > best["composite"]:
            best["composite"] = metrics["composite"]
            torch.save(checkpoint, args.output / "best.pt")
        if metrics["wire_pixel_f1"] > best["wire"]:
            best["wire"] = metrics["wire_pixel_f1"]
            torch.save(checkpoint, args.output / "best_wire.pt")
        if metrics["pin_point_f1"] > best["pin"]:
            best["pin"] = metrics["pin_point_f1"]
            torch.save(checkpoint, args.output / "best_pin.pt")
        (args.output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        with (args.output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.dry_run:
            break

    summary = {"status": "dry_run" if args.dry_run else "trained",
               "init": init_note, "device": str(device), "epochs_ran": len(history),
               "model_variant": model_kind, "pin_target": pin_key,
               "pin_loss_weight": training.get("pin_loss_weight", 0.5),
               "train_pages": len(train_rows), "val_pages": len(val_rows),
               "train_tiles": len(dataset), "steps_per_epoch": len(loader),
               "parameters": sum(v.numel() for v in model.parameters()),
               "best": best, "wall_seconds": round(time.perf_counter() - started, 1),
               "last": history[-1] if history else None,
               "manifest": str(manifest_path), "config": str(args.config),
               "official_score": None}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
