"""Reproduce frozen predictions and count dataset/split facts without training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import time

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
import torch

from unet_audit_common import ROOT, OUT, read, write, sha, rows, image, truth, gt_path, image_path, metrics, counts, aggregate, val_cases, assert_protected
from pcb_parser.unet_data import predict_tiled, tile_starts
from pcb_parser.unet_model import WirePinUNet


def decoded_hash(rgb):
    h = hashlib.sha256(); h.update(np.asarray(rgb.shape, dtype=np.int64).tobytes()); h.update(rgb.tobytes())
    return h.hexdigest()


def mask_widths(gt):
    skel = skeletonize(gt)
    values = np.maximum(1, np.rint(2*ndimage.distance_transform_edt(gt)[skel]-1).astype(int))
    levels, n = np.unique(values, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(levels, n)}, int(skel.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-predictions", action="store_true", help="Also predict train pages for clearly separated diagnostic review")
    args = parser.parse_args()
    nprotected = assert_protected()
    protocol = read(OUT/"protocol.json")
    dataset = rows()
    selected = set(val_cases())
    prediction_cases = selected | (set(protocol["train_cases"]) if args.train_predictions else set())
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 2*1024**3:
        raise RuntimeError("GPU inference unavailable; no fallback training or long CPU job")
    device = "cuda:0"
    checkpoint = torch.load(ROOT/"weights/unet_wire_pin_v1.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 57 and sha(ROOT/"weights/unet_wire_pin_v1.pt") == protocol["checkpoint_sha256"]
    model = WirePinUNet(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    old = read(ROOT/"runs/unet_contrast_verification_v1/evaluation.json")
    old_records = {r["case_id"]: r for r in old["cases"] if r["gamma"] == 1}
    raw = {r["case_id"]: r for r in map(json.loads, (ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    records = []
    start = time.perf_counter()
    for index, row in enumerate(dataset):
        c = row["case_id"]; rgb = image(c)
        assert rgb.shape[:2] == (row["height"], row["width"])
        folder = OUT/f"baseline/{c}"; folder.mkdir(parents=True, exist_ok=True)
        cache = folder/"probabilities.npz"
        p = None
        if c in prediction_cases:
            if cache.exists():
                meta = read(folder/"prediction_meta.json")
                assert meta["image_sha256"] == sha(image_path(c)) and meta["checkpoint_sha256"] == protocol["checkpoint_sha256"]
                with np.load(cache, allow_pickle=False) as values:
                    p = values["wire"].copy()
            else:
                probability = predict_tiled(model, rgb, device, 512, 384)
                p = probability[0]
                np.savez_compressed(cache, wire=p, pin_tip=probability[1])
                write(folder/"prediction_meta.json", dict(image_sha256=sha(image_path(c)), checkpoint_sha256=protocol["checkpoint_sha256"],
                    image_only=True, resize=False, tile_size=512, stride=384, fusion="uniform probability mean", code_sha256=sha(ROOT/"src/pcb_parser/unet_data.py")))
        # Labels are loaded after the image-only prediction (or cache retrieval).
        gt = truth(c)
        assert gt.shape == rgb.shape[:2]
        hist, skeleton_pixels = mask_widths(gt)
        byte_values, byte_counts = np.unique(np.asarray(Image.open(gt_path(c))), return_counts=True)
        valid_pixels = positive_pixels = empty_tiles = ntiles = 0
        for y in tile_starts(row["height"], 512, 384):
            for x in tile_starts(row["width"], 512, 384):
                tile = gt[y:y+512, x:x+512]
                ntiles += 1; valid_pixels += tile.size; positive_pixels += int(tile.sum()); empty_tiles += not tile.any()
        record = dict(case_id=c, split=row["split"], included_in_26=c in selected, size=[row["width"], row["height"]],
            pixels=gt.size, positive_pixels=int(gt.sum()), positive_fraction=float(gt.mean()),
            mask_value_counts={str(int(v)): int(n) for v, n in zip(byte_values, byte_counts)},
            gt_width_histogram=hist, skeleton_pixels=skeleton_pixels,
            tile_count=ntiles, empty_wire_tiles=empty_tiles, tile_valid_pixels=valid_pixels, tile_positive_pixels=positive_pixels,
            decoded_pixel_sha256=decoded_hash(rgb), original_png_sha256=sha(image_path(c)),
            source_group=row.get("source_group"), source_image_mode=raw[c].get("image_mode"), source_image_format=raw[c].get("image_format"),
            wire_positive_image=bool(gt.any()), image_and_mask_resize=False)
        if p is not None:
            record["wire"] = metrics(*counts(p >= .5, gt))
            if c in selected:
                expected = old_records[c]["wire"]
                assert all(record["wire"][key] == expected[key] for key in ("tp", "fp", "fn")), f"Baseline reproduction differs for {c}"
            Image.fromarray((p >= .5).astype(np.uint8)*255).save(folder/"wire_mask.png")
        records.append(record)
        write(folder/"dataset_stats.json", record)
        print(json.dumps(dict(case_id=c, completed=index+1, total=len(dataset), split=row["split"], dice=record.get("wire",{}).get("dice"))), flush=True)
    stats = {}
    for name, group in (("train103", [r for r in records if r["split"] == "train"]),
                        ("val28", [r for r in records if r["split"] == "val"]),
                        ("val26_unique", [r for r in records if r["included_in_26"]])):
        width_hist = Counter()
        for r in group:
            width_hist.update(r["gt_width_histogram"])
        stats[name] = dict(images=len(group), pixels=sum(r["pixels"] for r in group), positive_pixels=sum(r["positive_pixels"] for r in group),
            positive_fraction=sum(r["positive_pixels"] for r in group)/sum(r["pixels"] for r in group),
            gt_width_histogram=dict(width_hist), original_sizes=[r["size"] for r in group],
            width_min=min(r["size"][0] for r in group), width_max=max(r["size"][0] for r in group),
            height_min=min(r["size"][1] for r in group), height_max=max(r["size"][1] for r in group),
            all_background_images=sum(not r["wire_positive_image"] for r in group),
            tiles=sum(r["tile_count"] for r in group), empty_wire_tiles=sum(r["empty_wire_tiles"] for r in group),
            tile_valid_pixels=sum(r["tile_valid_pixels"] for r in group), tile_positive_pixels=sum(r["tile_positive_pixels"] for r in group),
            mask_values=sorted(set(k for r in group for k in r["mask_value_counts"])),
            source_groups_known=sum(r["source_group"] is not None for r in group), source_mode_counts=dict(Counter(r["source_image_mode"] for r in group)))
    hashes = defaultdict(list)
    for r in records:
        hashes[r["decoded_pixel_sha256"]].append(r)
    duplicate_groups = []
    for h, group in hashes.items():
        if len(group) > 1:
            duplicate_groups.append(dict(pixel_hash=h, cases=[r["case_id"] for r in group], splits=[r["split"] for r in group],
                cross_split=len(set(r["split"] for r in group)) > 1,
                GT_pair_dice=[dict(a=group[0]["case_id"], b=r["case_id"], dice=metrics(*counts(truth(group[0]["case_id"]), truth(r["case_id"])))["dice"])
                    for r in group[1:]]))
    val_records = [r for r in records if r["included_in_26"]]
    micro = aggregate(val_records)
    assert micro["tp"] == 222927 and micro["fp"] == 83747 and micro["fn"] == 31495
    result = dict(status="baseline_reproduced_exact_counts", micro=micro, macro_dice=float(np.mean([r["wire"]["dice"] for r in val_records])),
        stats=stats, duplicate_groups=duplicate_groups, cross_split_identical_groups=sum(r["cross_split"] for r in duplicate_groups),
        split_unit="case assigned before any tiles are generated", source_groups_verified=False,
        actual_input_resolution="original scale; 512 crops with white padding, no resize", records=records,
        predictions_cover_train=args.train_predictions, protected_files=assert_protected(), elapsed_seconds=time.perf_counter()-start,
        width_definition="GT skeleton DT: rounded max(1,2*d-1); label stroke width, NOT actual PNG wire width", training_started=False)
    write(OUT/"baseline_reproduction.json", result)
    print(json.dumps(dict(output=str(OUT), micro=micro, stats=stats, duplicate_groups=duplicate_groups, elapsed_seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
