"""Review an existing U-Net run without changing or training the source project.

Full-page metrics use draft labels. Pin matching measures coordinates only,
not pin ownership, number, name, electrical connectivity, or official PinF1.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }


def match_points(predicted, expected, radius=5.0):
    """Maximum cardinality, then minimum distance, one-to-one radius matching."""
    if not len(predicted) or not len(expected):
        return [], metrics(0, len(predicted), len(expected))
    distances = np.linalg.norm(np.asarray(predicted)[:, None] - np.asarray(expected)[None], axis=2)
    penalty = (min(len(predicted), len(expected)) + 1) * (radius + 1)
    rows, columns = linear_sum_assignment(np.where(distances <= radius, distances, penalty))
    matches = [(int(i), int(j), float(distances[i, j]))
               for i, j in zip(rows, columns) if distances[i, j] <= radius]
    return matches, metrics(len(matches), len(predicted) - len(matches), len(expected) - len(matches))


def candidates_from_heatmap(probability, threshold):
    heat = torch.from_numpy(probability)[None, None]
    peaks = (heat == torch.nn.functional.max_pool2d(heat, 9, stride=1, padding=4)) & (heat >= threshold)
    ys, xs = torch.where(peaks[0, 0])
    candidates = sorted(
        [{"point": [int(x), int(y)], "confidence": float(heat[0, 0, y, x])}
         for y, x in zip(ys, xs)], key=lambda value: -value["confidence"])
    return candidates[:500], len(candidates)


def draw_review(rgb, expected_wire, predicted_wire, expected_pins, predicted_pins, matches, record):
    """A derived diagnostic figure, never fed back into the predictor."""
    wire = rgb.copy()
    for selected, colour in [
        (expected_wire & predicted_wire, (0, 170, 70)),
        (~expected_wire & predicted_wire, (240, 40, 40)),
        (expected_wire & ~predicted_wire, (0, 100, 240)),
    ]:
        wire[selected] = (0.2 * wire[selected] + 0.8 * np.asarray(colour)).astype(np.uint8)
    pins = Image.fromarray(rgb)
    pen = ImageDraw.Draw(pins)
    matched_predictions = {i for i, _, _ in matches}
    matched_targets = {j for _, j, _ in matches}
    for j, (x, y) in enumerate(expected_pins):
        if j not in matched_targets:
            pen.rectangle((x - 5, y - 5, x + 5, y + 5), outline=(0, 100, 240), width=2)
    for i, (x, y) in enumerate(predicted_pins):
        colour = (0, 160, 60) if i in matched_predictions else (240, 40, 40)
        pen.ellipse((x - 5, y - 5, x + 5, y + 5), outline=colour, width=2)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    width, height = 850, 500
    sheet = Image.new("RGB", (width * 2, height + 94), "white")
    for column, source in enumerate([Image.fromarray(wire), pins]):
        source.thumbnail((width - 20, height - 10))
        sheet.paste(source, (column * width + (width - source.width) // 2, 90 + (height - source.height) // 2))
    pen = ImageDraw.Draw(sheet)
    pen.text((15, 8), f"Case {record['case_id']} | best checkpoint epoch {record['checkpoint_epoch']} | DRAFT validation labels", font=font, fill="black")
    pen.text((15, 34), "WIRE: green=TP, red=extra pixels, blue=missed pixels", font=font, fill="black")
    pen.text((width + 15, 34), "PIN: green=matched prediction, red=extra, blue=missed target", font=font, fill="black")
    w, p = record["wire_pixels"], record["pin_localization_5px"]
    pen.text((15, 59), f"P={w['precision']:.3f} R={w['recall']:.3f} Dice={w['f1_dice']:.3f}", font=font, fill="black")
    pen.text((width + 15, 59), f"5px P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1_dice']:.3f}", font=font, fill="black")
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if output == project or project in output.parents:
        raise ValueError("Review output must be outside the source project")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(project / "src"))
    from pcb_parser.unet_data import predict_tiled, tile_starts
    from pcb_parser.unet_model import WirePinUNet

    # Catch non-greedy matching and the inclusive five-pixel boundary.
    assert match_points([[0, 0], [4, 0]], [[2, 0], [-3, 0]], 3)[1]["tp"] == 2
    assert match_points([[0, 0]], [[3, 4]], 5)[1]["tp"] == 1
    assert match_points([], [[0, 0]])[1]["fn"] == 1

    run = project / "runs/training/unet_wire_pin_v1"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    manifest = project / "data/segmentation_unet_v1/manifest.jsonl"
    weights = run / "best.pt"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    assert checkpoint["kind"] not in {"cpu_smoke_only", "partial_training_debug"}
    assert digest(weights) == summary["checkpoint_sha256"]
    assert digest(manifest) == summary["manifest_sha256"] == checkpoint["manifest_sha256"]
    assert summary["history"] == history
    assert len(history) == checkpoint["config"]["training"]["epochs"]
    assert all(row["epoch"] == i + 1 for i, row in enumerate(history))
    best_record = min(history, key=lambda row: row["val_loss"])
    assert checkpoint["epoch"] == best_record["epoch"]
    assert not set(summary["train_cases"]) & set(summary["validation_cases"])
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["case_id"] in summary["validation_cases"]]
    assert {row["case_id"] for row in rows} == set(summary["validation_cases"])
    assert all(row["split"] == "val" and row["structurally_usable_draft"] for row in rows)
    reviews = json.loads((project / "configs/unet_label_reviews.json").read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(args.device)
        if free < 2 * 1024**3:
            raise RuntimeError("Insufficient free GPU memory for review; use CPU")
    model = WirePinUNet(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    cfg = checkpoint["config"]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.perf_counter()
    print(json.dumps({"phase": "review_start", "cases": len(rows), "device": args.device,
                      "best_epoch": checkpoint["epoch"]}), flush=True)
    for row in rows:
        tick = time.perf_counter()
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        assert list(rgb.shape[:2]) == [row["height"], row["width"]]
        # Predictor receives PNG pixels and checkpoint settings only.
        probability = predict_tiled(model, rgb, args.device, cfg["tile_size"], cfg["stride"])
        assert np.isfinite(probability).all()
        predicted_wire = probability[0] >= cfg["inference"]["wire_threshold"]
        candidates, raw_count = candidates_from_heatmap(probability[1], cfg["inference"]["pin_threshold"])
        predicted_pins = [candidate["point"] for candidate in candidates]
        # Labels are loaded after image-only inference, for scoring/display.
        with Image.open(row["wire_mask"]) as image:
            expected_wire = np.asarray(image) >= 128
        tips = json.loads((Path(row["wire_mask"]).parent / "pin_points.json").read_text(encoding="utf-8"))
        expected_pins = [tip["point"] for tip in tips]
        matched, point_metrics = match_points(predicted_pins, expected_pins)
        wire_metrics = metrics(int((predicted_wire & expected_wire).sum()),
                               int((predicted_wire & ~expected_wire).sum()),
                               int((~predicted_wire & expected_wire).sum()))
        distances = [distance for _, _, distance in matched]
        record = {
            "case_id": row["case_id"], "size": [row["width"], row["height"]],
            "checkpoint_epoch": checkpoint["epoch"],
            "wire_pixels": wire_metrics, "pin_localization_5px": point_metrics,
            "predicted_pins": len(candidates), "raw_peak_count": raw_count,
            "predictions_capped_at_500": raw_count > 500, "target_pins": len(tips),
            "coincident_target_coordinates": len(tips) - len(set(map(tuple, expected_pins))),
            "matched_pin_distance_mean": float(np.mean(distances)) if distances else None,
            "matched_pin_distances": distances,
            "tile_count": len(tile_starts(row["width"], cfg["tile_size"], cfg["stride"])) *
                          len(tile_starts(row["height"], cfg["tile_size"], cfg["stride"])),
            "inference_and_scoring_seconds": time.perf_counter() - tick,
        }
        folder = output / row["case_id"]
        folder.mkdir(exist_ok=True)
        Image.fromarray(predicted_wire.astype(np.uint8) * 255).save(folder / "wire_mask.png")
        Image.fromarray((probability[1] * 255).astype(np.uint8)).save(folder / "pin_probability.png")
        (folder / "pin_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        draw_review(rgb, expected_wire, predicted_wire, expected_pins, predicted_pins, matched, record).save(folder / "review.png")
        records.append(record)
        (folder / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in record.items() if k in ["case_id", "wire_pixels", "pin_localization_5px"]}), flush=True)

    def aggregate(key):
        return metrics(*(sum(row[key][count] for row in records) for count in ["tp", "fp", "fn"]))

    result = {
        "status": "full_validation_review_against_draft_labels",
        "source_project": str(project), "source_training_run": str(run),
        "checkpoint": str(weights), "checkpoint_sha256": digest(weights),
        "manifest_sha256": digest(manifest), "checkpoint_epoch": checkpoint["epoch"],
        "training_epochs": len(history), "training_wall_seconds": summary["wall_seconds"],
        "train_cases": len(summary["train_cases"]), "validation_cases": len(records),
        "device": args.device, "review_wall_seconds": time.perf_counter() - start,
        "best_checkpoint_tile_metrics": best_record,
        "last_epoch_tile_metrics": history[-1],
        "full_page_wire_pixels_micro": aggregate("wire_pixels"),
        "pin_localization_5px_micro": aggregate("pin_localization_5px"),
        "pin_localization_scope": "coordinate-only; no owner/number/name; NOT official PinF1",
        "wire_scope": "stitched full pages, each pixel counted once; differs from overlapping training tiles",
        "wire_threshold": cfg["inference"]["wire_threshold"],
        "pin_threshold": cfg["inference"]["pin_threshold"],
        "peak_kernel": 9, "pin_candidate_limit": 500, "matching_radius_px": 5,
        "manually_approved_cases": len(reviews["approved_cases"]),
        "engineering_groups_verified": False, "official_score": None,
        "limitations": ["Draft path strokes and pin labels need review", "Engineering-level split not verified",
                        "Coordinate-only pin matching does not evaluate identity or nets",
                        "No threshold tuning or new training performed"],
        "cases": records,
    }
    assert all(math.isfinite(row["wire_pixels"]["f1_dice"]) for row in records)
    (output / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    wire, pins = result["full_page_wire_pixels_micro"], result["pin_localization_5px_micro"]
    report = [
        "# U-Net 训练结果复核", "",
        f"来源：`{run}`。完成 {len(history)} 轮，训练耗时 {summary['wall_seconds']/60:.1f} 分钟。",
        f"已验证模型和数据清单 SHA256；使用第 {checkpoint['epoch']} 轮 best.pt，严格加载权重。",
        f"{len(summary['train_cases'])} 页训练、{len(records)} 页验证；{cfg['tile_size']} 切片、{cfg['stride']} 步长；不重新训练，不修改来源工程。", "",
        "| 指标 | 最佳轮训练日志（重叠切片） | 本次整图复核 |",
        "|---|---:|---:|",
    ]
    for title, old_key, new_key in [
        ("导线像素 Precision", "wire_pixel_precision", "precision"),
        ("导线像素 Recall", "wire_pixel_recall", "recall"),
        ("导线像素 Dice/F1", "wire_pixel_dice", "f1_dice"),
        ("导线像素 IoU", "wire_pixel_iou", "iou"),
    ]:
        report.append(f"| {title} | {best_record['val_wire_pixels'][old_key]:.2%} | {wire[new_key]:.2%} |")
    report += ["", f"引脚坐标匹配（欧氏距离 ≤5px、一对一）：Precision {pins['precision']:.2%}，Recall {pins['recall']:.2%}，F1 {pins['f1_dice']:.2%}。",
               f"TP={pins['tp']}，FP={pins['fp']}，FN={pins['fn']}。阈值沿用 0.5，没有在验证集上调参。", "",
               "**口径：标签为未经完整人工复核的草稿，工程级划分尚未核验。引脚指标仅衡量坐标定位，不包含归属、编号、名称，也不代表比赛 PinF1。导线像素 Dice 不代表网连接准确率。**", "",
               "`summary.json` 的 quality_metrics 对应第 60 轮 last.pt，并非第 57 轮 best.pt；本报告按实际使用权重分别列出。", "",
               "## 每页结果", "", "| 样例 | 导线 Dice | 引脚坐标 F1 | 引脚 TP/FP/FN | 预测图 |", "|---|---:|---:|---|---|"]
    for row in records:
        p = row["pin_localization_5px"]
        picture = (output / row["case_id"] / "review.png").as_posix()
        report.append(f"| {row['case_id']} | {row['wire_pixels']['f1_dice']:.2%} | {p['f1_dice']:.2%} | {p['tp']}/{p['fp']}/{p['fn']} | [查看](<{picture}>) |")
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["cases", "best_checkpoint_tile_metrics", "last_epoch_tile_metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
