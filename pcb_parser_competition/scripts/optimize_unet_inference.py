"""Calibrate U-Net inference on fixed development pages, then accept on holdout.

The base checkpoint was selected using the old validation set. The holdout here
is reserved for postprocessing acceptance, not an independent model test set.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_inference import predict_at_scale, combine_probabilities, peak_candidates, select_candidates, decode_pins
from evaluate_trained_unet import match_points, metrics, draw_review

MODES = ["native", "zoom15", "zoom20", "mean_native_zoom15", "mean_native_zoom20", "mean_all"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def counts_add(total, value):
    for i, key in enumerate(["tp", "fp", "fn"]):
        total[i] += value[key]


def wire_score(probability, expected, threshold):
    tp = int(np.count_nonzero(probability[expected] >= threshold))
    count = int(np.count_nonzero(probability >= threshold))
    return metrics(tp, count - tp, int(expected.sum()) - tp)


def load_targets(row):
    with Image.open(row["wire_mask"]) as image:
        wire = np.asarray(image) >= 128
    points = read(Path(row["wire_mask"]).parent / "pin_points.json")
    return wire, [p["point"] for p in points]


def cache(args):
    project, output = args.project.resolve(), args.output.resolve()
    source_run = project / "runs/training/unet_wire_pin_v1"
    source_summary = read(source_run / "summary.json")
    manifest = project / "data/segmentation_unet_v1/manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    weights = source_run / "best.pt"
    fingerprint = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert fingerprint == source_summary["checkpoint_sha256"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == source_summary["manifest_sha256"]
    selected = set(source_summary["validation_cases"])
    rows = [row for row in rows if row["case_id"] in selected]
    groups = {}
    for row in rows:
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        image_digest = hashlib.sha256(str(rgb.shape).encode("ascii") + rgb.tobytes()).hexdigest()
        row["decoded_pixel_sha256"] = image_digest
        groups.setdefault(image_digest, []).append(row)
    rows = [min(group, key=lambda row: row["case_id"]) for group in groups.values()]
    order = sorted(rows, key=lambda row: hashlib.sha256(("unet-v2-calibration:" + row["decoded_pixel_sha256"]).encode()).hexdigest())
    development = [row["case_id"] for row in order[:len(order) // 2]]
    holdout = [row["case_id"] for row in order[len(order) // 2:]]
    split = {"development": development, "acceptance_holdout": holdout,
             "unique_cases": len(rows), "source_checkpoint_sha256": fingerprint,
             "checkpoint_epoch": 57, "source_manifest_sha256": source_summary["manifest_sha256"],
             "policy": "Split fixed before search. Tune only development; holdout used once for acceptance.",
             "scope": "Old model validation pages; not independent engineering test data",
             "acceptance_rule": "Adopt only if holdout wire Dice and pin coordinate F1 are both nondecreasing and at least one improves by 0.005.",
             "omitted_duplicate_cases": sorted(selected - {row["case_id"] for row in rows}), "rows": rows}
    if (output / "split.json").exists():
        assert read(output / "split.json") == split
    save(output / "split.json", split)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    assert checkpoint["kind"] == "trained_on_draft_labels"
    split["checkpoint_epoch"] = checkpoint["epoch"]
    model = WirePinUNet(checkpoint["base_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    if args.device.startswith("cuda"):
        assert torch.cuda.mem_get_info(args.device)[0] > 2 * 1024**3
    cfg = checkpoint["config"]
    for row in rows:
        folder = output / "cache" / row["case_id"]
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        for scale in [1.0, 1.5, 2.0]:
            destination = folder / f"scale_{scale:.1f}.npz"
            if destination.exists():
                continue
            probability = predict_at_scale(model, rgb, args.device, cfg["tile_size"], cfg["stride"], scale)
            np.savez_compressed(destination, probability=probability)
        print(json.dumps({"phase": "cache", "case": row["case_id"]}), flush=True)


def predictions(output, case):
    return {scale: np.load(output / "cache" / case / f"scale_{scale:.1f}.npz")["probability"]
            for scale in [1.0, 1.5, 2.0]}


def tune(args):
    output = args.output.resolve()
    if (output / "frozen_inference.json").exists():
        raise ValueError("Calibration is frozen; use a new output directory for a new experiment")
    split = read(output / "split.json")
    wire_specs = [{"mode": mode, "threshold": threshold} for mode, threshold in
                  itertools.product(MODES, [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85])]
    pin_specs = [{"mode": mode, "threshold": threshold, "kernel": kernel, "refinement": refinement, "nms_radius": radius}
                 for mode, threshold, kernel, refinement, radius in
                 itertools.product(MODES, [0.35, 0.5, 0.65, 0.8, 0.9, 0.95], [3, 5, 9], ["none", "quadratic"], [0.0, 3.0])]
    wire_counts, pin_counts = [[0, 0, 0] for _ in wire_specs], [[0, 0, 0] for _ in pin_specs]
    dev_ids = set(split["development"])
    for row in split["rows"]:
        if row["case_id"] not in dev_ids:
            continue
        expected_wire, expected_pins = load_targets(row)
        maps = predictions(output, row["case_id"])
        for mode in MODES:
            probability = combine_probabilities(maps, mode)
            for index, spec in enumerate(wire_specs):
                if spec["mode"] == mode:
                    counts_add(wire_counts[index], wire_score(probability[0], expected_wire, spec["threshold"]))
            peaks = {(kernel, refinement): peak_candidates(probability[1], 0.35, kernel, refinement)
                     for kernel, refinement in itertools.product([3, 5, 9], ["none", "quadratic"])}
            for index, spec in enumerate(pin_specs):
                if spec["mode"] != mode:
                    continue
                selected = select_candidates(peaks[(spec["kernel"], spec["refinement"])],
                                             spec["threshold"], spec["nms_radius"], 500)
                _, scored = match_points([item["point"] for item in selected], expected_pins)
                counts_add(pin_counts[index], scored)
        print(json.dumps({"phase": "calibration", "case": row["case_id"]}), flush=True)
    wire_results = [{"settings": spec, "metrics": metrics(*counts)} for spec, counts in zip(wire_specs, wire_counts)]
    pin_results = [{"settings": spec, "metrics": metrics(*counts)} for spec, counts in zip(pin_specs, pin_counts)]
    best_wire = max(wire_results, key=lambda row: row["metrics"]["f1_dice"])
    best_pin = max(pin_results, key=lambda row: row["metrics"]["f1_dice"])
    frozen = {"version": "2.0", "wire_mode": best_wire["settings"]["mode"],
              "wire_threshold": best_wire["settings"]["threshold"], "pin_mode": best_pin["settings"]["mode"],
              "pin_threshold": best_pin["settings"]["threshold"], "pin_peak_kernel": best_pin["settings"]["kernel"],
              "pin_refinement": best_pin["settings"]["refinement"], "pin_nms_radius": best_pin["settings"]["nms_radius"],
              "pin_candidate_limit": 500, "traditional_image_segmentation": False,
              "checkpoint_sha256": split["source_checkpoint_sha256"],
              "calibration_cases": split["development"], "acceptance_cases": split["acceptance_holdout"],
              "labels_status": "draft_not_certified_Golden", "official_score": None}
    save(output / "calibration_search.json", {"wire": wire_results, "pin": pin_results,
                                             "selected_wire": best_wire, "selected_pin": best_pin})
    save(output / "frozen_inference.json", frozen)
    print(json.dumps({"selected": frozen, "wire_metrics": best_wire["metrics"], "pin_metrics": best_pin["metrics"]}), flush=True)


def evaluate(args):
    output = args.output.resolve()
    split = read(output / "split.json")
    config_path = output / "frozen_inference.json"
    frozen_bytes = config_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    result = {"calibration_config_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "config": frozen,
              "status": "frozen_postprocessing_acceptance_against_draft_labels",
              "scope": split["scope"], "official_score": None, "cases": []}
    for row in split["rows"]:
        maps = predictions(output, row["case_id"])
        native = maps[1.0]
        wire = combine_probabilities(maps, frozen["wire_mode"])[0]
        pin = combine_probabilities(maps, frozen["pin_mode"])[1]
        expected_wire, expected_pins = load_targets(row)
        baseline_candidates = decode_pins(native[1])
        improved_candidates = decode_pins(pin, frozen["pin_threshold"], frozen["pin_peak_kernel"],
                                         frozen["pin_refinement"], frozen["pin_nms_radius"], frozen["pin_candidate_limit"])
        baseline_matches, baseline_pin = match_points([p["point"] for p in baseline_candidates], expected_pins)
        improved_matches, improved_pin = match_points([p["point"] for p in improved_candidates], expected_pins)
        baseline_wire = wire_score(native[0], expected_wire, 0.5)
        improved_wire = wire_score(wire, expected_wire, frozen["wire_threshold"])
        record = {"case_id": row["case_id"], "subset": "development" if row["case_id"] in split["development"] else "acceptance_holdout",
                  "baseline": {"wire": baseline_wire, "pin": baseline_pin},
                  "optimized": {"wire": improved_wire, "pin": improved_pin}}
        result["cases"].append(record)
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        folder = output / "predictions" / row["case_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for name, probability, threshold, candidates, matches, wire_metrics, pin_metrics in [
            ("baseline", native[0], 0.5, baseline_candidates, baseline_matches, baseline_wire, baseline_pin),
            ("optimized", wire, frozen["wire_threshold"], improved_candidates, improved_matches, improved_wire, improved_pin),
        ]:
            view_record = {"case_id": row["case_id"] + " " + name, "checkpoint_epoch": split["checkpoint_epoch"],
                           "wire_pixels": wire_metrics, "pin_localization_5px": pin_metrics}
            draw_review(rgb, expected_wire, probability >= threshold, expected_pins,
                        [p["point"] for p in candidates], matches, view_record).save(folder / f"{name}.png")
        Image.fromarray((wire >= frozen["wire_threshold"]).astype(np.uint8) * 255).save(folder / "wire_mask.png")
        save(folder / "pin_candidates.json", improved_candidates)
    for subset in ["development", "acceptance_holdout", "all_unique"]:
        chosen = [row for row in result["cases"] if subset == "all_unique" or row["subset"] == subset]
        summary = {"cases": len(chosen)}
        for version in ["baseline", "optimized"]:
            summary[version] = {}
            for head in ["wire", "pin"]:
                counts = [sum(row[version][head][key] for row in chosen) for key in ["tp", "fp", "fn"]]
                summary[version][head] = metrics(*counts)
        result[subset] = summary
    holdout = result["acceptance_holdout"]
    delta_wire = holdout["optimized"]["wire"]["f1_dice"] - holdout["baseline"]["wire"]["f1_dice"]
    delta_pin = holdout["optimized"]["pin"]["f1_dice"] - holdout["baseline"]["pin"]["f1_dice"]
    result["acceptance_passed"] = delta_wire >= 0 and delta_pin >= 0 and max(delta_wire, delta_pin) >= 0.005
    result["acceptance_deltas"] = {"wire_dice": delta_wire, "pin_coordinate_f1": delta_pin}
    assert config_path.read_bytes() == frozen_bytes
    save(output / "comparison.json", result)
    lines = ["# U-Net 推理优化 v2", "", "使用相同第 57 轮权重，不重新训练。先固定去重分组和开发/验收划分，再冻结调参结果，最后进行验收。",
             "原模型曾用这些页选择 checkpoint；本次验收页只用于后处理验收，不是独立工程测试集。标签仍为草稿。", "",
             "| 子集 | 张数 | 导线 Dice 原版→优化 | 引脚坐标 F1 原版→优化 |", "|---|---:|---:|---:|"]
    for subset, name in [("development", "调参"), ("acceptance_holdout", "保留验收"), ("all_unique", "全部唯一图片")]:
        item = result[subset]
        lines.append(f"| {name} | {item['cases']} | {item['baseline']['wire']['f1_dice']:.2%} → {item['optimized']['wire']['f1_dice']:.2%} | {item['baseline']['pin']['f1_dice']:.2%} → {item['optimized']['pin']['f1_dice']:.2%} |")
    lines += ["", f"预先规定的验收条件通过：{result['acceptance_passed']}。", "",
              "引脚采用 5px 欧氏距离、一对一最大匹配，仅衡量坐标，不是正式 PinF1；导线 Dice 不是网连接分数。",
              "", "## 冻结配置", "", "```json", json.dumps(frozen, ensure_ascii=False, indent=2), "```", "",
              "## 逐页变化", "", "| 图片 | 子集 | 导线 Dice 原版→优化 | 引脚 F1 原版→优化 | 对照图 |", "|---|---|---:|---:|---|"]
    for row in result["cases"]:
        image = (output / "predictions" / row["case_id"] / "optimized.png").as_posix()
        lines.append(f"| {row['case_id']} | {row['subset']} | {row['baseline']['wire']['f1_dice']:.2%} → {row['optimized']['wire']['f1_dice']:.2%} | {row['baseline']['pin']['f1_dice']:.2%} → {row['optimized']['pin']['f1_dice']:.2%} | [查看](<{image}>) |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["cases", "config"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["cache", "tune", "evaluate"], required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.resolve() == args.project.resolve() or args.project.resolve() in args.output.resolve().parents:
        raise ValueError("Experiment output must be outside the source training project")
    {"cache": cache, "tune": tune, "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
