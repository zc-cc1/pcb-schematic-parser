"""Image-only detector + OCR baseline. Annotations are never loaded by this entrypoint."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, settings
from pcb_parser.dataset_converter import validate_output
from pcb_parser.fusion import apply_class_confidence, wbf_merge

settings.update({"sync": False})


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark(function, device, repeats):
    function()  # exclude model initialization and one warm-up from the timed repetitions
    sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        result = function()
        sync(device)
        timings.append((time.perf_counter() - start) * 1000)
    return result, {"wall_ms": timings, "median_ms": statistics.median(timings),
                    "warmup_runs": 1, "timed_repeats": repeats,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith("cuda") else None,
                    "scope": "inference including preprocessing and postprocessing; excludes model load, image read and artifact writing"}


def whole_predictions(model, bgr, cfg, device):
    result = model.predict(source=bgr, imgsz=cfg["imgsz"], conf=cfg["confidence"], iou=cfg["iou"],
                           max_det=cfg["max_det"], device=device, half=cfg["half"], verbose=False)[0]
    output = []
    for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
        output.append({"bbox": xyxy, "confidence": conf, "class_id": int(cls), "class_name": result.names[int(cls)]})
    return output


def sliced_predictions(model, rgb, cfg):
    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(rgb, model, slice_height=cfg["slice_height"], slice_width=cfg["slice_width"],
                                  overlap_height_ratio=cfg["overlap"], overlap_width_ratio=cfg["overlap"],
                                  perform_standard_pred=True, postprocess_type=cfg.get("merge_method", "GREEDYNMM"),
                                  postprocess_match_metric=cfg["merge_metric"], postprocess_match_threshold=cfg["merge_threshold"],
                                  postprocess_class_agnostic=False, verbose=0)
    return [{"bbox": p.bbox.to_xyxy(), "confidence": float(p.score.value), "class_id": p.category.id,
             "class_name": p.category.name} for p in result.object_prediction_list]


def fused_predictions(whole_model, sliced_model, rgb, device, cfg):
    """Dual-path inference: whole-image plus sliced, fused with class-aware WBF.

    Whole mode keeps precision, sliced mode recovers small-symbol recall; WBF
    merges their agreement into one box (confidence-weighted mean coordinates,
    mean score) instead of letting NMS pick a winner between the two paths.
    Per-path confidence: the whole path usually wants a higher floor (it floods
    with FPs when opened up) while the sliced path benefits from a low floor
    (tiny symbols only show up there); fusion.wildcard keys whole_confidence /
    sliced_confidence override detector.confidence per path.
    """
    fusion_cfg = cfg.get("fusion", {})
    whole_cfg = dict(cfg["detector"])
    if fusion_cfg.get("whole_confidence") is not None:
        whole_cfg["confidence"] = fusion_cfg["whole_confidence"]
    whole = whole_predictions(whole_model, np.ascontiguousarray(rgb[:, :, ::-1]), whole_cfg, device)
    sliced = sliced_predictions(sliced_model, rgb, cfg["sahi"])
    return wbf_merge(whole + sliced, iou_threshold=fusion_cfg.get("iou", 0.55))


def ocr_predictions(engine, bgr):
    result = engine(bgr)
    if result.boxes is None:
        return []
    return [{"polygon": np.asarray(box).tolist(), "text": str(text), "confidence": float(score)}
            for box, text, score in zip(result.boxes, result.txts, result.scores)]


def rectangle_gap(a, b):
    return (max(a[0] - b[2], b[0] - a[2], 0)**2 + max(a[1] - b[3], b[1] - a[3], 0)**2)**0.5


def bind_designators(detections, texts, width, height, cfg):
    # All associations are provisional. Keep uncertain OCR and assignments in debug JSON.
    prefixes = {"resistor": "R", "capacitor": "C", "inductor": "L", "diode": "D", "bjt": "Q",
                "opamp": "U", "ic": "U", "crystal": ("X", "Y"), "mosfet": "Q", "switch": "SW", "led": "D",
                "gnd": None, "vcc": None}
    candidates, rejected = [], []
    for text in texts:
        token = text["text"].strip()
        if not re.fullmatch(r"(?:R|C|L|D|Q|U|IC|X|Y|J|P|F|SW|TP)\d+", token, re.IGNORECASE):
            continue
        points = np.asarray(text["polygon"])
        box = [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]
        matches = []
        for idx, detection in enumerate(detections):
            kind = cfg["class_mapping"].get(detection["class_name"], detection["class_name"])
            prefix = prefixes.get(kind)
            if prefix is None or not (token.upper().startswith(prefix) or kind == "ic" and token.upper().startswith("IC")):
                continue
            gap = rectangle_gap(box, detection["bbox"])
            if gap <= cfg["binding"]["max_gap_px"]:
                matches.append((gap, idx))
        matches.sort()
        if matches and (len(matches) == 1 or matches[1][0] - matches[0][0] >= cfg["binding"]["ambiguity_margin_px"]):
            candidates.append((matches[0][0], matches[0][1], token, text))
        else:
            rejected.append({"text": token, "reason": "ambiguous_or_no_compatible_detection", "candidates": matches})
    assignments, names_used = {}, set()
    for gap, idx, name, text in sorted(candidates, key=lambda c: (c[0], -c[3]["confidence"])):
        if idx in assignments or name in names_used:
            rejected.append({"text": name, "reason": "conflicting_component_or_designator", "detection_index": idx})
            continue
        assignments[idx] = {"designator": name, "gap_px": gap, "ocr_evidence": text, "verified": False}
        names_used.add(name)
    output = {"components": {}, "pins": {}, "nets": {}}
    provenance = []
    for idx, detection in enumerate(detections):
        cid = assignments[idx]["designator"] if idx in assignments else f"UNRESOLVED_DET_{idx+1:04d}"
        kind = cfg["class_mapping"].get(detection["class_name"], detection["class_name"])
        x1, y1, x2, y2 = detection["bbox"]
        x1, x2 = max(0., float(x1)), min(float(width), float(x2))
        y1, y2 = max(0., float(y1)), min(float(height), float(y2))
        output["components"][cid] = {"type": kind, "Name": None, "value": None,
                                     "bbox": [x1, height - y2, x2, height - y1]}
        provenance.append({"component_id": cid, "detection": detection, "binding": assignments.get(idx)})
    return output, {"provisional_bindings": provenance, "rejected_candidates": rejected,
                    "resolved_designator_count": len(assignments), "unresolved_designator_count": len(detections)-len(assignments),
                    "status": "partial_components_only_Name_value_pins_nets_not_implemented"}


def draw_overlay(rgb, detections, texts, path):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(Path(__file__).resolve().parents[1] / 'assets/fonts/DejaVuSansMono.ttf'), 18) if Path(str(Path(__file__).resolve().parents[1] / 'assets/fonts/DejaVuSansMono.ttf')).exists() else ImageFont.load_default()
    for idx, detection in enumerate(detections):
        box = detection["bbox"]
        draw.rectangle(box, outline="#E53B32", width=3)
        draw.text((box[0], max(0, box[1]-22)), f'{idx+1} {detection["class_name"]} {detection["confidence"]:.2f}', fill="#E53B32", font=font)
    for text in texts:
        polygon = [tuple(p) for p in text["polygon"]]
        draw.line(polygon+[polygon[0]], fill="#258441", width=2)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--modes", nargs="+", choices=["whole", "sliced", "fused"], default=["whole", "sliced"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    weights = ROOT / cfg["weights"]
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if any(d.startswith("cuda") for d in args.devices) and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
               "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "python": sys.executable,
               "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "ultralytics", "sahi", "rapidocr", "onnxruntime")},
               "input_contract": "images + model/config only; no raw/standardized reference JSON read",
               "status": "partial_detection_ocr_baseline", "runs": [], "ocr": {},
               "caveats": ["Detection count is not accuracy or recall.", "No reference F1 is computed.",
                           "Name/value/pins/nets are not implemented; exported JSON is partial.",
                           "H-y coordinates and class mapping remain provisional."]}
    engine = None
    if not args.skip_ocr:
        from rapidocr import RapidOCR
        engine = RapidOCR(params={"Det.limit_side_len": cfg["ocr"]["det_limit_side_len"], "Det.limit_type": "max",
                                  "Global.max_side_len": cfg["ocr"]["det_limit_side_len"]})
    for image_path in args.image:
        image_path = image_path.resolve()
        case_id = image_path.stem
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        height, width = rgb.shape[:2]
        texts = []
        if engine is not None:
            texts, timing = benchmark(lambda: ocr_predictions(engine, bgr), "cpu", args.repeats)
            summary["ocr"][case_id] = {"backend": "onnxruntime_cpu", "count": len(texts), "timing": timing}
            save_json(args.output / case_id / "ocr.json", {"texts": texts, "timing": timing})
        for device in args.devices:
            for mode in args.modes:
                out = args.output / case_id / device.replace(":", "_") / mode
                out.mkdir(parents=True, exist_ok=True)
                load_start = time.perf_counter()
                skip_below = cfg.get("sahi", {}).get("skip_if_max_side_below")
                sahi_skipped = mode == "sliced" and bool(skip_below) and max(height, width) < skip_below
                if mode == "whole" or sahi_skipped:
                    model = YOLO(str(weights))
                    call = lambda: whole_predictions(model, bgr, cfg["detector"], device)
                elif mode == "sliced":
                    from sahi import AutoDetectionModel
                    model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                               confidence_threshold=cfg["detector"]["confidence"],
                                                               device=device, image_size=cfg["detector"]["imgsz"])
                    model.model.overrides.update({"iou": cfg["detector"]["iou"], "max_det": cfg["detector"]["max_det"],
                                                  "half": cfg["detector"]["half"]})
                    call = lambda: sliced_predictions(model, rgb, cfg["sahi"])
                else:  # fused
                    model = YOLO(str(weights))
                    from sahi import AutoDetectionModel
                    sliced_conf = cfg.get("fusion", {}).get("sliced_confidence", cfg["detector"]["confidence"])
                    sliced_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                                      confidence_threshold=sliced_conf,
                                                                      device=device, image_size=cfg["detector"]["imgsz"])
                    sliced_model.model.overrides.update({"iou": cfg["detector"]["iou"], "max_det": cfg["detector"]["max_det"],
                                                         "half": cfg["detector"]["half"]})
                    call = lambda: fused_predictions(model, sliced_model, rgb, device, cfg)
                load_ms = (time.perf_counter() - load_start) * 1000
                detections, timing = benchmark(call, device, args.repeats)
                underlying = model.model if mode in ("whole", "fused") else model.model.model
                actual_device = str(next(underlying.parameters()).device)
                if device.startswith("cuda") and not actual_device.startswith("cuda"):
                    raise RuntimeError(f"Requested GPU but model is on {actual_device}")
                summary["detector_classes"] = underlying.names
                class_conf = cfg.get("detector", {}).get("class_confidence") or {}
                dropped_by_class_threshold = len(detections) - len(apply_class_confidence(detections, class_conf))
                detections = apply_class_confidence(detections, class_conf)
                detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0], d["class_name"]))
                partial, binding = bind_designators(detections, texts, width, height, cfg)
                errors, bounds = validate_output(partial, width, height)
                if errors or bounds:
                    raise RuntimeError({"format_errors": errors, "bounds": bounds})
                save_json(out / "detections.json", {"image": str(image_path), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                                                     "coordinate_origin": "top_left", "size": [width, height], "detections": detections,
                                                     "class_confidence_dropped": dropped_by_class_threshold,
                                                     "sahi_skipped_small_image": sahi_skipped, "timing": timing})
                save_json(out / "components_partial.json", partial)
                save_json(out / "binding_debug.json", binding)
                draw_overlay(rgb, detections, texts, out / "overlay.png")
                row = {"image": str(image_path), "case_id": case_id, "size": [width, height], "device": device, "mode": mode,
                       "detection_count": len(detections), "class_counts": dict(Counter(d["class_name"] for d in detections)),
                       "actual_model_device": actual_device, "torch_cpu_threads": torch.get_num_threads(),
                       "resolved_designator_count": binding["resolved_designator_count"], "model_load_ms": load_ms,
                       "class_confidence_dropped": dropped_by_class_threshold, "sahi_skipped_small_image": sahi_skipped,
                       "timing": timing, "format_valid": True, "output": str(out.resolve())}
                summary["runs"].append(row)
                save_json(args.output / "summary.json", summary)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                del model, call
                if mode == "fused":
                    del sliced_model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    if engine is not None:
        import onnxruntime
        summary["ocr_runtime_providers_available"] = onnxruntime.get_available_providers()
        summary["ocr_models"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                 for p in Path(importlib.metadata.distribution("rapidocr").locate_file("rapidocr/models")).glob("*.onnx")]
    save_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
