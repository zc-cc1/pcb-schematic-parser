"""Image + frozen detections -> OCR, component fields and review artifacts.

This entrypoint never reads paired annotations. --reuse-texts is a validated
cache of OCR predictions, not a reference/Golden input.
"""
import argparse
import hashlib
import json
import importlib.metadata
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pcb_parser.component_text import bind_components, run_ocr
from pcb_parser.text_recovery import recover_texts
from pcb_parser.dataset_converter import validate_output


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay(rgb, records, texts, path):
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 13)
    for text in texts:
        p = [tuple(x) for x in text["polygon"]]
        draw.line(p+[p[0]], fill="#bbbbbb", width=1)
    for record in records:
        b = record["bbox_top_left"]
        color = "#128239" if record["designator"] else "#d28a00"
        draw.rectangle(b, outline=color, width=2)
        title = record["designator"] or record["component_id"].replace("UNRESOLVED_DET_", "?")
        title += " " + (record["value"] or record["Name"] or "")
        draw.text((b[0], max(0, b[1]-15)), title, fill=color, font=font, stroke_width=1, stroke_fill="white")
        for field in ("designator", "Name", "value"):
            token = record["fields"].get(field)
            if not token:
                continue
            tb = token["bbox"]
            draw.rectangle(tb, outline="#1665c0", width=2)
            draw.line([(sum(b[::2])/2, sum(b[1::2])/2), (sum(tb[::2])/2, sum(tb[1::2])/2)], fill="#1665c0", width=1)
    im.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--detector-run", type=Path, required=True)
    parser.add_argument("--detector-device", default="cpu", help="subdirectory name cpu or cuda_0")
    parser.add_argument("--mode", default="fused", choices=["whole", "sliced", "fused"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_y11s.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-texts", action="store_true")
    parser.add_argument("--base-texts-run", type=Path, help="Reuse image-validated base OCR and run local recovery again")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ocr_cfg = cfg.get("component_text", {}).get("ocr", {})
    signature = hashlib.sha256(json.dumps(ocr_cfg, sort_keys=True).encode()).hexdigest()
    recovery_cfg = cfg.get("component_text", {}).get("recovery", {})
    recovery_signature = hashlib.sha256(json.dumps(recovery_cfg,sort_keys=True).encode()).hexdigest()
    if args.reuse_texts and args.base_texts_run:
        parser.error("--reuse-texts and --base-texts-run are mutually exclusive")
    engine = None
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(args.config.resolve()),
               "config_sha256": sha(args.config), "ocr_config": ocr_cfg,
               "binding_source_sha256": sha(ROOT / "src/pcb_parser/component_text.py"),
               "recovery_source_sha256": sha(ROOT / "src/pcb_parser/text_recovery.py"),
               "recovery_config": recovery_cfg,
               "rapidocr_version": importlib.metadata.version("rapidocr"),
               "onnxruntime_version": importlib.metadata.version("onnxruntime"),
               "input_contract": "PNG + detector prediction JSON; no annotation/Golden file read",
               "status": "partial_components_OCR_binding_pins_nets_pending", "cases": []}
    for image_path in args.image:
        image_path = image_path.resolve()
        case = image_path.stem
        image_sha = sha(image_path)
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        h, w = rgb.shape[:2]
        detections_path = args.detector_run / case / args.detector_device / args.mode / "detections.json"
        detected = json.loads(detections_path.read_text(encoding="utf-8"))
        if detected["image_sha256"] != image_sha or detected["size"] != [w, h] or detected["coordinate_origin"] != "top_left":
            raise ValueError("Detector predictions do not match image/hash/coordinates")
        detector_summary_path = args.detector_run / "summary.json"
        detector_summary = json.loads(detector_summary_path.read_text(encoding="utf-8"))
        if detector_summary["weights_sha256"] != cfg.get("weights_sha256", detector_summary["weights_sha256"]):
            raise ValueError("Frozen detector run does not use the configured checkpoint")
        out = args.output / case
        cache = out / "texts.json"
        if args.reuse_texts:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached["image_sha256"] != image_sha or cached["ocr_config_sha256"] != signature:
                raise ValueError("OCR cache does not match image or OCR config")
            if recovery_cfg.get("enabled") and cached.get("recovery_config_sha256")!=recovery_signature:
                raise ValueError("OCR cache does not match local recovery config")
            texts, timing = cached["texts"], cached["timing"]
        else:
            if engine is None and (not args.base_texts_run or recovery_cfg.get("enabled")):
                from rapidocr import RapidOCR
                engine = RapidOCR(params={"Det.limit_side_len": cfg["ocr"]["det_limit_side_len"],
                                          "Det.limit_type": "max", "Global.max_side_len": cfg["ocr"]["det_limit_side_len"]})
            if args.base_texts_run:
                base = json.loads((args.base_texts_run/case/"texts.json").read_text(encoding="utf-8"))
                if base["image_sha256"]!=image_sha or base["ocr_config_sha256"]!=signature:
                    raise ValueError("Base OCR cache does not match PNG or base OCR configuration")
                texts,timing=base["texts"],base["timing"].copy()
                timing["base_cache_source"]=str(args.base_texts_run.resolve())
            else:
                texts, timing = run_ocr(rgb, engine, ocr_cfg)
            save(out/"base_texts.json",{"image_sha256":image_sha,"ocr_config_sha256":signature,
                                        "coordinate_origin":"top_left","texts":texts,"timing":timing.copy()})
            if recovery_cfg.get("enabled"):
                texts,recovery= recover_texts(rgb,engine,texts,recovery_cfg)
                timing["recovery_seconds"]=recovery["wall_seconds"]
                timing["base_ocr_seconds"]=timing["wall_seconds"]
                timing["wall_seconds"]+=recovery["wall_seconds"]
                save(out/"recovery_debug.json",recovery)
            save(cache, {"image_sha256": image_sha, "ocr_config_sha256": signature,
                         "recovery_config_sha256": recovery_signature,
                         "coordinate_origin": "top_left", "texts": texts, "timing": timing})
        started = time.perf_counter()
        partial, records, debug = bind_components(detected["detections"], texts, w, h, cfg)
        binding_seconds = time.perf_counter()-started
        errors, bounds = validate_output(partial, w, h)
        if errors or bounds:
            raise ValueError({"schema_errors": errors, "out_of_bounds": bounds})
        save(out / "components_partial.json", partial)
        save(out / "component_records.json", {"image_sha256": image_sha, "records": records})
        save(out / "binding_debug.json", debug)
        overlay(rgb, records, texts, out / "overlay.png")
        row = {"case_id": case, "detections": str(detections_path.resolve()), "detections_sha256": sha(detections_path),
               "image_sha256": image_sha, "size": [w, h], **debug["counts"], "ocr_texts": len(texts),
               "detector_summary_sha256": sha(detector_summary_path),
               "weights_sha256": detector_summary["weights_sha256"],
               "ocr_seconds": timing["wall_seconds"], "binding_seconds": binding_seconds,
               "ocr_cache_reused": args.reuse_texts, "format_valid": True, "output": str(out.resolve())}
        row["base_ocr_cache_reused"]=bool(args.base_texts_run)
        row["recovery_seconds"]=timing.get("recovery_seconds",0.)
        summary["cases"].append(row)
        save(args.output / "summary.json", summary)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
