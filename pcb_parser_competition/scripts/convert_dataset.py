"""Batch conversion entrypoint for user-provided raw reference annotations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pcb_parser.dataset_converter import Converter, digest, load_json, load_source_annotation, normalize_image, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/manifest.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/conversion.json")
    parser.add_argument("--overrides", type=Path, default=PROJECT_ROOT / "configs/annotation_overrides.json")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--y-origin", choices=["height", "height_minus_one"])
    parser.add_argument("--flywire-geometry", choices=["label_points", "preserve_drawn"])
    args = parser.parse_args()
    policy, override_data = load_json(args.config), load_json(args.overrides)
    if args.y_origin:
        policy["export_y_origin"] = args.y_origin
    if args.flywire_geometry:
        policy["flywire_geometry"] = args.flywire_geometry
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.case_id:
        rows = [r for r in rows if r["case_id"] in set(args.case_id)]
        missing = set(args.case_id) - {r["case_id"] for r in rows}
        if missing:
            parser.error(f"Unknown case IDs: {sorted(missing)}")
    output_root = PROJECT_ROOT / "data/standardized"
    report_root = PROJECT_ROOT / "data/conversion_reports"
    completed, failed, totals, warnings = [], [], Counter(), Counter()
    for row in rows:
        case_id = row["case_id"]
        try:
            source, image = Path(row["raw_annotation_path"]), Path(row["image_path"])
            source_hash, image_hash = digest(source), digest(image)
            if source_hash != row["annotation_sha256"] or image_hash != row["image_sha256"]:
                raise ValueError("Input changed since intake manifest; audit again before conversion")
            data, duplicates = load_source_annotation(source)
            converter = Converter(data, case_id, policy, override_data.get("cases", {}).get(case_id, {}))
            for duplicate in duplicates:
                converter.warn("identical_duplicate_json_key_removed", **duplicate)
            output, report = converter.run()
            if report["validation_errors"]:
                save_json(report_root / f"{case_id}.json", report)
                raise ValueError(f"{len(report['validation_errors'])} project contract errors")
            target = output_root / case_id / f"{case_id}.json"
            target_image = output_root / case_id / f"{case_id}.png"
            image_record = normalize_image(image, target_image)
            save_json(target, output)
            # Confirm conversion has not modified the user-provided source files.
            if digest(source) != source_hash or digest(image) != image_hash:
                raise ValueError("Source integrity check failed")
            report["inputs"] = {"image": str(image), "annotation": str(source), "image_sha256": image_hash, "annotation_sha256": source_hash}
            output_hash = digest(target)
            report["outputs"] = {"image": str(target_image), "annotation": str(target), "annotation_sha256": output_hash, "image_normalization": image_record}
            report["manual_overrides"] = override_data.get("cases", {}).get(case_id, {})
            report["policy"] = policy
            save_json(report_root / f"{case_id}.json", report)
            totals.update(report["counts"])
            warnings.update(report["warning_counts"])
            completed.append({"case_id": case_id, "image_path": str(target_image), "reference_json_path": str(target), "reference_json_sha256": output_hash, "report_path": str(report_root / f"{case_id}.json"), "status": report["status"], "counts": report["counts"], "warning_counts": report["warning_counts"], "official_golden_verified": False})
        except Exception as exc:
            failed.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
        if len(completed) + len(failed) <= 2 or (len(completed) + len(failed)) % 50 == 0:
            print(f"processed={len(completed)+len(failed)}/{len(rows)} converted={len(completed)} failed={len(failed)}", flush=True)
    run_id = "selected" if args.case_id else "all"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / ("manifest_selected.jsonl" if args.case_id else "manifest.jsonl")
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completed), encoding="utf-8")
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "attempted": len(rows), "converted": len(completed), "failed": failed, "counts": dict(totals), "warning_counts": dict(warnings), "policy": policy, "output_root": str(output_root), "manifest": str(manifest), "official_golden_verified": False, "official_f1": None, "source_integrity_checked": True, "all_output_png_pixels_checked": True, "cases": completed}
    save_json(report_root / f"summary_{run_id}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"cases", "policy"}}, ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
