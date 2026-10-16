"""Offline text-owner evaluation against manually reviewed development fields.

Predicted class + IoU>=0.5 matches component ownership; field strings are exact.
Missing detections/fields remain in end-to-end denominators. This is not the
competition's ComponentF1 and references are never passed to the inference code.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
from evaluate_detector import bbox_iou
from run_baseline import bind_designators


def compare(records, expected, threshold=.5):
    eligible = {"designator": [], "value": [], "Name": []}
    costs = np.asarray([[bbox_iou(p["bbox_top_left"], g["bbox_top_left"]) if p["type"] == g["type"] else 0
                         for g in expected] for p in records])
    matched = {}
    if costs.size:
        reward = (costs >= threshold)*(min(costs.shape)+1)+costs
        pi, gi = linear_sum_assignment(-reward)
        matched = {int(g): int(p) for p,g in zip(pi,gi) if costs[p,g] >= threshold}
    details = []
    for gidx, reference in enumerate(expected):
        prediction = records[matched[gidx]] if gidx in matched else None
        for field, value in reference["fields"].items():
            actual = prediction[field] if prediction else None
            eligible[field].append({"correct": actual == value, "matched_detection": prediction is not None})
            details.append({"reference": reference["designator"], "field": field, "expected": value, "predicted": actual,
                            "correct": actual == value, "failure": None if actual == value else
                            ("detector_missed_or_wrong_class" if prediction is None else "empty_field" if actual is None else "wrong_text_or_owner"),
                            "detection_index": prediction["detection_index"] if prediction else None})
    fields = {}
    for field, rows in eligible.items():
        correct = sum(r["correct"] for r in rows)
        supported = sum(r["matched_detection"] for r in rows)
        fields[field] = {"correct": correct, "expected": len(rows), "exact_match_rate": correct/len(rows) if rows else None,
                         "matched_detection_fields": supported,
                         "exact_rate_given_matched_detection": correct/supported if supported else None}
    assigned = sum(r["designator"] is not None for r in records)
    fields["designator"].update({"assigned": assigned, "binding_precision": fields["designator"]["correct"]/assigned if assigned else None})
    return fields, details


def aggregate(rows):
    output = {}
    for field in ["designator","value","Name"]:
        counts = Counter()
        for row in rows:
            for key, value in row["fields"][field].items():
                if key in {"correct","expected","matched_detection_fields","assigned"}:
                    counts[key] += value
        output[field] = {**counts, "exact_match_rate": counts["correct"]/counts["expected"] if counts["expected"] else None,
                         "exact_rate_given_matched_detection": counts["correct"]/counts["matched_detection_fields"] if counts["matched_detection_fields"] else None}
        if field == "designator":
            output[field]["binding_precision"] = counts["correct"]/counts["assigned"] if counts["assigned"] else None
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,default=ROOT/"runs/s3_component_text_quick_dev")
    p.add_argument("--detector-run",type=Path,default=ROOT/"runs/s3_v3_detector_quick_dev")
    p.add_argument("--reference",type=Path,default=ROOT/"data/reviews/component_text_quick_dev.json")
    p.add_argument("--config",type=Path,default=ROOT/"configs/baseline_v3_adopted.json")
    p.add_argument("--output",type=Path,default=ROOT/"runs/s3_component_text_quick_dev/evaluation.json")
    p.add_argument("--baseline-run",type=Path,help="Compare with saved earlier component records instead of the legacy binder")
    p.add_argument("--skip-legacy",action="store_true")
    args=p.parse_args()
    reference=json.loads(args.reference.read_text(encoding="utf-8"))
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    result={"status":"manual_local_text_owner_proxy_not_official_competition_score",
            "reference":str(args.reference.resolve()),"reference_sha256":hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "field_count":reference["field_count"], "development_set_reused":True,
            "ownership_matching":"type + bbox IoU>=0.5, maximum-cardinality one-to-one",
            "string_matching":"case/unit/characters exact; no normalization", "cases":[],"legacy_cases":[]}
    for case, truth in reference["cases"].items():
        records=json.loads((args.run/case/"component_records.json").read_text(encoding="utf-8"))
        if records["image_sha256"]!=truth["image_sha256"]:
            raise ValueError("Reference and predictions refer to different PNGs")
        fields, details=compare(records["records"],truth["components"])
        result["cases"].append({"case_id":case,"fields":fields,"details":details})
        if args.skip_legacy:
            continue
        if args.baseline_run:
            before=json.loads((args.baseline_run/case/"component_records.json").read_text(encoding="utf-8"))
            if before["image_sha256"]!=truth["image_sha256"]:
                raise ValueError("Baseline and reference use different images")
            fields,details=compare(before["records"],truth["components"])
            result["legacy_cases"].append({"case_id":case,"fields":fields,"details":details})
            continue
        # Prior whole-image OCR + old nearest-ref binder, with exactly the same
        # frozen v3 detections. This comparison does not change the detector.
        detections=json.loads((args.detector_run/case/"cpu/fused/detections.json").read_text(encoding="utf-8"))["detections"]
        texts=json.loads((ROOT/f"runs/s2_balanced_quick_dev/{case}/ocr.json").read_text(encoding="utf-8"))["texts"]
        w,h=truth["size"]
        old, provenance=bind_designators(detections,texts,w,h,cfg)
        old_records=[]
        for i,item in enumerate(provenance["provisional_bindings"]):
            c=old["components"][item["component_id"]]
            old_records.append({"detection_index":i,"type":c["type"],"bbox_top_left":item["detection"]["bbox"],
                                "designator":item["binding"]["designator"] if item["binding"] else None,
                                "Name":c["Name"],"value":c["value"]})
        fields,details=compare(old_records,truth["components"])
        result["legacy_cases"].append({"case_id":case,"fields":fields,"details":details})
    result["overall"]=aggregate(result["cases"])
    result["legacy_overall"]=aggregate(result["legacy_cases"])
    result["comparison_source"]=str(args.baseline_run.resolve()) if args.baseline_run else "legacy_whole_OCR_and_nearest_reference_binder"
    result["error_counts"]=dict(Counter(d["failure"] for c in result["cases"] for d in c["details"] if d["failure"]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["field_count","overall","legacy_overall","error_counts"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
