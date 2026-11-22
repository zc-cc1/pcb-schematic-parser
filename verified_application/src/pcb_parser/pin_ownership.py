"""Associate learned pin points with predicted components, without annotations.

This is relation construction, not pin localization: no point is moved to a
box edge, no pin count or number is guessed, and ambiguous candidates remain.
"""
from collections import Counter
import math

import numpy as np


def normalized_components(detections, class_mapping, records=None):
    records_by_index = {r["detection_index"]: r for r in (records or [])}
    if len(records_by_index)!=len(records or []) or any(i not in range(len(detections)) for i in records_by_index):
        raise ValueError("OCR records contain duplicate or unknown detection indices")
    output = []
    for index, detection in enumerate(detections):
        box = list(map(float, detection["bbox"]))
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid predicted component box")
        record = records_by_index.get(index)
        if record and not np.allclose(record["bbox_top_left"], box, atol=0.05, rtol=0):
            raise ValueError("OCR component records do not match frozen detection indices/boxes")
        output.append({"component_id": record["component_id"] if record else f"DET_{index + 1:04d}",
                       "detection_index": index, "class_name": detection["class_name"],
                       "type": class_mapping.get(detection["class_name"], detection["class_name"]),
                       "bbox_top_left": box, "confidence": float(detection["confidence"]),
                       "designator": record.get("designator") if record else None,
                       "identity_status": "OCR_prediction" if record and record.get("designator") else "unresolved_detection_id"})
    if len({c["component_id"] for c in output}) != len(output):
        raise ValueError("Predicted component IDs must be unique")
    return output


def relation(point, component, cfg):
    x, y = point
    x1, y1, x2, y2 = component["bbox_top_left"]
    short = min(x2 - x1, y2 - y1)
    outside = math.hypot(max(x1 - x, x - x2, 0), max(y1 - y, y - y2, 0))
    margin = max(cfg["outer_margin_min_px"], min(cfg["outer_margin_max_px"], short * cfg["outer_margin_fraction"]))
    if outside > margin:
        return None
    inside = x1 <= x <= x2 and y1 <= y <= y2
    boundary_distance = min(x - x1, x2 - x, y - y1, y2 - y) if inside else 0.
    warning = None
    if component["type"] in {"ic", "connector", "block", "opamp"}:
        band = max(cfg["large_body_min_band_px"], min(cfg["large_body_max_band_px"], short * cfg["large_body_band_fraction"]))
        if inside and boundary_distance > band:
            warning = "deep_inside_large_component"
        corner_distance = min(math.hypot(x - px, y - py) for px, py in [(x1,y1),(x1,y2),(x2,y1),(x2,y2)])
        if corner_distance <= cfg["large_body_corner_caution_px"]:
            warning = "large_component_corner_needs_review"
    score = outside + cfg["inside_boundary_cost_weight"] * boundary_distance + cfg["detection_confidence_cost_px"] * (1 - component["confidence"])
    return {"component_id": component["component_id"], "detection_index": component["detection_index"],
            "component_type": component["type"], "cost_px": float(score), "outside_gap_px": outside,
            "inside_boundary_distance_px": float(boundary_distance), "eligible_margin_px": margin,
            "warning": warning}


def associate_pins(candidates, components, cfg, size, wire_probability=None):
    width, height = size
    if wire_probability is not None and (wire_probability.shape != (height,width) or not np.isfinite(wire_probability).all()):
        raise ValueError("Wire probability map does not match the image")
    results = []
    for index, original in enumerate(candidates):
        x, y = map(float, original["point"])
        if not all(math.isfinite(v) for v in [x,y,original["confidence"]]) or not 0 <= x < width or not 0 <= y < height:
            raise ValueError("Invalid learned pin coordinate or confidence")
        proposals = sorted((p for c in components if (p := relation((x,y),c,cfg)) is not None),
                           key=lambda p:(bool(p["warning"]),p["cost_px"],p["detection_index"]))
        plausible = [p for p in proposals if not p["warning"]]
        status, owner = "no_component_nearby", None
        if plausible:
            best = plausible[0]
            if len(plausible)>1 and plausible[1]["cost_px"]-best["cost_px"] < cfg["ambiguity_margin_px"]:
                status = "ambiguous_owner"
            else:
                status,owner = "assigned",best
        elif proposals:
            status = proposals[0]["warning"]
        item = {"candidate_id": f"PIN_CAND_{index+1:04d}", "point":[x,y], "confidence":float(original["confidence"]),
                "owner": owner["component_id"] if owner else None,
                "owner_detection_index": owner["detection_index"] if owner else None,
                "owner_status":status, "owner_candidates":proposals[:cfg["saved_owner_candidates"]],
                "number":None,"pinname":None,"number_status":"requires_OCR_or_verified_symbol_evidence"}
        if wire_probability is not None:
            ix,iy=round(x),round(y)
            radius=cfg["wire_context_radius_px"]
            region=wire_probability[max(0,iy-radius):min(height,iy+radius+1),max(0,ix-radius):min(width,ix+radius+1)]
            item["nearby_wire_probability_max"]=float(region.max()) if region.size else 0.
            item["wire_context_status"]="learned_wire_nearby" if item["nearby_wire_probability_max"]>=cfg["wire_context_threshold"] else "no_confident_wire_observed"
        results.append(item)
    counts=Counter(p["owner_status"] for p in results)
    return {"coordinate_origin":"top_left", "size":[width,height], "components":components,"pin_candidates":results,
            "summary":{"candidate_count":len(results),"assigned":counts["assigned"],"status_counts":dict(counts),
                       "all_candidates_preserved":len(results)==len(candidates),
                       "coordinate_changes":0,"pin_numbers_guessed":0,"nets_constructed":False},
            "status":"component_ownership_proposals_number_name_topology_pending",
            "input_contract":"model-predicted pin coordinates and component boxes only; no annotations",
            "traditional_image_segmentation_used":False,"reference_score":None}
