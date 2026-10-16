"""Class-aware Weighted Boxes Fusion (WBF) for merging detector outputs.

Whole-image and sliced (SAHI) predictions of the same schematic overlap heavily;
GREEDYNMM inside SAHI only merges within one run. WBF merges the two runs:
same-class boxes with IoU >= threshold fuse into one box whose coordinates are
the confidence-weighted mean and whose score is the mean of member scores.
Pure numpy, no torch dependency, deterministic ordering.
"""
import numpy as np


def _iou(a, b):
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    overlap_w = min(a[2], b[2]) - max(a[0], b[0])
    overlap_h = min(a[3], b[3]) - max(a[1], b[1])
    overlap = max(0.0, overlap_w) * max(0.0, overlap_h)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def wbf_merge(detections, iou_threshold=0.55):
    """Fuse a list of {"bbox": [x1,y1,x2,y2], "confidence": float,
    "class_id": int, "class_name": str} dicts.

    Same-class boxes fuse transitively: a box joins the first same-class
    cluster where it overlaps ANY member with IoU >= threshold (comparing
    against the running fused box alone would break on chains of shifted
    duplicates). Different classes never fuse. Returns detections sorted by
    descending confidence. Members are processed by descending confidence.
    """
    if not detections:
        return []
    boxes = [list(map(float, d["bbox"])) for d in detections]
    scores = [float(d["confidence"]) for d in detections]
    classes = [(int(d["class_id"]), d["class_name"]) for d in detections]
    order = sorted(range(len(detections)), key=lambda i: (-scores[i], classes[i][1], i))

    clusters = []  # each: {"members": [indices], "box": fused_xyxy}
    for idx in order:
        box, score, cls = boxes[idx], scores[idx], classes[idx]
        target = None
        for cluster in clusters:
            if cluster["class"] != cls:
                continue
            if any(_iou(boxes[m], box) >= iou_threshold for m in cluster["members"]):
                target = cluster
                break
        if target is None:
            clusters.append({"class": cls, "members": [idx], "box": list(box)})
            continue
        target["members"].append(idx)
        members = target["members"]
        weights = np.asarray([scores[m] for m in members], dtype=np.float64)
        coords = np.asarray([boxes[m] for m in members], dtype=np.float64)
        target["box"] = (weights[:, None] * coords).sum(axis=0) / weights.sum()

    merged = []
    for cluster in sorted(clusters, key=lambda c: -np.mean([scores[m] for m in c["members"]])):
        members = cluster["members"]
        conf = float(np.mean([scores[m] for m in members]))
        class_id, class_name = cluster["class"]
        merged.append({"bbox": [round(v, 3) for v in cluster["box"]], "confidence": round(conf, 6),
                       "class_id": class_id, "class_name": class_name,
                       "members": len(members)})
    return merged


def apply_class_confidence(detections, class_confidence):
    """Drop detections whose confidence is below a per-class threshold.

    class_confidence: {class_name: threshold}; classes absent from the mapping
    keep every detection (the global threshold was already applied at
    inference). Pure filter, no rescoring.
    """
    if not class_confidence:
        return list(detections)
    kept = [d for d in detections if d["confidence"] >= float(class_confidence.get(d["class_name"], 0.0))]
    return kept
