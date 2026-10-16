"""Unit tests for the class-aware WBF fusion used by the dual-path detector."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.fusion import apply_class_confidence, wbf_merge


def det(bbox, conf, name, cid=0):
    return {"bbox": bbox, "confidence": conf, "class_id": cid, "class_name": name}


def test_empty_input():
    assert wbf_merge([]) == []


def test_single_detection_passthrough():
    merged = wbf_merge([det([10, 10, 50, 50], 0.9, "resistor")])
    assert len(merged) == 1
    assert merged[0]["bbox"] == [10.0, 10.0, 50.0, 50.0]
    assert merged[0]["members"] == 1


def test_same_class_high_iou_fuses():
    a = det([10, 10, 50, 50], 0.9, "resistor")
    b = det([12, 11, 52, 51], 0.7, "resistor")  # IoU ~0.72
    merged = wbf_merge([a, b], iou_threshold=0.55)
    assert len(merged) == 1
    assert merged[0]["members"] == 2
    # Coordinates are the confidence-weighted mean, closer to the 0.9 box.
    assert 10.0 < merged[0]["bbox"][0] < 12.0
    assert abs(merged[0]["confidence"] - 0.8) < 1e-6


def test_low_iou_stays_separate():
    a = det([10, 10, 50, 50], 0.9, "resistor")
    b = det([100, 100, 140, 140], 0.8, "resistor")
    assert len(wbf_merge([a, b], iou_threshold=0.55)) == 2


def test_different_classes_never_fuse():
    a = det([10, 10, 50, 50], 0.9, "resistor", cid=1)
    b = det([10, 10, 50, 50], 0.8, "capacitor", cid=2)
    merged = wbf_merge([a, b], iou_threshold=0.55)
    assert len(merged) == 2
    assert {m["class_name"] for m in merged} == {"resistor", "capacitor"}


def test_three_way_chain_fuses_to_one_cluster():
    a = det([0, 0, 40, 40], 0.9, "ic", cid=3)
    b = det([8, 0, 48, 40], 0.8, "ic", cid=3)   # IoU(a,b) ~0.67
    c = det([16, 0, 56, 40], 0.7, "ic", cid=3)  # IoU(b,c) ~0.67, IoU(a,c) ~0.43
    merged = wbf_merge([a, b, c], iou_threshold=0.55)
    assert len(merged) == 1
    assert merged[0]["members"] == 3
    assert 0 < merged[0]["bbox"][0] < 16


def test_output_sorted_by_confidence():
    merged = wbf_merge([det([10, 10, 50, 50], 0.6, "resistor"),
                        det([100, 100, 140, 140], 0.95, "capacitor", cid=2),
                        det([200, 200, 240, 240], 0.75, "ic", cid=3)])
    confs = [m["confidence"] for m in merged]
    assert confs == sorted(confs, reverse=True)


def test_apply_class_confidence_filters_only_mapped_classes():
    detections = [det([0, 0, 10, 10], 0.30, "resistor"),
                  det([5, 5, 15, 15], 0.45, "crystal", cid=7),
                  det([9, 9, 19, 19], 0.60, "ic", cid=3)]
    kept = apply_class_confidence(detections, {"resistor": 0.4, "crystal": 0.4})
    assert [d["class_name"] for d in kept] == ["crystal", "ic"]
    assert apply_class_confidence(detections, {}) == detections
