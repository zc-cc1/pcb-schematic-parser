"""Unit tests for the official-rubric component evaluator and bbox-convention fit."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_official_component import (class_matches, corner_error, flip, iou,
                                         match_case, official_type, shrink_bbox)


def comp(key, cls, bbox, value=None, name=None, raw_type=None):
    return {"key": key, "class_name": cls, "raw_type": raw_type or cls,
            "bbox": bbox, "Name": name, "value": value}


def pred(cls, bbox, key=None, value=None, name=None):
    return {"key": key, "class_name": cls, "bbox": bbox, "Name": name, "value": value}


def test_flip_roundtrip():
    box = [10.0, 20.0, 30.0, 50.0]
    assert flip(flip(box, 100), 100) == box


def test_flip_maps_bottom_left_to_top_left():
    # A 20 px tall box sitting 5 px above the bottom edge of a 100 px image.
    assert flip([0, 5, 20, 25], 100) == [0, 75, 20, 95]


def test_official_type_normalisation():
    assert official_type("box") == "ic"
    assert official_type("crystal_4pin") == "crystal"
    assert official_type("m3螺丝") == "raw_m"
    assert official_type("net_bidirection") == "net_bidirectional"
    assert official_type("amp_5pin") == "opamp"
    assert official_type("not_a_type") is None


def test_class_alias_only_where_official_allows():
    # documented leniency: ICs and power symbols, spelled either way
    assert class_matches("ground", "gnd") and class_matches("gnd", "gnd")
    assert class_matches("vcc", "v") and class_matches("v", "v")
    assert class_matches("ic", "box") and class_matches("box", "ic")
    # everything else is compared as spelled: "resistor" is NOT "r"
    assert not class_matches("resistor", "r")
    assert not class_matches("capacitor", "c")
    assert not class_matches("resistor", "ground")
    assert not class_matches("vbus_in", "testpoint")


def test_shrink_is_about_the_centre():
    # 40x40 box at (100,100)-(140,140): half width/height, centre moved down 0.25*h
    out = shrink_bbox([100, 100, 140, 140], 0.5, 0.5, 0.0, 0.25)
    assert out == [110, 120, 130, 140]


def test_corner_error_is_max_coordinate():
    assert corner_error([0, 0, 10, 10], [1, 2, 3, 25]) == 15


def test_match_accepts_within_tolerance_only():
    gts = [comp("R1", "resistor", [100, 100, 108, 120], value="1k")]
    inside = [pred("resistor", [105, 105, 113, 125], key="R1", value="1k")]
    outside = [pred("resistor", [121, 100, 129, 120], key="R1", value="1k")]
    assert match_case(gts, [dict(p) for p in inside], "class", 20)[:3] == (1, 0, 0)
    assert match_case(gts, [dict(p) for p in outside], "class", 20)[:3] == (0, 1, 1)


def test_power_match_ignores_key_but_not_type():
    gts = [comp("∅97", "ground", [316, 462, 336, 481], raw_type="gnd")]
    renamed = [pred("ground", [316, 462, 336, 481], key="UNRESOLVED_DET_0003")]
    right_type_wrong_place = [pred("voltage_source", [316, 462, 336, 481])]
    assert match_case([dict(g) for g in gts], [dict(p) for p in renamed], "key", 20)[:3] == (1, 0, 0)
    assert match_case([dict(g) for g in gts],
                      [dict(p) for p in right_type_wrong_place], "key", 20)[:3] == (0, 1, 1)


def test_key_mode_requires_designator_for_ordinary_components():
    gts = [comp("C1", "capacitor", [0, 0, 16, 20], value="100nF")]
    unresolved = [pred("capacitor", [0, 0, 16, 20], key="UNRESOLVED_DET_0001", value="100nF")]
    named = [pred("capacitor", [0, 0, 16, 20], key="C1", value="100nF")]
    # class mode is an upper bound: it credits an unresolved designator
    assert match_case([dict(g) for g in gts], [dict(p) for p in unresolved], "class", 20)[:3] == (1, 0, 0)
    assert match_case([dict(g) for g in gts], [dict(p) for p in unresolved], "key", 20)[:3] == (0, 1, 1)
    assert match_case([dict(g) for g in gts], [dict(p) for p in named], "key", 20)[:3] == (1, 0, 0)


def test_one_to_one_matching_does_not_reuse_a_prediction():
    gts = [comp("C1", "capacitor", [0, 0, 16, 20]), comp("C2", "capacitor", [20, 0, 36, 20])]
    only_one = [pred("capacitor", [0, 0, 16, 20], key="C1")]
    tp, fp, fn, _ = match_case([dict(g) for g in gts], [dict(p) for p in only_one], "class", 20)
    assert (tp, fp, fn) == (1, 0, 1)


def test_shipped_config_shape():
    cfg = json.loads((ROOT / "configs/bbox_convention_v1.json").read_text(encoding="utf-8"))
    transform = cfg["transform"]
    assert "resistor" in transform and "capacitor" in transform
    # power symbols and ICs must never be affine-shrunk by this table
    assert "ground" not in transform and "voltage_source" not in transform
    assert "ic" not in transform
    for name, spec in transform.items():
        assert set(spec) == {"rw", "rh", "dxf", "dyf"}, name
        assert 0.3 <= spec["rw"] <= 1.2 and 0.3 <= spec["rh"] <= 1.2
