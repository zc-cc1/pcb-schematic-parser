"""Unit tests for export-time component geometry (canonical size + centre offset)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser import export_geometry as eg

CFG_PATH = ROOT / "configs/bbox_convention_v1.json"


def table(**kw):
    return kw


def test_disabled_by_default():
    assert eg.load_table({}) == (None, None)
    assert eg.load_table({"bbox_convention": {"enabled": False, "path": str(CFG_PATH)}}) == (None, None)


def test_loads_shipped_config():
    mode, tbl = eg.load_table({"bbox_convention": {"enabled": True, "mode": "canonical",
                                                   "path": "configs/bbox_convention_v1.json"}},
                              root=ROOT)
    assert mode == "canonical"
    assert tbl["resistor"]["w"] == 8.0 and tbl["resistor"]["h"] == 20.0


def test_missing_file_is_off_not_crash():
    mode, tbl = eg.load_table({"bbox_convention": {"enabled": True, "path": "configs/nope.json"}},
                              root=ROOT)
    assert mode is None and tbl is None


def test_canonical_replaces_size_keeps_centre():
    tbl = table(resistor={"w": 8.0, "h": 20.0, "w_v": 20.0, "h_v": 8.0, "dxf": 0.0, "dyf": 0.0})
    box, proof = eg.apply_to_box([100, 100, 128, 180], "resistor", "canonical", tbl)
    # The variant follows the DETECTION's orientation compared with the base entry's:
    # a portrait detection (h>w) of a portrait-base class keeps the base 8x20.  The
    # old test asserted 20x8 here, i.e. it encoded the transposition bug.
    assert proof["size"] == [8.0, 20.0]
    cx, cy = (box[0] + box[2]) / 2., (box[1] + box[3]) / 2.
    assert (round(cx, 3), round(cy, 3)) == (114.0, 140.0)


def test_canonical_horizontal_variant():
    tbl = table(capacitor={"w": 16.0, "h": 20.0, "w_v": 20.0, "h_v": 16.0, "dxf": 0.0, "dyf": 0.0})
    box, proof = eg.apply_to_box([0, 0, 60, 40], "capacitor", "canonical", tbl)
    # landscape detection vs portrait base -> the transposed entry 20x16
    assert proof["size"] == [20.0, 16.0]
    assert round(box[2] - box[0], 3) == 20.0 and round(box[3] - box[1], 3) == 16.0


def test_canonical_round_trips_a_landscape_passive():
    """A box already at the landscape canonical size must come out unchanged.

    This is the property whose absence cost cases 12/13: a 20x8 resistor was
    rewritten to 8x20, moving every corner 6 px past the 20 px rubric.
    """
    tbl = table(resistor={"w": 8.0, "h": 20.0, "w_v": 20.0, "h_v": 8.0, "dxf": 0.0, "dyf": 0.0})
    box, proof = eg.apply_to_box([100, 100, 120, 108], "resistor", "canonical", tbl)
    assert proof["size"] == [20.0, 8.0]
    assert [round(v, 3) for v in box] == [100.0, 100.0, 120.0, 108.0]


def test_offset_is_relative_to_the_detection_size():
    tbl = table(ground={"w": 20.0, "h": 19.0, "dxf": 0.0, "dyf": 0.5,
                        "clamp_to_detection": False})
    # 40 px tall detection -> centre moves down 20 px
    box, _ = eg.apply_to_box([0, 0, 40, 40], "ground", "canonical", tbl)
    cy = (box[1] + box[3]) / 2.
    assert round(cy, 3) == 40.0


def test_clamp_keeps_the_symbol_inside_the_detected_cell():
    # dyf 0.5 on a 40 px cell would put the 19 px symbol at y 30.5..49.5, i.e.
    # outside the cell; the clamp pins it to the cell's bottom edge instead.
    tbl = table(ground={"w": 20.0, "h": 19.0, "dxf": 0.0, "dyf": 0.5})
    box, proof = eg.apply_to_box([0, 0, 40, 40], "ground", "canonical", tbl)
    assert proof["clamped_to_detection"] is True
    assert box[3] <= 40.0 + 1e-9 and box[1] >= -1e-9


def test_variable_size_class_is_left_alone():
    tbl = table(ic={"w": 45.0, "h": 108.0})  # ICs are excluded from the shipped table
    mode, shipped = eg.load_table({"bbox_convention": {"enabled": True, "mode": "canonical",
                                                       "path": str(CFG_PATH)}}, root=ROOT)
    box, proof = eg.apply_to_box([10, 10, 300, 400], "ic", mode, shipped)
    assert proof is None
    assert box == [10, 10, 300, 400]


def test_raw_vocabulary_is_resolved():
    tbl = table(voltage_source={"w": 10.0, "h": 10.0, "dxf": 0.0, "dyf": 0.0})
    for kind in ("v", "vcc", "voltage_source"):
        _, proof = eg.apply_to_box([0, 0, 50, 50], kind, "canonical", tbl)
        assert proof and proof["size"] == [10.0, 10.0], kind


def test_affine_mode_still_works():
    tbl = table(resistor={"rw": 0.5, "rh": 0.5, "dxf": 0.0, "dyf": 0.0})
    box, proof = eg.apply_to_box([0, 0, 40, 40], "resistor", "affine", tbl)
    assert proof["mode"] == "affine"
    assert round(box[2] - box[0], 3) == 20.0


def test_unknown_class_is_passthrough():
    tbl = table(resistor={"w": 8.0, "h": 20.0})
    box, proof = eg.apply_to_box([1, 2, 3, 4], "current_source", "canonical", tbl)
    assert proof is None and box == [1, 2, 3, 4]


def test_degenerate_box_is_passthrough():
    tbl = table(resistor={"w": 8.0, "h": 20.0})
    box, proof = eg.apply_to_box([5, 5, 5, 5], "resistor", "canonical", tbl)
    assert proof is None


def test_summary_counts_applied_components():
    s = eg.summarise("canonical", [{"class": "resistor"}, {"class": "ground"}, {"class": "resistor"}])
    assert s["enabled"] is True and s["components_rewritten"] == 3
    assert s["classes"] == ["ground", "resistor"]
    assert eg.summarise(None, [])["enabled"] is False


def test_shipped_canonical_table_is_sane():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    canon = cfg["canonical"]
    for cls in ("resistor", "capacitor", "ground", "voltage_source", "inductor"):
        entry = canon[cls]
        assert 3 <= entry["w"] <= 200 and 3 <= entry["h"] <= 200, cls
        assert abs(entry["dxf"]) <= 0.5 and abs(entry["dyf"]) <= 0.6, cls
    # variable-size symbols must never get a canonical size
    for cls in ("ic", "switch", "opamp"):
        assert cls not in canon
