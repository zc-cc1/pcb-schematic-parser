"""Contract regressions for annotation conversion; fixtures are not evaluation data."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from pcb_parser.dataset_converter import Converter, load_source_annotation, normalize_image


ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "configs/conversion.json").read_text(encoding="utf-8"))


def component(designator, pin_defs, category="R", label=None):
    texts = {"identity": {"content": designator, "details": "designator"}} if designator else {}
    if label:
        texts["label"] = {"content": label, "details": "netLabel"}
    return {"state": "active", "category": category, "location": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "associatedTexts": texts, "pins": {pid: {"tip": list(tip), "base": [tip[0] - 2, tip[1]], "pinNumberByOCR": n, "pinNameByOCR": "P"} for pid, n, tip in pin_defs}}


def dataset(components, edges=None):
    return {"canvas": {"width": 100, "height": 100}, "components": components, "pins": {}, "texts": {}, "edges": edges or {}}


def edge(owners, path):
    return {"state": "active", "connect": {pid: {"pinId": pid, "cpntId": cid} for cid, pid in owners}, "keyPointPaths": {"wire": [path]}}


class ConversionRegressions(unittest.TestCase):
    def convert(self, data, policy=None):
        output, report = Converter(data, "fixture", policy or POLICY).run()
        self.assertEqual(report["validation_errors"], [])
        return output, report

    def test_origin_bbox_and_tip_are_converted_once(self):
        data = dataset({"C_1": component("R1", [("P_1", "1", (30, 25))])})
        out, _ = self.convert(data)
        self.assertEqual(out["components"]["R1"]["bbox"], [10, 60, 30, 80])
        self.assertEqual(out["pins"]["R1"]["pin_1"]["point"], {"x": 30, "y": 75})
        policy = {**POLICY, "export_y_origin": "height_minus_one"}
        out, _ = self.convert(data, policy)
        self.assertEqual(out["components"]["R1"]["bbox"], [10, 59, 30, 79])

    def test_duplicate_component_suffixes_use_natural_ids(self):
        data = dataset({"C_10": component("U1", [("P_10", None, (20, 20))], "Box"), "C_2": component("U1", [("P_10", None, (21, 20)), ("P_2", None, (22, 20))], "Box")})
        out, report = self.convert(data)
        self.assertEqual(report["component_map"]["C_2"]["designator"], "U1_1")
        self.assertEqual(report["component_map"]["C_10"]["designator"], "U1_2")
        self.assertEqual(out["pins"]["U1_1"]["pin_1"]["point"]["x"], 22)

    def test_multi_number_pin_and_bent_wire(self):
        data = dataset({"C_1": component("U1", [("P_1", "5,6", (10, 20)), ("P_2", None, (10, 30))], "Box"), "C_2": component("R1", [("P_3", "9", (30, 40))])}, {"E_1": edge([("C_1", "P_1"), ("C_2", "P_3")], [[10, 20], [30, 20], [30, 40]])})
        out, _ = self.convert(data)
        net = next(iter(out["nets"].values()))
        self.assertEqual(set(net["hyperGraph"][1:-1].split(",")), {"U1.5", "U1.6", "R1.9"})
        self.assertEqual(len(net["edges"]), 2)
        self.assertIn("pin_1", out["pins"]["U1"])

    def test_direct_pin_contact_generates_point_edge(self):
        data = dataset({"C_1": component("R1", [("P_1", "1", (40, 50))]), "C_2": component("R2", [("P_2", "2", (40, 50))])})
        out, _ = self.convert(data)
        net = next(iter(out["nets"].values()))
        self.assertEqual(net["hyperGraph"], "(R1.1,R2.2)")
        self.assertEqual(net["edges"]["edge_1"], [{"x": 40, "y": 50}] * 2)

    def test_dangling_reference_and_deleted_objects_do_not_leak(self):
        data = dataset({"C_1": component("R1", [("P_1", "1", (10, 20))]), "C_2": {**component("R2", []), "state": "deleted"}}, {"E_1": edge([("C_1", "P_1"), ("C_1", "P_999")], [[10, 20], [30, 20]]), "E_2": {**edge([], [[0, 0], [90, 90]]), "state": "deleted"}})
        out, report = self.convert(data)
        self.assertEqual(set(out["components"]), {"R1"})
        self.assertEqual(next(iter(out["nets"].values()))["hyperGraph"], "(R1.1)")
        self.assertEqual(report["warning_counts"]["unresolved_edge_pin_omitted"], 1)

    def test_signal_labels_and_power_use_different_geometry(self):
        for power in [False, True]:
            with self.subTest(power=power):
                category, label = ("Gnd", None) if power else ("NET_input", "DATA")
                data = dataset({"C_1": component(None, [("P_1", "1", (10, 20))], category, label), "C_2": component(None, [("P_2", "1", (30, 40))], category, label), "C_3": component("R1", [("P_3", "1", (20, 20))]), "C_4": component("R2", [("P_4", "1", (40, 40))])}, {"E_1": edge([("C_1", "P_1"), ("C_3", "P_3")], [[10, 20], [20, 20]]), "E_2": edge([("C_2", "P_2"), ("C_4", "P_4")], [[30, 40], [40, 40]])})
                out, _ = self.convert(data)
                self.assertEqual(len(out["nets"]), 1)
                edges = next(iter(out["nets"].values()))["edges"]
                self.assertEqual(len(edges), 2)
                self.assertEqual(all(p[0] == p[1] for p in edges.values()), not power)

    def test_wrong_suffix_is_normalized_without_changing_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "disguised.png", Path(temp) / "normalized.png"
            Image.new("RGB", (8, 6), (100, 150, 200)).save(source, format="JPEG")
            result = normalize_image(source, target)
            self.assertEqual(result["source_format"], "JPEG")
            self.assertTrue(result["rgba_pixels_equal"])
            with Image.open(target) as image:
                self.assertEqual((image.format, image.size), ("PNG", (8, 6)))

    def test_identical_source_duplicate_is_logged_and_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.json"
            source.write_text('{"connect":{"P_1":{"pinId":"P_1"},"P_1":{"pinId":"P_1"}}}', encoding="utf-8")
            data, duplicates = load_source_annotation(source)
            self.assertEqual(len(data["connect"]), 1)
            self.assertEqual(duplicates[0]["source_pointer"], "/connect/P_1")
            source.write_text('{"connect":{"P_1":{"x":1},"P_1":{"x":2}}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                load_source_annotation(source)

    def test_global_pin_without_owner_is_reported_without_invented_component(self):
        data = dataset({"C_1": component("R1", [("P_1", "1", (10, 20))])})
        data["pins"]["P_99"] = {"tip": [90, 90], "edgeId": "E_missing"}
        out, report = self.convert(data)
        self.assertEqual(set(out["components"]), {"R1"})
        self.assertEqual(report["warning_counts"]["unowned_global_pin_omitted"], 1)


if __name__ == "__main__":
    unittest.main()
