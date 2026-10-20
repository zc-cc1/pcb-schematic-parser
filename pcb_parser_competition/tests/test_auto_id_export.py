"""Auto-id export and reading-order numbering must not silently change behaviour."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.pin_numbering import assign_numbers  # noqa: E402
from pcb_parser.wire_graph_artifacts import partial_submission  # noqa: E402


def _record(index, designator, bbox=(10.0, 10.0, 30.0, 40.0), kind="resistor"):
    return {"component_id": designator or f"DET_{index}", "detection_index": index,
            "type": kind, "designator": designator,
            "Name": designator, "value": "1k", "bbox_top_left": list(bbox),
            "fields": {}}


def _pin(candidate, owner_index, number, name, point):
    return {"candidate_id": candidate, "point": list(point), "owner_detection_index": owner_index,
            "owner": f"DET_{owner_index}", "owner_status": "assigned",
            "number": number, "pinname": name}


def _graph(networks):
    return {"edges": [], "networks": networks, "gap_candidates": []}


def _net(network_id, pin_ids):
    return {"network_id": network_id, "pin_candidate_ids": list(pin_ids), "edge_ids": [],
            "unresolved_junction_ids": []}


def _payload(components, pins):
    return {"size": [100, 100], "components": components, "pin_candidates": pins}


class AutoIdExportTest(unittest.TestCase):
    def test_unnamed_component_is_skipped_by_default(self):
        payload = _payload([_record(0, None)], [_pin("P1", 0, "1", "1", (12, 12))])
        out, coverage = partial_submission(payload, _graph([]), [_record(0, None)])
        self.assertEqual(out["components"], {})
        self.assertEqual(out["pins"], {})
        self.assertEqual(coverage["auto_id_count"], 0)

    def test_unnamed_component_exports_under_auto_id_when_enabled(self):
        payload = _payload([_record(7, None)], [_pin("P1", 7, "1", "1", (12, 12))])
        out, coverage = partial_submission(payload, _graph([]), [_record(7, None)],
                                           auto_id_for_unnamed=True, auto_id_prefix="DET")
        self.assertEqual(list(out["components"]), ["DET_0007"])
        # no designator was observed, so Name must stay null rather than be invented
        self.assertIsNone(out["components"]["DET_0007"]["Name"])
        self.assertEqual(out["components"]["DET_0007"]["type"], "resistor")
        self.assertEqual(coverage["auto_id_count"], 1)
        self.assertEqual(out["pins"]["DET_0007"]["pin_1"]["pinname"], "1")

    def test_auto_id_lets_the_net_export(self):
        records = [_record(7, None)]
        payload = _payload([_record(7, None)],
                           [_pin("P1", 7, "1", "1", (12, 12)), _pin("P2", 7, "2", "2", (12, 28))])
        graph = _graph([_net("net_1", ["P1", "P2"])])
        out, _ = partial_submission(payload, graph, records, auto_id_for_unnamed=True)
        self.assertIn("net_1", out["nets"])
        self.assertEqual(out["nets"]["net_1"]["hyperGraph"], "(DET_0007.1,DET_0007.2)")

    def test_named_component_still_uses_its_designator(self):
        records = [_record(0, "R5")]
        payload = _payload([_record(0, "R5")], [_pin("P1", 0, "1", "1", (12, 12))])
        out, coverage = partial_submission(payload, _graph([]), records, auto_id_for_unnamed=True)
        self.assertEqual(list(out["components"]), ["R5"])
        self.assertEqual(out["components"]["R5"]["Name"], "R5")
        self.assertEqual(coverage["auto_id_count"], 0)


class PinNumberingTest(unittest.TestCase):
    def test_numbering_direction_matches_the_calibrated_rule(self):
        """The rule sorts on descending image y (visually lower pin first), which is
        the order that scored higher end to end against the reference."""
        pins = [_pin("A", 0, None, None, (10, 20)), _pin("B", 0, None, None, (10, 60))]
        assigned, audit = assign_numbers(pins, [_record(0, "R1")])
        self.assertEqual(assigned["B"], "1")
        self.assertEqual(assigned["A"], "2")
        self.assertEqual(audit["pins_auto_numbered"], 2)

    def test_numbering_is_stable_and_unique_for_dense_rows(self):
        pins = [_pin(f"P{i}", 0, None, None, (10 + 5 * i, 30)) for i in range(6)]
        assigned, _ = assign_numbers(pins, [_record(0, "U1")])
        self.assertEqual(sorted(assigned.values(), key=int), [str(i) for i in range(1, 7)])

    def test_existing_numbers_are_preserved_for_non_passive_types(self):
        pins = [_pin("A", 0, "4", "4", (10, 20)), _pin("B", 0, None, None, (10, 60))]
        assigned, audit = assign_numbers(pins, [_record(0, "U1", kind="ic")], geometric_types=())
        self.assertEqual(assigned["A"], "4")
        self.assertEqual(assigned["B"], "1")
        self.assertEqual(audit["pins_kept_existing"], 1)

    def test_two_pin_passive_prefers_the_geometric_number(self):
        """Measured: geometric order is 74.5% correct on digit-named two-pin parts while
        OCR numbers were 56.7% correct overall, so the derived number wins here."""
        pins = [_pin("A", 0, "4", "4", (10, 20)), _pin("B", 0, "9", "9", (10, 60))]
        assigned, audit = assign_numbers(pins, [_record(0, "R1", kind="resistor")],
                                         geometric_types={"resistor", "capacitor", "inductor"})
        self.assertEqual(sorted(assigned.values()), ["1", "2"])
        self.assertEqual(audit["pins_geometric_override"], 2)

    def test_three_pin_passive_keeps_observed_numbers(self):
        pins = [_pin("A", 0, "7", "7", (10, 20)), _pin("B", 0, "8", "8", (10, 60)),
                _pin("C", 0, "9", "9", (10, 90))]
        assigned, _ = assign_numbers(pins, [_record(0, "R1", kind="resistor")],
                                     geometric_types={"resistor"})
        self.assertEqual(assigned, {"A": "7", "B": "8", "C": "9"})


if __name__ == "__main__":
    unittest.main()
