"""The component library must only claim what the page actually evidences."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_parser.box_ic_library import (apply_library, apply_library_from_records,  # noqa: E402
                                       canonical_number, identify_models, load_library)


def _write(tmp, payload):
    path = Path(tmp) / "lib.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


LIB = {
    "DRV8824-Q1": {"pins": [
        {"pin_id": "e1", "name": "CP1", "number": "1", "pin_type": "IN"},
        {"pin_id": "e2", "name": "CP2", "number": "2", "pin_type": "IN"},
        {"pin_id": "e3", "name": "VCP", "number": "3", "pin_type": "IN"},
        {"pin_id": "e9", "name": "VCP", "number": "3", "pin_type": "Undefined"},
        {"pin_id": "e4", "name": "AOUT1", "number": "5", "pin_type": "OUT"},
        {"pin_id": "e5", "name": "ISENA", "number": "6_1", "pin_type": "IN"},
        {"pin_id": "e6", "name": "ISENB", "number": "9_1", "pin_type": "IN"},
    ]},
    "S-8252 Series": {"pins": [
        {"pin_id": "e1", "name": "VDD", "number": "23", "pin_type": "Power"},
        {"pin_id": "e2", "name": "VSS", "number": "24", "pin_type": "Power"},
        {"pin_id": "e3", "name": "CO", "number": "25", "pin_type": "OUT"},
    ]},
    "NC": {"pins": [
        {"pin_id": "e1", "name": "NC", "number": "1", "pin_type": "IN"},
        {"pin_id": "e2", "name": "NC", "number": "2", "pin_type": "IN"},
        {"pin_id": "e3", "name": "NC", "number": "3", "pin_type": "IN"},
    ]},
}


def _component(index, box, kind="ic"):
    return {"component_id": f"DET_{index}", "detection_index": index, "type": kind,
            "bbox_top_left": list(box)}


def _pin(candidate, owner, point, number=None, name=None, number_status="no_reliable_text",
         name_status="no_reliable_text"):
    return {"candidate_id": candidate, "point": list(point), "owner_detection_index": owner,
            "owner": f"DET_{owner}", "owner_status": "assigned", "number": number,
            "pinname": name, "number_status": number_status, "pinname_status": name_status}


TOKENS = [{"text": "DRV8824-Q1", "bbox": [480, 80, 619, 96]}]


class LoadLibraryTest(unittest.TestCase):
    def test_aggregation_prefers_frequent_upper_case_spelling(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = load_library([_write(tmp, LIB)])
        self.assertEqual(library["drv8824q1"]["pins"]["3"], "VCP")

    def test_split_leads_collapse_onto_the_package_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = load_library([_write(tmp, LIB)])
        self.assertIn("6", library["drv8824q1"]["pins"])
        self.assertIn("9", library["drv8824q1"]["pins"])

    def test_canonical_number_only_strips_numeric_suffixes(self):
        self.assertEqual(canonical_number("6_1"), "6")
        self.assertEqual(canonical_number("A1B12"), "A1B12")
        self.assertEqual(canonical_number("12"), "12")

    def test_empty_number_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = load_library([_write(tmp, {"X1": {"pins": [
                {"pin_id": "e1", "name": "A", "number": "", "pin_type": "IN"},
                {"pin_id": "e2", "name": "B", "number": "1", "pin_type": "IN"}]}})])
        self.assertEqual(library["x1"]["pins"], {"1": "B"})


class IdentifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.library = load_library([_write(self.tmp.name, LIB)])
        self.config = {"model_search_margin_px": 40, "min_pins": 3, "min_name_matches": 1,
                       "min_model_chars": 4, "require_digit_in_model": True}

    def tearDown(self):
        self.tmp.cleanup()

    def test_model_text_must_be_next_to_the_component(self):
        components = [_component(0, (400, 100, 700, 500))]
        pins = {0: [_pin("P1", 0, (413, 200), name="VCP", name_status="observed_OCR_text")]}
        far = [{"text": "DRV8824-Q1", "bbox": [480, 5, 619, 36]}]
        self.assertEqual(identify_models(components, pins, far, self.library, self.config), {})
        found = identify_models(components, pins, TOKENS, self.library, self.config)
        self.assertEqual(found[0]["model"], "DRV8824-Q1")

    def test_competing_models_are_split_by_pin_names(self):
        components = [_component(0, (400, 100, 700, 500))]
        pins = {0: [_pin("P1", 0, (413, 200), name="VCP", name_status="observed_OCR_text"),
                    _pin("P2", 0, (413, 260), name="AOUT1", name_status="observed_OCR_text")]}
        tokens = TOKENS + [{"text": "S-8252 Series", "bbox": [500, 150, 640, 170]}]
        self.assertEqual(identify_models(components, pins, tokens, self.library, self.config)[0]["model"],
                         "DRV8824-Q1")

    def test_a_bare_word_is_not_a_part_number(self):
        components = [_component(0, (400, 100, 700, 500))]
        pins = {0: [_pin("P1", 0, (413, 200), name="NC", name_status="observed_OCR_text")]}
        tokens = [{"text": "NC", "bbox": [480, 80, 619, 96]}]
        self.assertEqual(identify_models(components, pins, tokens, self.library, self.config), {})


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.library = load_library([_write(self.tmp.name, LIB)])
        self.config = {"model_search_margin_px": 40, "min_pins": 3, "min_name_matches": 1,
                       "min_model_chars": 4, "require_digit_in_model": True,
                       "evidence_statuses": ["observed_OCR_text"], "fill_names_from_number": True,
                       "fill_missing_numbers_from_names": True, "fill_remaining_numbers": False,
                       "min_consistency_pairs": 3, "min_consistency": 0.6}

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self, pins):
        return {"components": [_component(0, (400, 100, 700, 600), kind="ic")],
                "pin_candidates": pins}

    def test_missing_number_is_filled_from_the_printed_name(self):
        result = self._result([_pin("P1", 0, (413, 200), name="VCP", name_status="observed_OCR_text")])
        audit = apply_library(result, TOKENS, self.library, self.config)
        self.assertEqual(result["pin_candidates"][0]["number"], "3")
        self.assertEqual(result["pin_candidates"][0]["number_source"], "library_pin_name")
        self.assertEqual(audit["pins_numbered_from_name"], 1)

    def test_a_derived_number_is_replaced_by_the_printed_name_mapping(self):
        # reading order guessed 4; the page prints VCP, and the entry says VCP is pin 3
        result = self._result([_pin("P1", 0, (413, 200), number="4", name="VCP",
                                    number_status="no_reliable_text", name_status="observed_OCR_text")])
        audit = apply_library(result, TOKENS, self.library, self.config)
        self.assertEqual(result["pin_candidates"][0]["number"], "3")
        self.assertEqual(audit["pins_numbered_from_name"], 1)

    def test_an_observed_number_is_never_overwritten(self):
        # the entry puts VCP at 3, the page printed 7: the page wins, no false positive
        result = self._result([_pin("P1", 0, (413, 200), number="7", name="VCP",
                                    number_status="observed_OCR_text", name_status="observed_OCR_text")])
        apply_library(result, TOKENS, self.library, self.config)
        self.assertEqual(result["pin_candidates"][0]["number"], "7")

    def test_missing_name_is_filled_from_the_entry(self):
        result = self._result([_pin("P1", 0, (413, 200), number="5", number_status="observed_OCR_text")])
        audit = apply_library(result, TOKENS, self.library, self.config)
        self.assertEqual(result["pin_candidates"][0]["pinname"], "AOUT1")
        self.assertEqual(audit["pins_named_from_library"], 1)

    def test_inconsistent_entry_is_rejected(self):
        # three pins carry both fields and all three contradict the entry's own numbering
        pins = [_pin(f"P{i}", 0, (413, 200 + i), number=str(i + 1), name=name,
                     number_status="observed_OCR_text", name_status="observed_OCR_text")
                for i, name in enumerate(("AOUT1", "ISENA", "ISENB"))]
        result = self._result(pins)
        audit = apply_library(result, TOKENS, self.library, self.config)
        self.assertEqual(audit["entries_rejected_by_consistency"], 1)
        self.assertEqual(audit["changes"], [])

    def test_consistent_entry_is_accepted(self):
        pins = [_pin("P1", 0, (413, 200), number="1", name="CP1",
                     number_status="observed_OCR_text", name_status="observed_OCR_text"),
                _pin("P2", 0, (413, 260), number="2", name="CP2",
                     number_status="observed_OCR_text", name_status="observed_OCR_text"),
                _pin("P3", 0, (413, 320), number="3", name="VCP",
                     number_status="observed_OCR_text", name_status="observed_OCR_text")]
        audit = apply_library(self._result(pins), TOKENS, self.library, self.config)
        self.assertEqual(audit["entries_rejected_by_consistency"], 0)
        self.assertEqual(audit["components_identified"], 1)

    def test_leftover_numbers_stay_unassigned_by_default(self):
        result = self._result([_pin("P1", 0, (413, 200))])
        apply_library(result, TOKENS, self.library, self.config)
        self.assertIsNone(result["pin_candidates"][0]["number"])

    def test_unidentified_component_is_untouched(self):
        result = self._result([_pin("P1", 0, (413, 200), number="7",
                                    number_status="observed_OCR_text")])
        audit = apply_library(result, [], self.library, self.config)
        self.assertEqual(result["pin_candidates"][0]["number"], "7")
        self.assertEqual(audit["components_identified"], 0)


class RecordsTest(unittest.TestCase):
    """Identification from the component record's Name, then fuzzy and pattern fallbacks."""

    LIB = {
        "HS96L03W2C03": {"pins": [
            {"pin_id": "e1", "name": "GND", "number": "1", "pin_type": "Power"},
            {"pin_id": "e2", "name": "VCC", "number": "2", "pin_type": "Power"},
            {"pin_id": "e3", "name": "SCL", "number": "3", "pin_type": "IN"},
            {"pin_id": "e4", "name": "SDA", "number": "4", "pin_type": "BI"},
        ]},
        "OTHER9": {"pins": [
            {"pin_id": "e1", "name": "A", "number": "1", "pin_type": "IN"},
            {"pin_id": "e2", "name": "B", "number": "2", "pin_type": "IN"},
            {"pin_id": "e3", "name": "C", "number": "3", "pin_type": "IN"},
        ]},
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.library = load_library([_write(self.tmp.name, self.LIB)])
        self.config = {"min_pins": 3, "types": ["ic"], "evidence_statuses": ["observed_OCR_text"],
                       "fill_names_from_number": True, "fill_missing_numbers_from_names": True,
                       "min_consistency": 0.6, "min_consistency_pairs": 3,
                       "model_fuzzy_match": True, "name_set_pattern_match": True}

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self, pins):
        return {"components": [_component(0, (0, 0, 100, 100), kind="ic")], "pin_candidates": pins}

    def _record(self, name):
        return [{"detection_index": 0, "component_id": "U1", "type": "ic", "Name": name,
                 "designator": "U1", "bbox_top_left": [0, 0, 100, 100]}]

    def test_exact_model_name_fills_numbers_from_names(self):
        pins = [_pin("P1", 0, (0, 10), name="SCL", name_status="observed_OCR_text")]
        audit = apply_library_from_records(self._result(pins), self._record("HS96L03W2C03"),
                                           self.library, self.config)
        self.assertEqual(pins[0]["number"], "3")
        self.assertEqual(audit["models"][0]["identified_by"], "component_record_Name")

    def test_truncated_model_name_is_matched_by_prefix(self):
        pins = [_pin("P1", 0, (0, 10), name="SCL", name_status="observed_OCR_text")]
        audit = apply_library_from_records(self._result(pins), self._record("HS96L03W2C0"),
                                           self.library, self.config)
        self.assertEqual(pins[0]["number"], "3")
        self.assertEqual(audit["models"][0]["identified_by"], "fuzzy_model_spelling")

    def test_no_model_name_is_resolved_from_the_pin_name_set(self):
        pins = [_pin("P1", 0, (0, 10), number="3", name="SCL",
                     number_status="observed_OCR_text", name_status="observed_OCR_text"),
                _pin("P2", 0, (0, 20), number="4", name="SDA",
                     number_status="observed_OCR_text", name_status="observed_OCR_text"),
                _pin("P3", 0, (0, 30), name="GND", name_status="observed_OCR_text"),
                _pin("P4", 0, (0, 40), name="VCC", name_status="observed_OCR_text")]
        audit = apply_library_from_records(self._result(pins), self._record(None), self.library, self.config)
        numbers = {p["candidate_id"]: p["number"] for p in pins}
        self.assertEqual(numbers["P3"], "1")
        self.assertEqual(numbers["P4"], "2")
        self.assertEqual(audit["models"][0]["identified_by"], "pin_name_pattern")

    def test_pattern_is_refused_when_the_read_numbers_contradict_it(self):
        pins = [_pin("P1", 0, (0, 10), number="9", name="SCL",
                     number_status="observed_OCR_text", name_status="observed_OCR_text"),
                _pin("P2", 0, (0, 20), number="8", name="SDA",
                     number_status="observed_OCR_text", name_status="observed_OCR_text"),
                _pin("P3", 0, (0, 30), name="GND", name_status="observed_OCR_text")]
        apply_library_from_records(self._result(pins), self._record(None), self.library, self.config)
        self.assertIsNone(pins[2]["number"])

    def test_near_tip_name_fill_counts_as_evidence(self):
        pins = [_pin("P1", 0, (0, 10), name="SCL",
                     name_status="observed_OCR_v2_missing_name_fill")]
        audit = apply_library_from_records(self._result(pins), self._record("HS96L03W2C03"),
                                           self.library, self.config)
        self.assertEqual(pins[0]["number"], "3")
        self.assertEqual(audit["pins_numbered_from_name"], 1)


    def test_last_missing_number_is_filled_by_elimination(self):
        pins = [_pin("P1", 0, (0, 10), number="1", number_status="observed_OCR_text"),
                _pin("P2", 0, (0, 20), number="2", number_status="observed_OCR_text"),
                _pin("P3", 0, (0, 30), number="3", number_status="observed_OCR_text"),
                _pin("P4", 0, (0, 40))]
        audit = apply_library_from_records(self._result(pins), self._record("HS96L03W2C03"),
                                           self.library, self.config)
        self.assertEqual(pins[3]["number"], "4")
        self.assertEqual(pins[3]["pinname"], "SDA")
        self.assertEqual(audit["pins_numbered_by_elimination"], 1)

    def test_elimination_is_refused_when_a_foreign_number_appears(self):
        pins = [_pin("P1", 0, (0, 10), number="1", number_status="observed_OCR_text"),
                _pin("P2", 0, (0, 20), number="9", number_status="observed_OCR_text"),
                _pin("P3", 0, (0, 30))]
        apply_library_from_records(self._result(pins), self._record("HS96L03W2C03"),
                                   self.library, self.config)
        # 9 is not in the entry, so the pin set cannot be that part: no invention
        self.assertIsNone(pins[2]["number"])

    def test_elimination_is_refused_when_two_pins_lack_numbers(self):
        pins = [_pin("P1", 0, (0, 10), number="1", number_status="observed_OCR_text"),
                _pin("P2", 0, (0, 20), number="2", number_status="observed_OCR_text"),
                _pin("P3", 0, (0, 30)),
                _pin("P4", 0, (0, 40))]
        apply_library_from_records(self._result(pins), self._record("HS96L03W2C03"),
                                   self.library, self.config)
        self.assertEqual([p["number"] for p in pins[2:]], [None, None])


if __name__ == "__main__":
    unittest.main()
