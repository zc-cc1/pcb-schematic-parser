"""Protect the offline metric against overlaps, duplicates and class-ID mismatches."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_detector import counts_for_case, metric


def obj(box, name="resistor"):
    return {"bbox": box, "class_name": name}


class DetectionMatchingTests(unittest.TestCase):
    def test_adjacent_components_use_maximum_cardinality_matching(self):
        truth = [obj([0, 0, 10, 10]), obj([4, 0, 14, 10])]
        predicted = [obj([2, 0, 12, 10]), obj([0, 0, 10, 10])]
        counts = counts_for_case(predicted, truth, .5)["resistor"]
        self.assertEqual(counts, {"tp": 2, "fp": 0, "fn": 0})

    def test_duplicate_detection_is_false_positive(self):
        counts = counts_for_case([obj([0, 0, 10, 10])]*2, [obj([0, 0, 10, 10])], .5)["resistor"]
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 0})

    def test_same_box_wrong_class_does_not_match(self):
        counts = counts_for_case([obj([0, 0, 10, 10], "capacitor")], [obj([0, 0, 10, 10])], .5)
        self.assertEqual(counts["capacitor"], {"tp": 0, "fp": 1, "fn": 0})
        self.assertEqual(counts["resistor"], {"tp": 0, "fp": 0, "fn": 1})


if __name__ == "__main__":
    unittest.main()
