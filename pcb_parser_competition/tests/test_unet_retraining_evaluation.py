import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_unet_retraining_v2 import aggregate, targets_0036


class RetrainingEvaluationTests(unittest.TestCase):
    def test_target_paths_match_runtime_and_serialized_coordinates(self):
        runtime = {"paths": [{"points": ((601, 80), (691, 80)), "connected_within_reference_tube": False},
                             {"points": ((601, 181), (634, 181)), "connected_within_reference_tube": True}]}
        serialized = {"paths": [{"points": [list(point) for point in p["points"]],
                                "connected_within_reference_tube": p["connected_within_reference_tube"]} for p in runtime["paths"]]}
        self.assertEqual(targets_0036(runtime), {"X1": False, "DOUT_horizontal": True})
        self.assertEqual(targets_0036(runtime), targets_0036(serialized))

    def test_group_aggregation_uses_counts_not_average_percentages(self):
        records = []
        for group, tp, fp, fn in (("development", 10, 0, 10), ("other_validation", 90, 20, 0)):
            records.append({"variant": "contrast", "group": group, "wire": {"tp": tp, "fp": fp, "fn": fn},
                            "pin_coordinates_5px": {"tp": 1, "fp": 0, "fn": 0},
                            "connected_path_count": 3, "path_count": 5, "extra_pixels_more_than_2px_from_draft": fp})
        total = aggregate(records, "contrast", "all")
        self.assertEqual(total["wire_micro"]["tp"], 100)
        self.assertAlmostEqual(total["wire_micro"]["dice"], 200/230)
        self.assertEqual(aggregate(records, "contrast", "other_validation")["pages"], 1)


if __name__ == "__main__":
    unittest.main()
