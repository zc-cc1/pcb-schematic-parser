import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_balanced_detection_dataset import complete_window, project_objects, assert_split_isolation


class DetectionAugmentationTests(unittest.TestCase):
    def test_chained_border_symbols_are_complete_and_coordinates_follow_translation(self):
        objects = [{"component_id": "A", "bbox": [25, 25, 45, 45]},
                   {"component_id": "B", "bbox": [40, 30, 70, 55]},
                   {"component_id": "C", "bbox": [69, 40, 95, 60]}]
        window = complete_window([20, 20, 50, 50], objects, 120, 100, 100, 2)
        projected = project_objects(objects, window)
        self.assertEqual(len(projected), 3)
        self.assertEqual(projected[0]["bbox"], [5, 5, 25, 25])
        self.assertGreaterEqual(window[2], 95)

    def test_cut_symbols_rejected_instead_of_silently_becoming_background(self):
        with self.assertRaises(ValueError):
            project_objects([{"bbox": [40, 40, 70, 70]}], [0, 0, 50, 50])

    def test_crop_cannot_cross_validation_source_group(self):
        rows = [{"case_id": "train", "group": "g1", "split": "train", "source_image_sha256": "a"},
                {"case_id": "val", "group": "g2", "split": "val", "source_image_sha256": "b"},
                {"case_id": "crop", "parent_case_id": "val", "sample_kind": "context_crop",
                 "group": "g2", "split": "train", "source_image_sha256": "b"}]
        with self.assertRaises(ValueError):
            assert_split_isolation(rows)


if __name__ == "__main__":
    unittest.main()
