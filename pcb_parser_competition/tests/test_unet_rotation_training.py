import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pcb_parser.unet_retraining import CachedTrainingTiles
from pcb_parser.unet_rotation_training import RotationTrainingTiles, rotate_supervised_tile


class RotationTrainingTests(unittest.TestCase):
    def test_coordinate_oracle_for_rgb_wire_tip_and_padding(self):
        image = torch.ones(3, 5, 5)
        targets = torch.zeros(2, 5, 5)
        valid = torch.zeros(5, 5); valid[:4, :5] = 1
        image[:, 2, 1] = .4; targets[0, 2, 1] = 1
        targets[1, 1, 4] = .8
        # Explicit coordinate maps, independent of torch.rot90 implementation.
        expected_wire = {1: (3, 2), 2: (2, 3), 3: (1, 2)}  # (row,col)
        expected_tip = {1: (0, 1), 2: (3, 0), 3: (4, 3)}
        for k in (1, 2, 3):
            rgb, truth, region = rotate_supervised_tile(image, targets, valid, k)
            y, x = expected_wire[k]; ty, tx = expected_tip[k]
            self.assertTrue(torch.equal(rgb[:, y, x], torch.full((3,), .4)))
            self.assertEqual(float(truth[0, y, x]), 1)
            self.assertAlmostEqual(float(truth[1, ty, tx]), .8)
            self.assertEqual(int(region.sum()), 20)
            self.assertTrue(torch.all(rgb[:, region == 0] == 1))

    def test_inverse_preserves_values_without_interpolation_or_input_mutation(self):
        image = torch.arange(75).reshape(3, 5, 5).float()/75
        targets = torch.arange(50).reshape(2, 5, 5).float()/50
        valid = torch.eye(5)
        original = tuple(v.clone() for v in (image, targets, valid))
        for k in range(4):
            changed = rotate_supervised_tile(image, targets, valid, k)
            restored = rotate_supervised_tile(*changed, (4-k)%4)
            for before, after in zip(original, restored):
                self.assertTrue(torch.equal(before, after))
        for a, b in zip(original, (image, targets, valid)):
            self.assertTrue(torch.equal(a, b))

    def test_invalid_turn_and_mismatched_label_geometry(self):
        image = torch.ones(3, 5, 5); truth = torch.zeros(2, 5, 5); valid = torch.ones(5, 5)
        for k in (-1, 4, 1.5, True):
            with self.assertRaises(ValueError):
                rotate_supervised_tile(image, truth, valid, k)
        with self.assertRaises(ValueError):
            rotate_supervised_tile(image, torch.zeros(2, 4, 5), valid, 1)

    def test_cached_identity_equivalence_and_repeatable_rotation_sequence(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = np.full((7, 19, 3), 255, np.uint8); image[3, 2:15] = 98
            wire = np.zeros((7, 19), np.uint8); wire[3, 2:15] = 255
            pin = np.zeros((7, 19), np.uint8); pin[3, 2] = 255
            for key, values in (("image", image), ("wire", wire), ("pin", pin)):
                Image.fromarray(values).save(folder/f"{key}.png")
            row = dict(image=str(folder/"image.png"), wire_mask=str(folder/"wire.png"),
                       pin_heatmap=str(folder/"pin.png"), width=19, height=7)
            base = CachedTrainingTiles([row], 16, 12)
            identity = RotationTrainingTiles([row], 16, 12, {"quarter_turns": [0], "probabilities": [1.]})
            a = RotationTrainingTiles([row], 16, 12, seed=17)
            b = RotationTrainingTiles([row], 16, 12, seed=17)
            for index in range(12):
                tile = index%len(base)
                for expected, actual in zip(base[tile], identity[tile]):
                    self.assertTrue(torch.equal(expected, actual))
                for left, right in zip(a[tile], b[tile]):
                    self.assertTrue(torch.equal(left, right))
            self.assertTrue(np.array_equal(a.rotation_counts, b.rotation_counts))
            for dataset in (base, identity, a, b):
                dataset.clear_cache()

    def test_invalid_rotation_distribution(self):
        for config in ({"quarter_turns": [0, 1], "probabilities": [.2, .2]},
                       {"quarter_turns": [0, 0], "probabilities": [.5, .5]},
                       {"quarter_turns": [0, 4], "probabilities": [.5, .5]}):
            with self.assertRaises(ValueError):
                RotationTrainingTiles([], rotation=config)


if __name__ == "__main__":
    unittest.main()
