import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from pcb_parser.unet_data import SegmentationTiles
from pcb_parser.unet_retraining import CachedTrainingTiles, augment_rgb


class RetrainingTests(unittest.TestCase):
    def test_augmentation_keeps_white_and_pixel_locations(self):
        rgb = np.full((9, 11, 3), 255, np.uint8)
        rgb[4, 2:8] = 98
        original = rgb.copy()
        cfg = {"identity_probability": 0., "gamma_range": [1.5, 1.5], "ink_strength_range": [1., 1.]}
        changed = augment_rgb(rgb, np.random.default_rng(7), cfg)
        self.assertTrue(np.all(changed[original == 255] == 255))
        self.assertTrue(np.all(changed[original == 98] == 61))
        self.assertTrue(np.array_equal(rgb, original))

    def test_deterministic_seed_and_invalid_range(self):
        rgb = np.arange(300, dtype=np.uint16).reshape(10, 10, 3).clip(0, 255).astype(np.uint8)
        cfg = {"identity_probability": .3, "gamma_range": [.55, 1.8], "ink_strength_range": [.75, 1.2]}
        a = augment_rgb(rgb, np.random.default_rng(9), cfg)
        b = augment_rgb(rgb, np.random.default_rng(9), cfg)
        self.assertTrue(np.array_equal(a, b))
        with self.assertRaises(ValueError):
            augment_rgb(rgb, np.random.default_rng(), cfg | {"gamma_range": [0, 1]})

    def test_cached_tiles_equal_original_and_augmented_targets_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            image = np.full((7, 19, 3), 255, np.uint8); image[3, 2:15] = 98
            target = np.zeros((7, 19), np.uint8); target[3, 2:15] = 255
            pin = np.zeros((7, 19), np.uint8); pin[3, 2] = 255
            Image.fromarray(image).save(p/"image.png")
            Image.fromarray(target).save(p/"wire.png")
            Image.fromarray(pin).save(p/"pin.png")
            row = {"image": str(p/"image.png"), "wire_mask": str(p/"wire.png"), "pin_heatmap": str(p/"pin.png"), "width": 19, "height": 7}
            base = SegmentationTiles([row], 16, 12)
            cached = CachedTrainingTiles([row], 16, 12)
            augmented = CachedTrainingTiles([row], 16, 12, {"identity_probability": 0., "gamma_range": [1.5, 1.5], "ink_strength_range": [1., 1.]})
            for i in range(len(base)):
                for a, b in zip(base[i], cached[i]):
                    self.assertTrue(torch.equal(a, b))
                original_image, truth, valid = base[i]
                new_image, new_truth, new_valid = augmented[i]
                self.assertTrue(torch.equal(truth, new_truth))
                self.assertTrue(torch.equal(valid, new_valid))
                self.assertTrue(torch.equal(new_image[:, valid == 0], original_image[:, valid == 0]))
                self.assertFalse(torch.equal(original_image, new_image))


if __name__ == "__main__":
    unittest.main()
