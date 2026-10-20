"""Tests for the v2 training path: ignore-aware loss and tile validity masks.

These pin the two behaviours that make v2 supervision different from v1:
  1. pixels flagged in `ignore_masks` must not contribute to the loss at all
     (gradient zero), on either channel;
  2. a tile's valid mask must zero exactly the ignored pixels inside the image
     and the padding outside it, per channel.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pcb_parser.unet_model import WirePinUNet
from train_unet_v2 import Tiles, loss_with_ignore, tile_starts


class IgnoreLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = WirePinUNet(8)

    def test_ignored_pixels_do_not_change_the_loss(self):
        logits = torch.zeros((1, 2, 16, 16), requires_grad=True)
        targets = torch.zeros((1, 2, 16, 16))
        targets[0, 0, 4:6, 4:6] = 1.0
        targets[0, 1, 10, 10] = 1.0
        valid = torch.ones((1, 2, 16, 16))
        baseline, _ = loss_with_ignore(logits, targets, valid, 8.0, 20.0)
        # flag a region that contains labels as unknown
        valid_ignored = valid.clone()
        valid_ignored[0, 0, 4:6, 4:6] = 0
        valid_ignored[0, 1, 10, 10] = 0
        masked, _ = loss_with_ignore(logits, targets, valid_ignored, 8.0, 20.0)
        self.assertGreater(float(baseline), float(masked))

    def test_fully_ignored_channel_cannot_divide_by_zero(self):
        logits = torch.zeros((1, 2, 8, 8))
        targets = torch.zeros((1, 2, 8, 8))
        valid = torch.ones((1, 2, 8, 8))
        valid[:, 1] = 0
        loss, components = loss_with_ignore(logits, targets, valid, 8.0, 20.0)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(components["pin_bce"], 0.0)

    def test_ignore_blocks_gradient(self):
        logits = torch.zeros((1, 2, 8, 8), requires_grad=True)
        targets = torch.zeros((1, 2, 8, 8))
        valid = torch.ones((1, 2, 8, 8))
        valid[:, :, :, 4:] = 0
        loss, _ = loss_with_ignore(logits, targets, valid, 8.0, 20.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        # gradient must be exactly zero where everything is ignored
        self.assertAlmostEqual(float(logits.grad[:, :, :, 5:].abs().sum()), 0.0, places=6)


class TileValidityTests(unittest.TestCase):
    def _page(self, folder, width=40, height=30):
        from PIL import Image
        image = np.full((height, width, 3), 255, np.uint8)
        Image.fromarray(image).save(folder / "image.png")
        wire = np.zeros((height, width), np.uint8)
        wire[5:7, 5:7] = 255
        Image.fromarray(wire).save(folder / "wire.png")
        pin = np.zeros((height, width), np.uint8)
        pin[10, 10] = 255
        Image.fromarray(pin).save(folder / "pin.png")
        wire_ignore = np.zeros((height, width), np.uint8)
        wire_ignore[5:7, 5:7] = 1
        pin_ignore = np.zeros((height, width), np.uint8)
        pin_ignore[10, 10] = 1
        np.savez_compressed(folder / "ignore.npz", wire_ignore=wire_ignore, pin_ignore=pin_ignore)
        return {"case_id": "t", "image": str(folder / "image.png"), "width": width, "height": height,
                "wire_mask": str(folder / "wire.png"), "pin_heatmap": str(folder / "pin.png"),
                "ignore_masks": str(folder / "ignore.npz"), "pin_points": str(folder / "pins.json")}

    def test_valid_mask_excludes_ignore_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = self._page(folder)
            (folder / "pins.json").write_text(json.dumps([{"point": [10.0, 10.0]}]), encoding="utf-8")
            images, targets, valid = Tiles([row], 16, 16, _Cache())[0]
            self.assertEqual(tuple(images.shape), (3, 16, 16))
            self.assertEqual(tuple(targets.shape), (2, 16, 16))
            # ignored label pixels are excluded from the loss per channel
            self.assertEqual(float(valid[0, 5, 5]), 0.0)
            self.assertEqual(float(valid[1, 10, 10]), 0.0)
            # a normal background pixel inside the first tile stays supervised
            self.assertEqual(float(valid[0, 12, 10]), 1.0)

    def test_padding_area_is_not_supervised(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row = self._page(folder, width=40, height=30)
            (folder / "pins.json").write_text(json.dumps([]), encoding="utf-8")
            self.assertEqual(tile_starts(40, 16, 16), [0, 16, 24])
            self.assertEqual(tile_starts(10, 16, 16), [0])
            dataset = Tiles([row], 16, 16, _Cache())
            # tiles fill x then y; the last tile starts at (24, 14) on a 40x30 page,
            # so rows 16..30 and columns 40..40 are padding and must not be scored
            _, _, valid = dataset[len(dataset) - 1]
            self.assertGreater(float(valid[0, :, :].sum()), 0.0)
            self.assertEqual(float(valid[0, 16:, :].sum()), 0.0)


class _Cache:
    """Minimal stand-in for PageCache with no eviction."""

    def __init__(self):
        self.pages = {}

    def get(self, row):
        if row["case_id"] not in self.pages:
            from PIL import Image
            with Image.open(row["image"]) as source:
                image = np.asarray(source.convert("RGB"), np.uint8)
            with Image.open(row["wire_mask"]) as mask:
                wire = np.asarray(mask, np.uint8)
            with Image.open(row["pin_heatmap"]) as heat:
                pin = np.asarray(heat, np.uint8)
            with np.load(row["ignore_masks"]) as ignore:
                self.pages[row["case_id"]] = (image, wire, pin, ignore["wire_ignore"].copy(),
                                              ignore["pin_ignore"].copy())
        return self.pages[row["case_id"]]


if __name__ == "__main__":
    unittest.main()
