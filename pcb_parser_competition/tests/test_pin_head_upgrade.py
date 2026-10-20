"""Tests for the pin-head upgrade: configurable loss weight, disk targets, split decoder."""
import unittest

import numpy as np
import torch

from pcb_parser.unet_model import WirePinUNet, segmentation_loss
from pcb_parser.unet_split_model import SplitWirePinUNet, load_shared_decoder_weights


class PinLossWeightTests(unittest.TestCase):
    def _inputs(self):
        logits = torch.zeros((1, 2, 16, 16))
        targets = torch.zeros((1, 2, 16, 16))
        targets[0, 0, 4:6, 4:6] = 1.0
        targets[0, 1, 10, 10] = 1.0
        valid = torch.ones((1, 2, 16, 16))
        return logits, targets, valid

    def test_default_matches_legacy_half_weight(self):
        logits, targets, valid = self._inputs()
        legacy, _ = segmentation_loss(logits, targets, valid, 8.0, 20.0)
        explicit, _ = segmentation_loss(logits, targets, valid, 8.0, 20.0, pin_loss_weight=0.5)
        self.assertAlmostEqual(float(legacy), float(explicit), places=6)

    def test_stronger_pin_weight_increases_total_loss(self):
        logits, targets, valid = self._inputs()
        weak, weak_parts = segmentation_loss(logits, targets, valid, 8.0, 20.0, pin_loss_weight=0.5)
        strong, strong_parts = segmentation_loss(logits, targets, valid, 8.0, 20.0, pin_loss_weight=1.0)
        self.assertGreater(float(strong), float(weak))
        # the wire terms must be untouched by the pin weight
        self.assertAlmostEqual(float(weak_parts["wire_bce"]), float(strong_parts["wire_bce"]), places=6)
        self.assertAlmostEqual(float(weak_parts["pin_bce"]), float(strong_parts["pin_bce"]), places=6)


class SplitModelTests(unittest.TestCase):
    def test_weight_transfer_copies_encoder_and_splits_output_head(self):
        source = WirePinUNet(8)
        target = SplitWirePinUNet(8)
        state = {k: v.clone() for k, v in source.state_dict().items()}
        # make the two output channels distinguishable
        state["out.conv.bias"] = torch.tensor([0.25, -0.75])
        info = load_shared_decoder_weights(target, state)
        self.assertEqual(info["missing"], [])
        self.assertEqual(info["unexpected"], [])
        self.assertTrue(torch.allclose(target.wire_out.conv.bias, torch.tensor([0.25])))
        self.assertTrue(torch.allclose(target.pin_out.conv.bias, torch.tensor([-0.75])))
        # encoder must be copied bit-exact
        for key, value in source.state_dict().items():
            if key.startswith(("inc", "down")):
                self.assertTrue(torch.allclose(target.state_dict()[key], value))

    def test_heads_are_independent(self):
        model = SplitWirePinUNet(8).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (1, 2, 64, 64))
        self.assertFalse(torch.allclose(out[:, 0], out[:, 1]))

    def test_split_model_has_more_capacity_than_shared(self):
        shared = sum(p.numel() for p in WirePinUNet(8).parameters())
        split = sum(p.numel() for p in SplitWirePinUNet(8).parameters())
        self.assertGreater(split, shared)


class DiskTargetTests(unittest.TestCase):
    def test_disk_has_flat_plateau_and_cutoff(self):
        from prepare_pin_disk_targets import render
        array = render([(20.0, 20.0)], 40, 40, "disk").astype(np.float32) / 255
        self.assertAlmostEqual(float(array[20, 20]), 1.0, places=2)
        self.assertAlmostEqual(float(array[20, 18]), 1.0, places=2)   # inside inner radius 3
        self.assertAlmostEqual(float(array[20, 15]), 0.50, places=2)  # 0.5 exactly at radius 5
        self.assertAlmostEqual(float(array[20, 13]), 0.0, places=2)   # beyond outer radius

    def test_softdisk_ramps_to_zero(self):
        from prepare_pin_disk_targets import render
        array = render([(20.0, 20.0)], 40, 40, "softdisk").astype(np.float32) / 255
        self.assertAlmostEqual(float(array[20, 20]), 1.0, places=2)
        # radius 5 sits on the ramp: (6-5)/(6-3) = 1/3
        self.assertAlmostEqual(float(array[20, 15]), 1 / 3, places=2)
        self.assertAlmostEqual(float(array[20, 14]), 0.0, places=2)   # radius 6 = zero

    def test_two_close_tips_do_not_sum_above_one(self):
        from prepare_pin_disk_targets import render
        array = render([(20.0, 20.0), (22.0, 20.0)], 40, 40, "disk").astype(np.float32) / 255
        self.assertLessEqual(float(array.max()), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
