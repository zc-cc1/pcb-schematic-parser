import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from pcb_parser.unet_negative_sampling import paired_schedule, NegativeFocusTrainingTiles
from pcb_parser.unet_retraining import CachedTrainingTiles


class NegativeSamplingTests(unittest.TestCase):
    def test_paired_quota_and_untouched_slots(self):
        base=[(i,0,0) for i in range(100)];focus=[(2,5,6)]
        a,b,selected=paired_schedule(base,focus,.25,17,1)
        self.assertEqual(len(selected),25);self.assertEqual(sorted(a),base)
        self.assertEqual((a,b,selected),paired_schedule(base,focus,.25,17,1))
        self.assertTrue(all(a[i]==b[i] for i in range(100) if i not in selected))
        self.assertNotEqual(a,paired_schedule(base,focus,.25,17,2)[0])

    def fixture(self,folder):
        rgb=np.full((20,24,3),255,np.uint8);rgb[2,2:18]=0;rgb[12,3:16]=100
        wire=np.zeros((20,24),np.uint8);wire[12,3:16]=255
        pin=np.zeros((20,24),np.uint8);pin[12,3]=255
        for name,a in (("image",rgb),("wire",wire),("pin",pin)):Image.fromarray(a).save(folder/f"{name}.png")
        row=dict(case_id="toy",split="train",source_split="train",width=24,height=20,
            image=str(folder/"image.png"),wire_mask=str(folder/"wire.png"),pin_heatmap=str(folder/"pin.png"))
        region=dict(case_id="toy",split="train",box=[2,1,18,4],hard_negative=True)
        return row,region

    def test_focused_crop_preserves_positive_labels_and_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            row,region=self.fixture(Path(temp))
            focused=NegativeFocusTrainingTiles([row],[region],1.,tile_size=32,stride=24)
            original=CachedTrainingTiles([row],32,24)
            for a,b in zip(focused[0],original[0]):self.assertTrue(torch.equal(a,b))
            self.assertGreater(int(focused[0][1][0].sum()),0)
            self.assertEqual(float(focused[0][2].sum()),480)

    def test_rejects_calibration_and_wire_inside_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            row,region=self.fixture(Path(temp))
            with self.assertRaises(ValueError):NegativeFocusTrainingTiles([row|dict(split="calibration")],[region],.25)
            with self.assertRaises(ValueError):NegativeFocusTrainingTiles([row],[region|dict(box=[2,10,18,14])],.25)
            with self.assertRaises(ValueError):NegativeFocusTrainingTiles([row],[region|dict(case_id="validation")],.25)


if __name__=="__main__":unittest.main()
