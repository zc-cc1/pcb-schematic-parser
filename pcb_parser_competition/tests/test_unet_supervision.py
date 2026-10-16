import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from pcb_parser.unet_data import annotation_targets, SegmentationTiles
from pcb_parser.unet_retraining import CachedTrainingTiles
from pcb_parser.unet_model import segmentation_loss
from pcb_parser.unet_supervision import load_valid_mask


class SupervisionReviewTests(unittest.TestCase):
    def source(self):
        return {"canvas":{"width":16,"height":16},"edges":{
            "wire":{"category":"wire","keyPointPaths":{"wire":[[[1,2],[12,2]]]}},
            "outline":{"category":"wire","keyPointPaths":{"wire":[[[1,9],[12,9]]]},
                "wire_supervision":{"decision":"exclude","classification":"symbol_outline","reason":"Verified component body outline"}},
            "unknown":{"category":"wire","keyPointPaths":{"wire":[]},"connect":{"P1":{"location":[3,11]}},
                "wire_supervision":{"decision":"ignore","classification":"unresolved_association","reason":"No visible route; source association pending"}}},
            "components":{"C1":{"pins":{"P1":{"tip":[3,11]}}}},"pins":{}}

    def test_excludes_outline_and_retains_explicit_unresolved_counters(self):
        source=self.source();old=copy.deepcopy(source)
        wire,pin,tips,stats=annotation_targets(source,16,16,1,1)
        self.assertEqual(np.asarray(wire)[2,7],255)
        self.assertEqual(np.asarray(wire)[9,7],0)
        self.assertEqual(stats["missing_physical_paths"],0)
        self.assertEqual(stats["excluded_missing_path_records"],1)
        self.assertEqual(stats["ignored_source_associations"],1)
        self.assertEqual(stats["excluded_nonwire_records"],1)
        self.assertEqual(stats["raw_physical_edge_records"],3)
        self.assertEqual(np.asarray(pin)[11,3],255)
        self.assertEqual(source,old)

    def test_unreviewed_missing_path_is_still_reported(self):
        source=self.source();del source["edges"]["unknown"]["wire_supervision"]
        _,_,_,stats=annotation_targets(source,16,16)
        self.assertEqual(stats["missing_physical_paths"],1)

    def test_rejects_unexplained_or_unknown_exclusion(self):
        for change in ({"reason":""},{"classification":"whatever"},{"decision":"ignore"}):
            source=self.source();source["edges"]["outline"]["wire_supervision"].update(change)
            with self.assertRaises(ValueError):
                annotation_targets(source,16,16)

    def test_both_training_loaders_mask_unknown_pixels_and_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            Image.new("RGB",(6,5),(100,100,100)).save(folder/"image.png")
            Image.new("L",(6,5),255).save(folder/"target.png")
            mask=np.full((5,6),255,np.uint8);mask[1:3,2:4]=0
            Image.fromarray(mask).save(folder/"valid.png")
            row={"image":str(folder/"image.png"),"wire_mask":str(folder/"target.png"),
                 "pin_heatmap":str(folder/"target.png"),"valid_mask":str(folder/"valid.png"),"width":6,"height":5}
            results=[]
            for loader in (SegmentationTiles([row],8,6),CachedTrainingTiles([row],8,6)):
                image,target,valid=loader[0]
                self.assertEqual(float(valid.sum()),26)
                self.assertEqual(float(valid[1,2]),0)
                self.assertEqual(float(valid[6,6]),0)
                logits=torch.zeros((1,2,8,8),requires_grad=True)
                loss,_=segmentation_loss(logits,target.unsqueeze(0),valid.unsqueeze(0))
                loss.backward()
                self.assertEqual(float(logits.grad[:,:,1:3,2:4].abs().sum()),0)
                self.assertGreater(float(logits.grad[:,:,0,0].abs().sum()),0)
                results.append(valid)
            self.assertTrue(torch.equal(*results))

    def test_mask_size_and_values_must_be_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"valid.png"
            Image.new("L",(3,3),255).save(path)
            row={"width":4,"height":4,"valid_mask":str(path)}
            with self.assertRaisesRegex(ValueError,"dimensions differ"):
                load_valid_mask(row)
            Image.new("L",(4,4),128).save(path)
            with self.assertRaisesRegex(ValueError,"only 0 and 255"):
                load_valid_mask(row)

    def test_missing_valid_mask_keeps_legacy_behavior(self):
        self.assertIsNone(load_valid_mask({"width":2,"height":2}))


if __name__=="__main__":
    unittest.main()
