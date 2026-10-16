import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from pcb_parser.unet_data import annotation_targets,SegmentationTiles,tile_starts,predict_tiled
from pcb_parser.unet_model import WirePinUNet,segmentation_loss


class UNetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.source={"canvas":{"width":32,"height":24},
                     "edges":{"physical":{"category":"wire","isFlywire":False,"keyPointPaths":{"wire":[[[2,4],[15,4]]]}},
                              "logical":{"isFlywire":True,"keyPointPaths":{"wire":[[[2,18],[15,18]]]}},
                              "deleted":{"state":"deleted","keyPointPaths":{"wire":[[[2,21],[15,21]]]}}},
                     "components":{"C_1":{"pins":{"P_1":{"tip":[7,9]}}}},"pins":{}}

    def test_physical_paths_exclude_flywires_and_deleted(self):
        wire,_,_,stats=annotation_targets(self.source,32,24,line_width=1)
        mask=np.asarray(wire)
        self.assertEqual(mask[4,8],255)
        self.assertEqual(mask[18,8],0)
        self.assertEqual(mask[21,8],0)
        self.assertEqual(stats["flywires_excluded"],1)

    def test_missing_path_never_draws_connection_from_endpoints(self):
        self.source["edges"]["physical"]={"connect":{"a":{"location":[2,4]},"b":{"location":[15,4]}}}
        wire,_,_,stats=annotation_targets(self.source,32,24)
        self.assertEqual(np.asarray(wire).sum(),0)
        self.assertEqual(stats["missing_physical_paths"],1)

    def test_tip_heatmap_is_original_tip_not_base(self):
        self.source["components"]["C_1"]["pins"]["P_1"]["base"]=[16,18]
        _,heat,tips,_=annotation_targets(self.source,32,24,pin_sigma=1)
        self.assertEqual(np.asarray(heat)[9,7],255)
        self.assertEqual(np.asarray(heat)[18,16],0)
        self.assertEqual(tips[0]["point"],[7.,9.])

    def test_canvas_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            annotation_targets(self.source,31,24)

    def test_padding_has_no_supervised_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            Image.new("RGB",(5,3),(20,40,60)).save(folder/"rgb.png")
            Image.new("L",(5,3),255).save(folder/"target.png")
            row={"image":str(folder/"rgb.png"),"wire_mask":str(folder/"target.png"),
                 "pin_heatmap":str(folder/"target.png"),"width":5,"height":3}
            image,target,valid=SegmentationTiles([row],16,12)[0]
            self.assertEqual(int(valid.sum()),15)
            self.assertEqual(float(target[:,3:].sum()),0.)
            self.assertAlmostEqual(float(image[0,0,0]),20/255,places=6)

    def test_padded_loss_does_not_affect_gradients(self):
        logits=torch.zeros((1,2,8,8),requires_grad=True)
        target=torch.zeros_like(logits)
        valid=torch.zeros((1,8,8))
        valid[:,:3,:4]=1
        loss,_=segmentation_loss(logits,target,valid)
        loss.backward()
        self.assertEqual(float(logits.grad[:,:,3:].abs().sum()),0.)
        self.assertGreater(float(logits.grad[:,:,:3,:4].abs().sum()),0.)

    def test_model_restores_odd_shape_and_backpropagates(self):
        model=WirePinUNet(base_channels=4)
        output=model(torch.rand(2,3,35,49))
        self.assertEqual(tuple(output.shape),(2,2,35,49))
        output.square().mean().backward()
        self.assertGreater(float(model.out.conv.weight.grad.abs().sum()),0.)

    def test_tiled_inference_covers_edges_without_labels_or_resize(self):
        class Constant(torch.nn.Module):
            def forward(self,image):
                return torch.zeros((image.shape[0],2,*image.shape[-2:]),device=image.device)
        output=predict_tiled(Constant(),np.zeros((19,37,3),np.uint8),tile_size=16,stride=12)
        self.assertEqual(output.shape,(2,19,37))
        self.assertTrue(np.allclose(output,.5))
        self.assertEqual(tile_starts(37,16,12)[-1],21)


if __name__=="__main__":
    unittest.main()
