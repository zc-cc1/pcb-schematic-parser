import unittest
import numpy as np
import torch

from pcb_parser.unet_inference import decode_pins, predict_at_scale, combine_probabilities


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_subpixel_gaussian_peak_recovers_fractional_coordinates(self):
        yy,xx=np.mgrid[:17,:21]
        heat=np.exp(-((xx-10.25)**2+(yy-8.35)**2)/8).astype(np.float32)
        found=decode_pins(heat,0.5,9,"quadratic",3)
        self.assertEqual(len(found),1)
        np.testing.assert_allclose(found[0]["point"],[10.25,8.35],atol=1e-4)

    def test_plateau_deduplication_and_separate_close_pins(self):
        heat=np.zeros((20,20),np.float32)
        heat[4:6,4:6]=0.9
        heat[4,11]=0.8
        found=decode_pins(heat,0.5,3,"none",3)
        self.assertEqual(len(found),2)
        self.assertEqual(found[0]["point"],[4.,4.])
        self.assertEqual(found[1]["point"],[11.,4.])

    def test_border_peaks_remain_in_bounds(self):
        heat=np.zeros((12,13),np.float32)
        heat[0,0]=0.9
        heat[-1,-1]=0.8
        found=decode_pins(heat,0.5,5,"quadratic",3)
        self.assertEqual([row["point"] for row in found],[[0.,0.],[12.,11.]])

    def test_scaling_returns_original_shape_without_targets(self):
        class Constant(torch.nn.Module):
            def forward(self,image):
                return torch.zeros((image.shape[0],2,*image.shape[-2:]),device=image.device)
        probability=predict_at_scale(Constant(),np.full((19,37,3),255,np.uint8),"cpu",16,12,1.5)
        self.assertEqual(probability.shape,(2,19,37))
        np.testing.assert_allclose(probability,0.5)

    def test_default_decoder_matches_existing_integer_peak_selection(self):
        rng=np.random.default_rng(42)
        heat=rng.random((19,27),dtype=np.float32)
        old=torch.from_numpy(heat)[None,None]
        mask=(old==torch.nn.functional.max_pool2d(old,9,stride=1,padding=4))&(old>=0.5)
        ys,xs=torch.where(mask[0,0])
        expected=sorted([[int(x),int(y)] for y,x in zip(ys,xs)],key=lambda p:-heat[p[1],p[0]])
        self.assertEqual([p["point"] for p in decode_pins(heat)],expected)

    def test_unknown_fusion_mode_rejected(self):
        with self.assertRaises(ValueError):
            combine_probabilities({},"auto_using_labels")


if __name__=="__main__":
    unittest.main()
