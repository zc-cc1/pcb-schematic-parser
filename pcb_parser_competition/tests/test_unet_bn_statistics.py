import unittest
import torch
from torch import nn
from pcb_parser.unet_bn_statistics import fixed_batches,reestimate_bn_buffers


class BNStatisticsTests(unittest.TestCase):
    def model(self):
        model=nn.Sequential(nn.Conv2d(1,1,1,bias=False),nn.BatchNorm2d(1),nn.Dropout2d(.9))
        with torch.no_grad():model[0].weight.fill_(1)
        model.train();model[2].eval()
        return model

    def test_reverse_preserves_batch_membership_and_singleton(self):
        forward=fixed_batches(5,2)
        self.assertEqual(forward,[[0,1],[2,3],[4]])
        self.assertEqual(fixed_batches(5,2,True),[[4],[2,3],[0,1]])
        with self.assertRaises(ValueError):fixed_batches(0,2)

    def test_cumulative_matches_independent_mean_and_restores_modes(self):
        batches=[torch.full((2,1,4,4),v) for v in (-8.,-2.,6.,18.)]
        results=[]
        for values in (batches,list(reversed(batches))):
            model=self.model();weight=model[0].weight.detach().clone()
            count=reestimate_bn_buffers(model,values,None)
            self.assertAlmostEqual(model[1].running_mean.item(),3.5,places=6)
            self.assertTrue(torch.equal(weight,model[0].weight))
            self.assertIsNone(model[0].weight.grad)
            self.assertEqual([m.training for m in model.modules()],[True,True,True,False])
            self.assertEqual(model[1].momentum,.1)
            self.assertEqual((count['batches'],count['image_crops']),(4,8))
            results.append(model[1].running_mean.item())
        self.assertAlmostEqual(*results,places=6)
        model=self.model();reestimate_bn_buffers(model,batches,.1)
        expected=0.
        for value in (-8.,-2.,6.,18.):expected=.9*expected+.1*value
        self.assertAlmostEqual(model[1].running_mean.item(),expected,places=6)
        reverse=self.model();reestimate_bn_buffers(reverse,list(reversed(batches)),.1)
        self.assertGreater(abs(model[1].running_mean.item()-reverse[1].running_mean.item()),.5)

    def test_failure_restores_statistics_and_rejects_targets(self):
        model=self.model();before={k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(ValueError):reestimate_bn_buffers(model,[],None)
        with self.assertRaises(ValueError):reestimate_bn_buffers(model,[torch.zeros(2,1,4,4),(torch.ones(2,1,4,4),torch.zeros(2))],None)
        for k,v in before.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        self.assertEqual(model[1].momentum,.1);self.assertTrue(model.training)
        with self.assertRaises(ValueError):reestimate_bn_buffers(nn.Conv2d(1,1,1),[torch.ones(2,1,4,4)],None)
        with self.assertRaises(ValueError):reestimate_bn_buffers(model,[torch.ones(2,1,4,4)],0.)


if __name__=='__main__':unittest.main()
