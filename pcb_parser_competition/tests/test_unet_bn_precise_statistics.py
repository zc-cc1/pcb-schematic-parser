import unittest
import torch
from torch import nn
from pcb_parser.unet_bn_precise_statistics import reestimate_bn_buffers_precise


class PreciseBNTests(unittest.TestCase):
    def test_cancellation_oracle_order_and_parameter_preservation(self):
        batches=[torch.full((2,1,4,4),v) for v in (100000000.,-100000000.,1.,1.)]
        states=[]
        for values in (batches,list(reversed(batches))):
            model=nn.Sequential(nn.Conv2d(1,1,1,bias=False),nn.BatchNorm2d(1),nn.Dropout2d(.5))
            with torch.no_grad():model[0].weight.fill_(1.)
            model.eval();parameter=model[0].weight.detach().clone()
            record=reestimate_bn_buffers_precise(model,values)
            self.assertEqual(model[1].running_mean.item(),.5)
            self.assertEqual(model[1].momentum,.1)
            self.assertFalse(model.training);self.assertFalse(model[1].training)
            self.assertTrue(torch.equal(parameter,model[0].weight));self.assertIsNone(model[0].weight.grad)
            self.assertEqual(record['accumulator_dtype'],'float64');self.assertFalse(record['population_variance'])
            states.append(model[1].running_mean.clone())
        self.assertTrue(torch.equal(*states))

    def test_failed_input_keeps_existing_buffers(self):
        model=nn.Sequential(nn.BatchNorm2d(1));model.train()
        before={k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(ValueError):reestimate_bn_buffers_precise(model,[torch.ones(2,1,4,4),None])
        for k,v in before.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        self.assertTrue(model.training)


if __name__=='__main__':unittest.main()
