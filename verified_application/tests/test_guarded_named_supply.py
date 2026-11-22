import unittest

import cv2
import numpy as np

from pcb_parser.structural_patterns import guarded_named_supply


class GuardedNamedSupplyTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((128,128,3),255,np.uint8)
        cv2.line(self.rgb,(48,47),(72,47),(190,0,0),2)
        cv2.line(self.rgb,(60,47),(60,60),(190,0,0),2)
        cv2.line(self.rgb,(60,60),(60,90),(0,160,0),2)
        self.box=[55,43,65,53]
        self.point=[60,60]
        self.words=[{'text':'VCC','confidence':.96,'view':'tile-a','bbox':[43,20,78,38]},
                    {'text':'vcc','confidence':.87,'view':'global','bbox':[43,20,78,38]}]

    def test_two_moderate_views_plus_observed_supply_geometry(self):
        result=guarded_named_supply(self.words,self.rgb,self.box,self.point)
        self.assertEqual(result['caption'],'VCC')
        self.assertEqual(result['views'],['global','tile-a'])
        self.assertTrue(result['outward_wire'])

    def test_rejects_missing_conductor_or_single_view(self):
        without_wire=self.rgb.copy()
        without_wire[58:95,55:66]=255
        self.assertIsNone(guarded_named_supply(self.words,without_wire,self.box,self.point))
        self.assertIsNone(guarded_named_supply(self.words[:1],self.rgb,self.box,self.point))

    def test_rejects_conflicting_caption(self):
        words=[*self.words,{'text':'VDD','confidence':.96,'view':'tile-b','bbox':[43,20,78,38]}]
        self.assertIsNone(guarded_named_supply(words,self.rgb,self.box,self.point))


if __name__=='__main__':unittest.main()
