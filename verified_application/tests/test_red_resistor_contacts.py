import unittest
import cv2
import numpy as np

from pcb_parser.red_resistor_contacts import resistor_contact_proof


class RedResistorContactTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((128,160,3),255,np.uint8)
        cv2.rectangle(rgb,(50,50),(90,62),(190,0,0),2)
        cv2.line(rgb,(30,56),(50,56),(190,0,0),2)
        cv2.line(rgb,(90,56),(110,56),(190,0,0),2)
        cv2.line(rgb,(30,56),(30,25),(0,160,0),2)
        cv2.line(rgb,(110,56),(110,85),(0,160,0),2)
        record={'type':'resistor','designator':'R1','detection_confidence':.95,
                'bbox_top_left':[43,43,97,69]}
        return rgb,record

    def test_equal_external_stems_prove_two_contacts(self):
        rgb,record=self.fixture()
        proof=resistor_contact_proof(rgb,record)
        self.assertIsNotNone(proof)
        self.assertEqual(proof['axis'],0)
        self.assertAlmostEqual(proof['terminals'][0][0],30,delta=3)
        self.assertAlmostEqual(proof['terminals'][1][0],110,delta=3)
        self.assertLessEqual(abs(proof['stem_lengths'][0]-proof['stem_lengths'][1]),4)

    def test_one_missing_stem_or_wrong_source_type_is_withheld(self):
        rgb,record=self.fixture()
        rgb[45:90,95:120]=255
        self.assertIsNone(resistor_contact_proof(rgb,record))
        rgb,record=self.fixture();record['type']='inductor'
        self.assertIsNone(resistor_contact_proof(rgb,record))


if __name__=='__main__':unittest.main()
