"""OCR rereading must preserve image coordinates and explicitly restore detection."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pcb_parser.component_text import run_ocr
from pcb_parser.text_recovery import recover_texts


class StubEngine:
    def __init__(self):self.calls=[]
    def __call__(self,image,**kwargs):
        self.calls.append(kwargs)
        if kwargs.get("use_det"):
            return SimpleNamespace(boxes=None,txts=[],scores=[])
        return SimpleNamespace(txts=["U2"],scores=[.9])


class RecoveryTests(unittest.TestCase):
    def test_recovery_coordinates_remain_original_and_detection_is_restored(self):
        rgb=np.full((80,120,3),255,dtype=np.uint8)
        texts=[{"text_id":"text_1","text":"u2","confidence":.8,"view":"global",
                "bbox":[40.,25.,60.,40.],"polygon":[[40.,25.],[60.,25.],[60.,40.],[40.,40.]]}]
        engine=StubEngine()
        recovered,debug=recover_texts(rgb,engine,texts,{"max_rois":1})
        observed=[o for r in recovered for o in r["observations"] if o["text"]=="U2"]
        self.assertTrue(observed)
        self.assertEqual(observed[0]["bbox"],[40.,25.,60.,40.])
        self.assertEqual(debug["roi_count"],1)
        run_ocr(rgb,engine,{"tile_size":128,"rotations":[0]})
        self.assertTrue(engine.calls[-1]["use_det"])
        self.assertTrue(engine.calls[-1]["use_rec"])


if __name__=="__main__":unittest.main()
