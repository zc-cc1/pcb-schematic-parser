from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tests'))
from test_submission_fields import inputs, CFG
from pcb_parser.submission_fields import complete_fields
from run_power_networks import export_fields
from run_pin_text import read


class CompletedFieldExportTests(unittest.TestCase):
    def test_already_completed_identities_export_once_and_match_raw_entry(self):
        p,g,r = inputs()
        completed,graph,records,_ = complete_fields(p,g,r,CFG)
        snapshot = deepcopy((completed,graph,records))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            raw = export_fields(p,g,r,base/'raw')
            ready = export_fields(completed,graph,records,base/'ready',fields_completed=True)
            self.assertEqual(raw,ready)
            self.assertEqual(read(base/'raw/partial_submission.json'),read(base/'ready/partial_submission.json'))
            self.assertEqual(ready['conservative']['exported_nets'],1)
        self.assertEqual((completed,graph,records),snapshot)

    def test_completed_mode_still_validates_owner_and_point_contract(self):
        p,g,r=inputs();p,g,r,_=complete_fields(p,g,r,CFG)
        p['pin_candidates'][0]['point']=[99,99]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                export_fields(p,g,r,Path(folder),fields_completed=True)
