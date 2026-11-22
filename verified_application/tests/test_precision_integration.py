"""Portable entrypoint contract: opt-in behavior and no unrelated edits."""
import copy,unittest
import numpy as np
import agent

class PrecisionIntegrationTests(unittest.TestCase):
    def fixture(self):
        rgb=np.full((140,180,3),255,np.uint8)
        rgb[68:71,20:41]=[150,0,0];rgb[68:71,41:151]=[0,130,0]
        target={'components':{},'pins':{},'nets':{'n':{'hyperGraph':'()',
            'edges':{'e':[{'x':30.,'y':70.},{'x':150.,'y':70.}]}}}}
        return target,rgb
    def test_enabled_corrects_existing_endpoint_only(self):
        target,rgb=self.fixture();original=copy.deepcopy(target)
        out,audit=agent.finalize_contact_precision(target,rgb,'fixture',
            {'precise_wire_contacts':{'enabled':True}})
        self.assertEqual(target,original)
        self.assertEqual(out['nets']['n']['edges']['e'][0],{'x':41.,'y':71.})
        self.assertEqual(out['nets']['n']['hyperGraph'],'()')
        self.assertEqual(out['components'],target['components'])
        self.assertEqual(out['pins'],target['pins'])
        self.assertTrue(audit['precise_wire_contacts']['changes'])
    def test_missing_or_disabled_config_preserves_input(self):
        target,rgb=self.fixture()
        for config in [{},{'precise_wire_contacts':{'enabled':False},'precise_pin_contacts':{'enabled':False}}]:
            out,audit=agent.finalize_contact_precision(target,rgb,'fixture',config)
            self.assertEqual(out,target)
    def test_repeated_entrypoint_is_idempotent(self):
        target,rgb=self.fixture();config={'precise_wire_contacts':{'enabled':True}}
        out,_=agent.finalize_contact_precision(target,rgb,'fixture',config)
        again,audit=agent.finalize_contact_precision(out,rgb,'fixture',config)
        self.assertEqual(out,again)
        self.assertFalse(audit['precise_wire_contacts']['changes'])

if __name__=='__main__':unittest.main()
