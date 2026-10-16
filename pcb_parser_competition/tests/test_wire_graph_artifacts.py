import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pcb_parser.wire_graph_artifacts import partial_submission
from run_wire_graph import validate_cache


def fixture():
    records=[{'component_id':'U1','designator':'U1','detection_index':0,'type':'ic','Name':'Chip','value':None,'bbox_top_left':[5,5,35,35]}]
    pins={'size':[100,80],'pin_candidates':[{'candidate_id':f'P{i}','owner':'U1','owner_status':'assigned','number':str(i+1),'pinname':f'IN{i}','point':[10+i*20,40]} for i in range(2)]}
    graph={'networks':[{'network_id':'N1','pin_candidate_ids':['P0','P1'],'unresolved_junction_ids':[],'edge_ids':['W1']}],
           'gap_candidates':[],'edges':[{'edge_id':'W1','polyline':[[10,40],[30,40]]}]}
    return pins,graph,records


class ArtifactTests(unittest.TestCase):
    def test_partial_export_uses_original_tip_and_one_H_minus_y_transform(self):
        pins,graph,records=fixture();output,coverage=partial_submission(pins,graph,records)
        self.assertEqual(output['pins']['U1']['pin_1']['point'],{'x':10,'y':40})
        self.assertEqual(output['components']['U1']['bbox'],[5.,45.,35.,75.])
        self.assertEqual(output['nets']['N1']['hyperGraph'],'(U1.1,U1.2)')
        self.assertTrue(coverage['local_contract_valid'])
        self.assertFalse(coverage['complete_competition_JSON'])

    def test_unknown_pin_name_is_not_guessed_empty_or_omitted_from_net_set(self):
        pins,graph,records=fixture();pins['pin_candidates'][1]['pinname']=None
        output,coverage=partial_submission(pins,graph,records)
        self.assertEqual(coverage['exported_pins'],1)
        self.assertEqual(output['nets'],{})
        self.assertEqual(coverage['blank_pin_names_guessed'],0)

    def test_uncertain_junction_and_gap_block_exported_net(self):
        pins,graph,records=fixture();graph['networks'][0]['unresolved_junction_ids']=['J1']
        output,_=partial_submission(pins,graph,records);self.assertEqual(output['nets'],{})
        graph['networks'][0]['unresolved_junction_ids']=[]
        graph['gap_candidates']=[{'network_ids':['N1']}]
        output,_=partial_submission(pins,graph,records);self.assertEqual(output['nets'],{})

    def test_probability_cache_requires_image_model_origin_and_byte_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'wire.npz';np.savez_compressed(path,wire=np.zeros((8,8),np.float32))
            payload={'image_sha256':'image','size':[8,8],'coordinate_origin':'top_left','weights_sha256':'model',
                     'inference_config_sha256':'config','prediction_only':True,'probabilities_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            validate_cache(payload,path,'image',[8,8],'model','config')
            for bad in [{'image_sha256':'other'},{'weights_sha256':'other'},{'coordinate_origin':'bottom_left'},{'prediction_only':False},{'probabilities_sha256':'other'}]:
                with self.assertRaises(ValueError):validate_cache({**payload,**bad},path,'image',[8,8],'model','config')


if __name__=='__main__':unittest.main()
