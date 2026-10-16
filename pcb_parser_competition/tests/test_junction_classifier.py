"""Learned decision invariants, rejection and source-group isolation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.junction_classifier import centered_views,JunctionPredictor
from pcb_parser.junction_training_data import ReviewedDataset
from pcb_parser.wire_graph import build_wire_graph
from test_wire_graph import pins,CFG


class LearnedJunctionTests(unittest.TestCase):
    def setUp(self):
        self.wire=np.zeros((64,64),np.float32);self.wire[30,5:56]=1;self.wire[5:56,30]=1
        self.source=pins([[5,30],[55,30],[30,5],[30,55]])
        self.base=build_wire_graph(self.wire,self.source,CFG)[0]
        self.node=self.base['junction_candidates'][0]['junction_id']

    def decision(self,label,accepted=True):
        return {self.node:{'label':label,'accepted':accepted,'decision_source':'learned_RGB_junction_CNN',
                           'rotation_agreement':True,'rotation_min_confidence':.99,'threshold':.95}}

    def test_learned_join_connects_all_arms_and_preserves_pin_points(self):
        result,_=build_wire_graph(self.wire,self.source,CFG,self.decision('connected_junction'))
        self.assertEqual(len({p['network_id'] for p in result['pin_attachments']}),1)
        self.assertEqual(result['summary']['uncertain_junctions'],0)
        self.assertEqual([p['point'] for p in result['pin_attachments']],[p['point'] for p in self.source['pin_candidates']])

    def test_learned_crossing_separates_arms_and_resolves_semantics(self):
        result,_=build_wire_graph(self.wire,self.source,CFG,self.decision('crossing_without_connection'))
        self.assertEqual(len({p['network_id'] for p in result['pin_attachments']}),2)
        self.assertTrue(all(not n['unresolved_junction_ids'] for n in result['networks']))

    def test_abstention_preserves_v1_membership_and_uncertainty(self):
        result,_=build_wire_graph(self.wire,self.source,CFG,self.decision('connected_junction',False))
        self.assertEqual(result['pin_attachments'],self.base['pin_attachments'])
        self.assertEqual(result['summary']['uncertain_junctions'],1)

    def test_invalid_confidence_cannot_join(self):
        decisions=self.decision('connected_junction');decisions[self.node]['rotation_min_confidence']=.5
        with self.assertRaises(ValueError):build_wire_graph(self.wire,self.source,CFG,decisions)
        decisions[self.node]['rotation_min_confidence']=float('nan')
        with self.assertRaises(ValueError):build_wire_graph(self.wire,self.source,CFG,decisions)

    def test_unknown_junction_cannot_change_graph(self):
        with self.assertRaises(ValueError):build_wire_graph(self.wire,self.source,CFG,{'missing':{}})

    def test_two_adjacent_resolved_junctions_keep_shared_edge(self):
        wire=self.wire.copy();wire[5:56,40]=1
        base=build_wire_graph(wire,self.source,CFG)[0]
        template=next(iter(self.decision('crossing_without_connection').values()))
        decisions={j['junction_id']:deepcopy(template) for j in base['junction_candidates']}
        result,_=build_wire_graph(wire,self.source,CFG,decisions)
        self.assertEqual(result['summary']['graph_edges'],base['summary']['graph_edges'])
        self.assertEqual(result['summary']['uncertain_junctions'],0)

    def test_roi_border_is_white_and_original_center_required(self):
        image=Image.new('RGB',(32,32),'black');views=centered_views(image,[0,0])
        self.assertEqual(tuple(views.shape),(6,64,64));self.assertGreater(float(views[:,:8,:8].mean()),.99)
        with self.assertRaises(ValueError):centered_views(image,[-1,0])

    def test_split_has_no_shared_source_hashes(self):
        records=json.loads((ROOT/'data/junction_classifier_v1/reviewed_annotations.json').read_text(encoding='utf-8'))['records']
        groups={s:{r['source_image_sha256'] for r in records if r['split']==s} for s in ['train','calibration','test']}
        self.assertFalse(groups['train']&groups['test']);self.assertFalse(groups['calibration']&groups['test'])
        original=json.loads((ROOT/'data/topology_review_v1/annotations.json').read_text(encoding='utf-8'))
        self.assertTrue(all(r['label'] is None and not r['reviewed'] for r in original['records']))

    def test_dataset_iteration_stops_at_declared_length(self):
        records=json.loads((ROOT/'data/junction_classifier_v1/reviewed_annotations.json').read_text(encoding='utf-8'))['records']
        dataset=ReviewedDataset(ROOT/'data/topology_review_v1',[r for r in records if r['split']=='calibration'])
        self.assertEqual(len(list(dataset)),len(dataset))
        with self.assertRaises(IndexError):dataset[len(dataset)]

    def test_checkpoint_hash_guard_and_unsupported_degree_rejection(self):
        weights=ROOT/'weights/junction_classifier_v1.pt'
        if not weights.is_file():self.skipTest('Pilot not trained yet')
        with self.assertRaises(ValueError):JunctionPredictor(weights,expected_sha256='changed')
        predictor=JunctionPredictor(weights)
        candidate={'junction_id':'J1','point':[30,30],'degree':5}
        result=predictor.predict(np.full((64,64,3),255,np.uint8),[candidate])['J1']
        self.assertFalse(result['accepted']);self.assertEqual(result['abstention_reason'],'unsupported_degree')


if __name__=='__main__':unittest.main()
