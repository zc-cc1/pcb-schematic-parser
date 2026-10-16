"""Topology invariants on controlled learned probability maps."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.wire_graph import build_wire_graph,pixel_adjacency,polyline

CFG=json.loads((ROOT/'configs/wire_graph_v1.json').read_text(encoding='utf-8'))


def pins(points,size=(64,64),components=None):
    return {'size':list(size),'coordinate_origin':'top_left','components':components or [],
            'pin_candidates':[{'candidate_id':f'P{i}','point':list(point),'owner':'U1','owner_status':'assigned',
                               'owner_detection_index':0,'number':str(i+1),'pinname':f'X{i}'} for i,point in enumerate(points)]}


class WireGraphTests(unittest.TestCase):
    def run_case(self,wire,points,cfg=None,components=None):
        return build_wire_graph(wire,pins(points,wire.shape[::-1],components),cfg or CFG)[0]

    def test_straight_wire_attaches_tips_without_moving_coordinates(self):
        wire=np.zeros((64,64),np.float32);wire[20,5:51]=.95
        points=[[5.25,20.2],[50.1,20.3]]
        result=self.run_case(wire,points)
        self.assertEqual(result['summary']['multi_pin_network_proposals'],1)
        self.assertEqual([p['point'] for p in result['pin_attachments']],points)

    def test_T_join_connects_three_branches(self):
        wire=np.zeros((64,64),np.float32);wire[30,5:56]=1;wire[5:31,30]=1
        result=self.run_case(wire,[[5,30],[55,30],[30,5]])
        self.assertEqual(result['summary']['uncertain_junctions'],0)
        self.assertEqual(len({p['network_id'] for p in result['pin_attachments']}),1)

    def test_four_way_crossing_stays_two_networks_and_is_uncertain(self):
        wire=np.zeros((64,64),np.float32);wire[30,5:56]=1;wire[5:56,30]=1
        result=self.run_case(wire,[[5,30],[55,30],[30,5],[30,55]])
        a=result['pin_attachments']
        self.assertEqual(a[0]['network_id'],a[1]['network_id'])
        self.assertEqual(a[2]['network_id'],a[3]['network_id'])
        self.assertNotEqual(a[0]['network_id'],a[2]['network_id'])
        self.assertEqual(result['summary']['uncertain_junctions'],1)

    def test_observed_crossing_variant_retains_uncertainty_even_if_connected(self):
        wire=np.zeros((64,64),np.float32);wire[30,5:56]=1;wire[5:56,30]=1
        cfg={**CFG,'junction_policy':'observed_connected'}
        result=self.run_case(wire,[[5,30],[30,55]],cfg)
        self.assertEqual(result['summary']['uncertain_junctions'],1)
        self.assertEqual(result['networks'][0]['status'],'requires_junction_review')

    def test_L_bend_has_no_diagonal_shortcut_junction(self):
        skeleton=np.zeros((6,6),bool);skeleton[1,1:5]=True;skeleton[1:5,4]=True
        adjacency=pixel_adjacency(skeleton)
        self.assertTrue(all(len(v)<=2 for v in adjacency.values()))

    def test_diagonal_stroke_remains_connected(self):
        wire=np.zeros((64,64),np.float32)
        for i in range(5,51):wire[i,i]=1
        result=self.run_case(wire,[[5,5],[50,50]])
        self.assertEqual(result['summary']['multi_pin_network_proposals'],1)

    def test_gap_is_recorded_and_never_filled(self):
        wire=np.zeros((64,64),np.float32);wire[20,5:26]=1;wire[20,32:56]=1
        result=self.run_case(wire,[[5,20],[55,20]])
        self.assertNotEqual(result['pin_attachments'][0]['network_id'],result['pin_attachments'][1]['network_id'])
        self.assertEqual(result['summary']['gap_candidates_not_filled'],1)
        self.assertEqual(result['summary']['automatic_gap_connections'],0)

    def test_nearby_parallel_wires_leave_pin_attachment_ambiguous(self):
        wire=np.zeros((64,64),np.float32);wire[20,5:56]=1;wire[24,5:56]=1
        result=self.run_case(wire,[[30,22]])
        self.assertEqual(result['pin_attachments'][0]['attachment_status'],'ambiguous_nearby_networks')
        self.assertIsNone(result['pin_attachments'][0]['network_id'])

    def test_component_body_cannot_short_its_two_terminals(self):
        wire=np.zeros((64,64),np.float32);wire[30,5:56]=1
        component={'detection_index':0,'component_id':'U1','designator':'U1','bbox_top_left':[20,22,40,38]}
        result=self.run_case(wire,[[20,30],[40,30]],components=[component])
        self.assertNotEqual(result['pin_attachments'][0]['network_id'],result['pin_attachments'][1]['network_id'])
        self.assertEqual(result['summary']['gap_candidates_not_filled'],0)

    def test_closed_cycle_is_preserved_as_self_loop(self):
        wire=np.zeros((64,64),np.float32);wire[10,10:41]=1;wire[40,10:41]=1;wire[10:41,10]=1;wire[10:41,40]=1
        result=self.run_case(wire,[[10,25],[40,25]])
        self.assertTrue(any(e['source']==e['target'] for e in result['edges']))
        self.assertEqual(result['summary']['multi_pin_network_proposals'],1)

    def test_empty_map_preserves_unattached_pins(self):
        result=self.run_case(np.zeros((64,64),np.float32),[[5,5]])
        self.assertEqual(result['summary']['attached_pins'],0)
        self.assertEqual(len(result['pin_attachments']),1)

    def test_invalid_probability_size_and_coordinates_rejected(self):
        source=pins([[5,5]])
        with self.assertRaises(ValueError):build_wire_graph(np.zeros((32,32)),source,CFG)
        wrong=np.zeros((64,64));wrong[0,0]=np.nan
        with self.assertRaises(ValueError):build_wire_graph(wrong,source,CFG)
        with self.assertRaises(ValueError):build_wire_graph(np.zeros((64,64)),pins([[64,5]]),CFG)

    def test_duplicate_pin_ids_rejected(self):
        source=pins([[5,5],[20,5]])
        source['pin_candidates'][1]['candidate_id']='P0'
        with self.assertRaises(ValueError):build_wire_graph(np.zeros((64,64)),source,CFG)

    def test_polyline_retains_real_corner(self):
        simplified=polyline([[x,5] for x in range(5,21)]+[[20,y] for y in range(6,21)],1)
        self.assertEqual(simplified,[[5.,5.],[20.,5.],[20.,20.]])


if __name__=='__main__':
    unittest.main()
