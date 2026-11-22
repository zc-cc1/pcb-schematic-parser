import unittest
from copy import deepcopy
from types import SimpleNamespace
import cv2
import numpy as np

from pcb_parser.terminal_topology import (ground_branch_proof,recover_ground_branches,
    no_connect_marker,prune_no_connect_nets,connected_corridor)


def refresh(d):
    r,g,b=[d.rgb[:,:,i].astype(int) for i in range(3)]
    d.red=(r>g+35)&(r>b+35)&(g<150)&(b<150)
    d.green=(g>r+20)&(g>b+20)&(g>60)
    return d


def base():
    d=SimpleNamespace(rgb=np.full((140,140,3),255,np.uint8),h=140,w=140,
        out={'components':{},'pins':{},'nets':{}},rmap={},pin_document={'pin_candidates':[]},events=[])
    d.box=lambda c:[c['bbox'][0],140-c['bbox'][3],c['bbox'][2],140-c['bbox'][1]]
    d.point=lambda p:[p['point']['x'],140-p['point']['y']]
    return d


def component(kind,box):return {'type':kind,'bbox':[box[0],140-box[3],box[2],140-box[1]]}
def pin(x,y):return {'pinname':'PP','point':{'x':float(x),'y':140-float(y)}}


def ground():
    d=base();d.out['components']={'CAP':component('c',[50,20,70,50]),'G':component('gnd',[50,90,70,110])}
    d.out['pins']={'CAP':{'pin_1':pin(60,15),'pin_2':pin(60,58)},'G':{'pin_1':pin(61.5,83)}}
    d.rmap={'G':{'component_id':'G','type':'gnd','detection_confidence':.9}}
    d.pin_document={'pin_candidates':[{'candidate_id':'source','owner':'G','owner_status':'assigned',
                                     'point':[61.5,83.],'confidence':.99}]}
    cv2.line(d.rgb,(60,58),(60,76),(0,160,0),1)
    cv2.line(d.rgb,(60,76),(60,90),(190,0,0),1)
    return refresh(d)


def nc(axis=0,direction=1):
    d=base();p=[70,70];p[axis]=90 if direction>0 else 50
    box=[50,50,90,90]
    box[axis]=20 if direction>0 else 80;box[axis+2]=60 if direction>0 else 120
    d.out['components']={'IC':component('box',box)};d.out['pins']={'IC':{'pin_3':pin(*p)}}
    boundary=box[axis+2] if direction>0 else box[axis]
    a=p.copy();a[axis]=boundary;b=p.copy()
    cv2.line(d.rgb,tuple(a),tuple(b),(190,0,0),1)
    cv2.line(d.rgb,(p[0]-8,p[1]-8),(p[0]+8,p[1]+8),(0,160,0),2)
    cv2.line(d.rgb,(p[0]-8,p[1]+8),(p[0]+8,p[1]-8),(0,160,0),2)
    a[axis]+=direction*3;b[axis]-=direction*10
    d.out['nets']={'stub':{'hyperGraph':'(IC.3)','edges':{'e':[pin(*a)['point'],pin(*b)['point']]}}}
    return refresh(d)


class MixedGroundTests(unittest.TestCase):
    def test_observed_red_green_transition_accepts_shifted_source_point(self):
        d=ground();p=ground_branch_proof(d,'CAP.2','G.1')
        self.assertTrue(p['connected_ink']);self.assertEqual(p['source_candidate'],'source')

    def test_one_pixel_white_gap_cannot_be_bridged(self):
        d=ground();d.rgb[67,:]=255;refresh(d)
        self.assertIsNone(ground_branch_proof(d,'CAP.2','G.1'))

    def test_unobserved_ground_endpoint_is_rejected(self):
        d=ground();d.pin_document['pin_candidates']=[]
        self.assertIsNone(ground_branch_proof(d,'CAP.2','G.1'))

    def test_ambiguous_or_weak_source_is_rejected(self):
        for change in [{'owner_status':'ambiguous_owner'},{'confidence':.79}]:
            d=ground();d.pin_document['pin_candidates'][0].update(change)
            self.assertIsNone(ground_branch_proof(d,'CAP.2','G.1'))

    def test_real_branch_in_corridor_defers_to_graph(self):
        d=ground();cv2.line(d.rgb,(60,68),(80,68),(0,160,0),1);refresh(d)
        self.assertIsNone(ground_branch_proof(d,'CAP.2','G.1'))

    def test_intervening_component_is_not_crossed(self):
        d=ground();d.out['components']['other']=component('r',[56,63,64,72])
        self.assertIsNone(ground_branch_proof(d,'CAP.2','G.1'))

    def test_supply_is_not_treated_as_ground(self):
        d=ground();d.out['components']['G']['type']='v'
        self.assertIsNone(ground_branch_proof(d,'CAP.2','G.1'))

    def test_existing_memberships_are_not_reassigned(self):
        d=ground();d.out['nets']['old']={'hyperGraph':'(CAP.2)','edges':{}}
        old=deepcopy(d.out);recover_ground_branches(d);self.assertEqual(d.out,old)

    def test_new_branch_is_idempotent_and_pins_unchanged(self):
        d=ground();pins=deepcopy(d.out['pins']);recover_ground_branches(d)
        self.assertEqual(len(d.out['nets']),1);first=deepcopy(d.out)
        recover_ground_branches(d);self.assertEqual(d.out,first);self.assertEqual(d.out['pins'],pins)

    def test_multiple_ground_identities_are_not_chosen_by_distance(self):
        d=ground();d.out['components']['G2']=deepcopy(d.out['components']['G']);d.out['pins']['G2']=deepcopy(d.out['pins']['G'])
        d.rmap['G2']={**d.rmap['G'],'component_id':'G2'}
        d.pin_document['pin_candidates'].append({**d.pin_document['pin_candidates'][0],'candidate_id':'other','owner':'G2'})
        recover_ground_branches(d);self.assertFalse(d.out['nets'])

    def test_corridor_requires_connected_ink_in_both_orientations(self):
        mask=np.zeros((30,30),bool);mask[15,5:25]=True
        self.assertTrue(connected_corridor(mask,[5,15],[24,15]));mask[15,14]=False
        self.assertFalse(connected_corridor(mask,[5,15],[24,15]))
        self.assertFalse(connected_corridor(mask.T,[15,5],[15,24]))


class ImageNoConnectTests(unittest.TestCase):
    def test_four_pin_directions_accept_terminal_x(self):
        for axis in (0,1):
            for direction in (-1,1):self.assertIsNotNone(no_connect_marker(nc(axis,direction),'IC.3'))

    def test_removal_preserves_pp_identity_and_location(self):
        d=nc();before=deepcopy(d.out);prune_no_connect_nets(d)
        self.assertFalse(d.out['nets']);self.assertEqual(d.out['pins'],before['pins']);self.assertEqual(d.out['components'],before['components'])

    def test_removal_is_idempotent(self):
        d=nc();prune_no_connect_nets(d);old=deepcopy(d.out);count=len(d.events)
        prune_no_connect_nets(d);self.assertEqual(d.out,old);self.assertEqual(len(d.events),count)

    def test_slash_plus_or_dot_is_not_no_connect(self):
        for shape in ['slash','plus','dot']:
            d=nc();d.rgb[d.green]=255
            if shape=='slash':cv2.line(d.rgb,(82,62),(98,78),(0,160,0),2)
            elif shape=='plus':
                cv2.line(d.rgb,(82,70),(98,70),(0,160,0),2);cv2.line(d.rgb,(90,62),(90,78),(0,160,0),2)
            else:cv2.circle(d.rgb,(90,70),8,(0,160,0),-1)
            refresh(d);self.assertIsNone(no_connect_marker(d,'IC.3'))

    def test_wire_continuing_after_x_is_not_isolated_no_connect(self):
        d=nc();cv2.line(d.rgb,(99,70),(116,70),(0,160,0),1);refresh(d)
        self.assertIsNone(no_connect_marker(d,'IC.3'))

    def test_real_diagonal_wire_crossing_is_not_terminal_marker(self):
        d=nc();cv2.line(d.rgb,(72,52),(108,88),(0,160,0),2);refresh(d)
        self.assertIsNone(no_connect_marker(d,'IC.3'))

    def test_marker_at_another_pin_is_rejected(self):
        d=nc();d.out['pins']['IC']['pin_3']=pin(84,70)
        self.assertIsNone(no_connect_marker(d,'IC.3'))

    def test_no_red_lead_does_not_turn_nearby_x_into_pin_evidence(self):
        d=nc();d.rgb[d.red]=255;refresh(d)
        self.assertIsNone(no_connect_marker(d,'IC.3'))

    def test_network_with_other_members_is_never_deleted(self):
        d=nc();d.out['nets']['stub']['hyperGraph']='(IC.3,OTHER.1)';old=deepcopy(d.out)
        prune_no_connect_nets(d);self.assertEqual(d.out,old)

    def test_large_or_remote_network_geometry_is_never_deleted(self):
        d=nc();d.out['nets']['stub']['edges']['extra']=[{'x':20.,'y':100.},{'x':90.,'y':100.}];old=deepcopy(d.out)
        prune_no_connect_nets(d);self.assertEqual(d.out,old)

    def test_ordinary_component_cannot_claim_ic_no_connect_rule(self):
        d=nc();d.out['components']['IC']['type']='r'
        self.assertIsNone(no_connect_marker(d,'IC.3'))


if __name__=='__main__':unittest.main()
