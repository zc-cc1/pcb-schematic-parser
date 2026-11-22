import unittest
from copy import deepcopy
import cv2
import numpy as np
from pcb_parser.junction_axis_offsets import refine_junction_axis_offsets


def fixture(*,offset=2,source=True):
    rgb=np.full((300,400,3),255,np.uint8)
    cv2.line(rgb,(120,150-offset),(200,150-offset),(0,160,0),1)
    cv2.line(rgb,(200,150),(280,150),(0,160,0),1)
    cv2.line(rgb,(200,150),(200,210),(0,160,0),1)
    cv2.circle(rgb,(200,150),5,(190,0,0),-1)
    def edge(a,b):return [{'x':float(a[0]),'y':float(a[1])},{'x':float(b[0]),'y':float(b[1])}]
    target={'components':{},'pins':{},'nets':{'N':{'hyperGraph':'(A.1,B.1,C.1)','edges':{
        'edge_1':edge((120,150+offset),(200,150)),
        'edge_2':edge((200,150),(280,150)),
        'edge_3':edge((200,150),(200,90))}}}}
    graph={'image_sha256':'same','size':[400,300],'coordinate_origin':'top_left','prediction_only':True,
           'nodes':[{'node_id':'J','point':[200,150],'kind':'junction'},
                    {'node_id':'A','point':[120,150-offset]},
                    {'node_id':'B','point':[280,150]}, {'node_id':'C','point':[200,210]}],
           'edges':[{'edge_id':'a','source':'A','target':'J','polyline':[[120,150-offset],[195,150-offset],[200,150]]},
                    {'edge_id':'b','source':'J','target':'B','polyline':[[200,150],[280,150]]},
                    {'edge_id':'c','source':'J','target':'C','polyline':[[200,150],[200,210]]}],
           'networks':[{'network_id':'N','node_ids':['A','B','C','J'],'edge_ids':['a','b','c']}] if source else []}
    return target,rgb,graph


def run(data,**cfg):
    return refine_junction_axis_offsets(*data,'same',cfg={'enabled':True,**cfg})


class JunctionAxisOffsetsTests(unittest.TestCase):
    def test_observed_offset_junction_retains_the_short_step(self):
        data=fixture();before=deepcopy(data[0]);out,audit=run(data)
        self.assertEqual(data[0],before)
        self.assertEqual(out['nets']['N']['edges']['edge_1'][1],{'x':200.,'y':152.})
        self.assertEqual(out['nets']['N']['edges']['edge_4'],[{'x':200.,'y':152.},{'x':200.,'y':150.}])
        self.assertEqual(out['nets']['N']['hyperGraph'],'(A.1,B.1,C.1)')
        self.assertEqual(len(audit['changes']),1)
        self.assertEqual(run((out,data[1],data[2]))[0],out)

    def test_aligned_arms_do_not_create_a_short_edge(self):
        data=fixture(offset=0);self.assertEqual(run(data)[0],data[0])

    def test_mirrored_junction_moves_the_other_horizontal_arm(self):
        target,rgb,graph=fixture();rgb=cv2.flip(rgb,1)
        for net in target['nets'].values():
            for edge in net['edges'].values():
                for p in edge:p['x']=399-p['x']
        for node in graph['nodes']:node['point'][0]=399-node['point'][0]
        for edge in graph['edges']:
            for p in edge['polyline']:p[0]=399-p[0]
        out,_=run((target,rgb,graph))
        self.assertEqual(out['nets']['N']['edges']['edge_4'],[{'x':199.,'y':152.},{'x':199.,'y':150.}])

    def test_transposed_image_handles_vertical_rail_offsets(self):
        target,rgb,graph=fixture();rgb=np.transpose(rgb,(1,0,2)).copy();graph['size']=[300,400]
        for net in target['nets'].values():
            for edge in net['edges'].values():
                for p in edge:p['x'],p['y']=300-p['y'],400-p['x']
        for node in graph['nodes']:node['point'].reverse()
        for edge in graph['edges']:
            for p in edge['polyline']:p.reverse()
        out,_=run((target,rgb,graph))
        self.assertEqual(out['nets']['N']['edges']['edge_4'],[{'x':148.,'y':200.},{'x':150.,'y':200.}])

    def test_missing_source_junction_proof_keeps_export(self):
        data=fixture(source=False);self.assertEqual(run(data)[0],data[0])

    def test_source_arm_must_confirm_the_observed_offset(self):
        data=fixture();data[2]['edges'][0]['polyline']=[[120,150],[200,150]]
        self.assertEqual(run(data)[0],data[0])

    def test_no_red_dot_does_not_create_an_offset(self):
        data=fixture();data[1][144:156,194:206]=255
        self.assertEqual(run(data)[0],data[0])

    def test_four_arm_crossing_is_kept(self):
        data=fixture();cv2.line(data[1],(200,100),(200,145),(0,160,0),1)
        data[0]['nets']['N']['edges']['edge_4']=[{'x':200.,'y':150.},{'x':200.,'y':200.}]
        self.assertEqual(run(data)[0],data[0])

    def test_change_budget_withholds_the_whole_delta(self):
        data=fixture();out,audit=run(data,maximum_changes=0)
        self.assertEqual(out,data[0]);self.assertEqual(audit['status'],'withheld_budget')

    def test_image_binding_is_required(self):
        data=fixture();data[2]['image_sha256']='stale'
        with self.assertRaises(ValueError):run(data)


if __name__=='__main__':unittest.main()
