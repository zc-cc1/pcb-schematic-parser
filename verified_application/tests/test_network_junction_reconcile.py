import unittest
from copy import deepcopy

import cv2
import numpy as np

from pcb_parser.network_junction_reconcile import reconcile_network_junctions


def point(x,y):return {'x':float(x),'y':float(y)}
def edge(a,b):return [point(*a),point(*b)]


def fixture(first,second,*,draw=True):
    target={'components':{'A':{'type':'box','bbox':[0,0,10,10]},
                          'B':{'type':'box','bbox':[110,110,120,120]}},
            'pins':{},'metadata':{'keep':True},
            'nets':{'left':{'hyperGraph':'(A.1)','edges':{'edge_1':edge(*first)}},
                    'right':{'hyperGraph':'(B.1)','edges':{'edge_1':edge(*second)}}}}
    rgb=np.full((128,128,3),255,np.uint8)
    if draw:
        for a,b in (first,second):
            cv2.line(rgb,(a[0],128-a[1]),(b[0],128-b[1]),(0,160,0),2)
    graph={'image_sha256':'same-image','size':[128,128],
           'coordinate_origin':'top_left','prediction_only':True}
    return target,rgb,graph


def run(data,dots=(),**config):
    return reconcile_network_junctions(*data,'same-image',dots,cfg={'enabled':True,**config})


class NetworkJunctionReconcileTests(unittest.TestCase):
    def test_long_collinear_overlap_merges_members_and_preserves_geometry(self):
        data=fixture(((20,50),(100,50)),((40,50),(120,50)))
        before=deepcopy(data[0]);out,audit=run(data)
        self.assertEqual(data[0],before)
        self.assertEqual(len(out['nets']),1)
        self.assertEqual(next(iter(out['nets'].values()))['hyperGraph'],'(A.1,B.1)')
        self.assertEqual(len(next(iter(out['nets'].values()))['edges']),2)
        self.assertEqual(out['components'],before['components'])
        self.assertTrue(any(p['kind']=='long_collinear_overlap' for p in audit['proposals']))
        self.assertEqual(run((out,data[1],data[2]))[0],out)

    def test_short_overlap_is_withheld(self):
        data=fixture(((20,50),(100,50)),((60,50),(120,50)))
        self.assertEqual(run(data)[0],data[0])

    def test_observed_red_junction_bridges_two_ink_supported_arms(self):
        data=fixture(((20,60),(50,60)),((60,50),(60,20)))
        cv2.line(data[1],(50,68),(60,68),(0,160,0),2)
        cv2.line(data[1],(60,78),(60,68),(0,160,0),2)
        cv2.circle(data[1],(60,68),5,(190,0,0),-1)
        out,audit=run(data,[[60.,60.]])
        self.assertEqual(len(out['nets']),1)
        self.assertTrue(any(p['kind']=='observed_red_junction' for p in audit['proposals']))

    def test_unmarked_crossing_is_not_merged(self):
        data=fixture(((20,60),(100,60)),((60,20),(60,100)))
        self.assertEqual(run(data)[0],data[0])

    def test_white_gap_at_claimed_dot_is_not_merged(self):
        data=fixture(((20,60),(50,60)),((60,50),(60,20)))
        self.assertEqual(run(data,[[60.,60.]])[0],data[0])

    def test_image_binding_is_required(self):
        data=fixture(((20,50),(100,50)),((40,50),(120,50)))
        data[2]['image_sha256']='other'
        with self.assertRaises(ValueError):run(data)


if __name__=='__main__':unittest.main()
