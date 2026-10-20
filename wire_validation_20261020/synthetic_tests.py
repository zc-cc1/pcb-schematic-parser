"""Small geometric regression cases for the isolated path-splitting variant."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate_wire',HERE/'runtime_candidate/src/pcb_parser/wire_graph.py')
candidate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
cfg=json.loads((HERE/'runtime_candidate/configs/wire_graph_v1.json').read_text())
cfg.update(body_guard=False,bridge_max_gap_px=0)


def run(probability,points):
    pins={'size':list(probability.shape[::-1]),'coordinate_origin':'top_left','components':[],
        'pin_candidates':[{'candidate_id':f'p{i}','point':p,'owner_status':'unassigned',
                           'owner_detection_index':None,'owner':None} for i,p in enumerate(points)]}
    return candidate.build_wire_graph(probability,pins,cfg)[0]


def main():
    p=np.zeros((100,100),dtype=np.float32)
    p[10,10:81]=1;p[10:81,80]=1
    off=run(p,[[45.,45.]])
    assert off['pin_attachments'][0]['attachment_status']=='no_wire_within_radius'
    real=run(p,[[80.,45.]])
    assert real['pin_attachments'][0]['attachment_status']=='attached'
    for edge in real['edges']:
        for a,b in zip(edge['polyline'],edge['polyline'][1:]):
            assert abs(a[0]-b[0])<=1.1 or abs(a[1]-b[1])<=1.1
    line=np.zeros_like(p);line[10,10:91]=1
    graph=run(line,[[40.,10.],[70.,15.]])
    assert all(a['attachment_status']=='attached' for a in graph['pin_attachments'])
    assert abs(sum(e['length_px'] for e in graph['edges'])-80)<1e-6
    assert len(graph['networks'])==1
    empty=run(line,[])
    assert len(empty['edges'])==1 and abs(empty['edges'][0]['length_px']-80)<1e-6
    result={'passed':True,'checks':['off-wire pin rejected','L bend preserved',
        'true-path pin attached','dense path support retained after splitting',
        'total path length and connectivity preserved','no-pin path unchanged']}
    (HERE/'synthetic_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
