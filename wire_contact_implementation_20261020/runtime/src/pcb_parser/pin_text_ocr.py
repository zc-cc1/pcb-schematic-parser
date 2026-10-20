"""RGB-only local text inference around learned pins; no annotation access."""
import hashlib
import importlib.util
from pathlib import Path
import time

import cv2
import numpy as np

from .component_text import merge_texts,restore_polygon
from .pin_text import local_views


def create_engine(cfg):
    from rapidocr import RapidOCR
    package=Path(importlib.util.find_spec('rapidocr').origin).parent
    names={'Det':'PP-OCRv6_det_small.onnx','Rec':'PP-OCRv6_rec_small.onnx','Cls':'ch_ppocr_mobile_v2.0_cls_mobile.onnx'}
    paths={role:Path(__file__).resolve().parents[2]/'assets'/'ocr'/name for role,name in names.items()}
    if not all(p.is_file() for p in paths.values()):
        raise FileNotFoundError('Expected installed RapidOCR models are missing; install the existing project OCR weights first')
    params={f'{role}.model_path':str(path) for role,path in paths.items()}
    params.update({'Det.limit_side_len':2048,'Det.limit_type':'max','Global.max_side_len':2048,
                   'EngineConfig.onnxruntime.intra_op_num_threads':cfg['onnx_threads'],
                   'EngineConfig.onnxruntime.inter_op_num_threads':1})
    engine=RapidOCR(params=params)
    fingerprint={role:{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for role,path in paths.items()}
    return engine,fingerprint


def run_local_ocr(rgb,engine,ownership,binding,texts,cfg):
    views,skipped=local_views(ownership,binding,cfg)
    observations=[]
    for row in texts:
        observations.append({key:value for key,value in row.items() if key!='observations'})
        for old in row.get('observations',[]):
            x1,y1,x2,y2=old.get('bbox',row['bbox'])
            observations.append({'text':old['text'],'confidence':old['confidence'],
                                 'polygon':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],'view':old.get('view','external')})
    timings=[]
    rejected=[]
    begin=time.perf_counter()
    for index,view in enumerate(views):
        x1,y1,x2,y2=view['crop']
        original=rgb[y1:y2,x1:x2]
        for rotation in cfg['local_ocr']['rotations']:
            patch=cv2.rotate(original,cv2.ROTATE_90_CLOCKWISE) if rotation==90 else original
            scale=cfg['local_ocr']['scale']
            patch=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            start=time.perf_counter()
            detected=engine(np.ascontiguousarray(patch[:,:,::-1]),use_det=True,use_cls=True,use_rec=True)
            count=0
            if detected.boxes is not None:
                for points,text,confidence in zip(detected.boxes,detected.txts,detected.scores):
                    if float(confidence)<cfg['local_ocr']['min_observation_confidence']:
                        continue
                    polygon=restore_polygon(points,view['crop'],scale,rotation)
                    polygon=np.clip(polygon,[0,0],[rgb.shape[1],rgb.shape[0]]).tolist()
                    box=np.asarray(polygon)
                    guard=cfg['local_ocr']['crop_boundary_guard_px']
                    if box[:,0].min()<=x1+guard or box[:,0].max()>=x2-guard or box[:,1].min()<=y1+guard or box[:,1].max()>=y2-guard:
                        rejected.append({**view,'rotation':rotation,'text':str(text),'confidence':float(confidence),
                                         'polygon':polygon,'reason':'OCR_touches_crop_boundary_possible_truncated_glyphs'})
                        continue
                    observations.append({'text':str(text),'confidence':float(confidence),'polygon':polygon,'view':f'pin_local_{index}_r{rotation}'})
                    count+=1
            timings.append({**view,'rotation':rotation,'scale':scale,'wall_seconds':time.perf_counter()-start,'observations':count})
    return merge_texts(observations),{'roi_count':len(views),'views':timings,'skipped_by_ROI_budget':skipped,
                                     'rejected_boundary_observations':rejected,'wall_seconds':time.perf_counter()-begin,'RGB_only':True}
