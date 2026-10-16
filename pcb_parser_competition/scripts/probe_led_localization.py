"""Development probe: local detector predictions; references used only in metrics."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'artifacts/ultralytics'))
os.environ['YOLO_AUTOINSTALL'] = 'false'
import numpy as np
from PIL import Image
from ultralytics import YOLO


def main():
    audit = json.loads((ROOT/'dist/official_public_dataset_audit_20261016/dataset_audit.json').read_text(encoding='utf-8'))
    public = Path(audit['source_root'])
    geometry = json.loads((ROOT/'dist/official_public_dataset_audit_20261016/geometry_diagnostic.json').read_text(encoding='utf-8'))
    rows = [row for row in geometry['pairs'] if row['type']=='led']
    model = YOLO(str(ROOT/'weights/detector_dataset300_v3_nano.pt'))
    output=[];start=time.perf_counter()
    for row in rows:
        rgb=np.array(Image.open(next((public/'200_train_cases'/row['case']).glob('*.png'))).convert('RGB'))
        height,width=rgb.shape[:2];b=row['raw_bbox'];cx=(b[0]+b[2])/2;cy=height-(b[1]+b[3])/2
        views=[]
        for side in [384,640]:
            x=max(0,min(width-side,round(cx-side/2)));y=max(0,min(height-side,round(cy-side/2)))
            crop=rgb[y:min(height,y+side),x:min(width,x+side)]
            pred=model.predict(np.ascontiguousarray(crop[:,:,::-1]),imgsz=640,conf=.3,iou=.5,device='cuda:0',verbose=False,save=False)[0]
            candidates=[]
            for box,conf,cls in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist()):
                if pred.names[int(cls)]!='led':continue
                q=[box[0]+x,box[1]+y,box[2]+x,box[3]+y]
                pc=[(q[0]+q[2])/2,height-(q[1]+q[3])/2]
                candidates.append({'bbox_top_left':q,'confidence':conf,'center_bottom_left':pc,'distance':float(np.linalg.norm(np.array(pc)-[(b[0]+b[2])/2,(b[1]+b[3])/2]))})
            candidates.sort(key=lambda p:p['distance'])
            views.append({'side':side,'candidates':candidates[:3]})
        output.append({**row,'views':views})
    out=ROOT/'dist/led_bbox_fix_20261016/local_probe.json'
    out.write_text(json.dumps({'seconds':time.perf_counter()-start,'rows':output},indent=2),encoding='utf-8')
    old_errors=[];new_errors=[];changes=[]
    for r in output:
        old_errors.append(r['proposed_max_error_px']);cs=[v['candidates'][0] for v in r['views'] if v['candidates']]
        if len(cs)==2 and min(c['confidence'] for c in cs)>=.7 and np.linalg.norm(np.array(cs[0]['center_bottom_left'])-cs[1]['center_bottom_left'])<3:
            center=np.mean([c['center_bottom_left'] for c in cs],axis=0);old=r['proposed_bbox'];wh=np.array([old[2]-old[0],old[3]-old[1]])
            q=[*(center-wh/2),*(center+wh/2)]
        else:q=r['proposed_bbox']
        error=max(abs(a-b) for a,b in zip(q,r['gt_bbox']));new_errors.append(error)
        changes.append((r['case'],r['gt_key'],round(r['proposed_max_error_px'],2),round(error,2)))
    print(json.dumps({'count':len(rows),'seconds':time.perf_counter()-start,'old_le5':int(sum(e<=5 for e in old_errors)),'new_le5':int(sum(e<=5 for e in new_errors)),'old_mean':float(np.mean(old_errors)),'new_mean':float(np.mean(new_errors)),'rows':changes}))


if __name__=='__main__':main()
