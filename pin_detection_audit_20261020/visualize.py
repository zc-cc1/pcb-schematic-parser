"""Export a diagnostic figure of unresolved geometry, without changing images."""
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
audit=read(ROOT/'audit.json')
cases={c['case']:c for c in audit['per_case']}
rows=[r for r in audit['misses'] if r['category']=='candidate_offset_or_crowded_assignment']
fig,axes=plt.subplots(2,4,figsize=(15,8))
details=[]
for ax,row in zip(axes.flat,rows):
    case=row['case']
    source=cv2.imdecode(np.fromfile(f'D:/pcb_open_gt_in/{case}.png',np.uint8),cv2.IMREAD_COLOR)
    rgb=cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
    height,width=rgb.shape[:2]
    gx,gy=row['gt_point']['x'],height-row['gt_point']['y']
    x0,x1=max(0,int(gx)-65),min(width,int(gx)+65)
    y0,y1=max(0,int(gy)-55),min(height,int(gy)+55)
    ax.imshow(rgb[y0:y1,x0:x1],extent=(x0,x1,y1,y0))
    model=read(Path(cases[case]['model_cache'])/'p/export/engineering_pin_text_result.json')
    secondaries=[]
    for pin in model['pin_candidates']:
        x,y=pin['point']
        d=math.hypot(x-gx,y-gy)
        secondary=bool(pin.get('text_hypothesis_primary_candidate_id'))
        if secondary and d<=25:
            secondaries.append({'id':pin['candidate_id'],'owner':pin.get('owner'),'distance_raw_px':round(d,2),
                                'primary':pin.get('text_hypothesis_primary_candidate_id')})
        if not(x0<=x<=x1 and y0<=y<=y1):continue
        color='#ef8900' if secondary else '#1368c4'
        ax.scatter([x],[y],s=70,facecolors='none',edgecolors=color,marker='s' if secondary else 'o',linewidths=1.5)
        ax.text(x+2,y-2,pin['candidate_id'][-4:],fontsize=7,color=color,
                bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':0})
    ax.scatter([gx],[gy],s=110,c='#d00000',marker='x',linewidths=2)
    ax.set_title(f"case {case} / {row['component']} / {row['key']} / {row['name']}",fontsize=10)
    ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xticks([]);ax.set_yticks([])
    details.append({'case':case,'component':row['component'],'key':row['key'],
                    'nearby_secondary_candidates':secondaries})
for ax in axes.flat[len(rows):]:ax.axis('off')
fig.suptitle('Red x: reference pin | Blue circle: raw primary candidate | Orange square: suppressed secondary',fontsize=12)
fig.tight_layout(rect=(0,0,1,.96))
fig.savefig(ROOT/'geometry_misses.png',dpi=160)
(ROOT/'secondary_checks.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(details,ensure_ascii=False))
