"""Keep every numeral component on one lead row, including leading digits."""
import runpy
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
scope=runpy.run_path(str(Path(__file__).with_name('experiment.py')))
globals().update({k:v for k,v in scope.items() if not k.startswith('__')})
rows=[]
for entry in results:
    if entry['name']!='NC' and entry['number']!='6':continue
    lead=entry['lead'];row=lead['row'];edge=lead['edge']
    box=[math.floor(edge-32),math.floor(row-15),math.ceil(edge+2),math.ceil(row+2)]
    patch=rgb[box[1]:box[3],box[0]:box[2]];v=patch.astype(np.int16)
    mask=((v.max(axis=2)-v.min(axis=2)<35)&(v.mean(axis=2)<205)).astype(np.uint8)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    accepted=[]
    for i in range(1,count):
        x,y,cw,ch,area=map(int,stats[i]);cx,cy=centroids[i]
        if 1<=cw<=24 and 2<=ch<=17 and area>=2 and -14<=box[1]+cy-row<=0:
            accepted.append(i)
    if not accepted:continue
    ys,xs=np.where(np.isin(labels,accepted));a,b,c,d=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
    isolated=np.full_like(patch,255);chosen=np.isin(labels,accepted);isolated[chosen]=patch[chosen]
    isolated=cv2.copyMakeBorder(isolated[b:d,a:c],8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
    tokens=[]
    for scale in [3.,5.]:tokens.extend(ocr(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,True,f'nc_grouped_{row}_s{scale}'))
    number,support=vote([t for t in tokens if t['text'].isdigit()],DEFAULT)
    Image.fromarray(isolated).save(ROOT/f'grouped_row_{row}.png')
    rows.append({'lead':lead,'number':number,'support':support,'readings':tokens,'components':[stats[i].tolist() for i in accepted],'crop':box,'glyph_bbox':[box[0]+a,box[1]+b,box[0]+c,box[1]+d]})
(ROOT/'experiment_grouped.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('GROUPED',json.dumps(rows,ensure_ascii=False))
