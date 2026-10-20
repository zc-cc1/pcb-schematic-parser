"""Tightly group complete neutral text, discarding adjacent row fragments."""
from pathlib import Path
import runpy
scope=runpy.run_path(str(Path(__file__).with_name('experiment.py')))
globals().update({k:v for k,v in scope.items() if not k.startswith('__')})
tight=read(ROOT/'experiment_tight.json');out=[]
for entry in tight:
    if entry['mode']!='neutral' or (entry['owner'],entry['field']) not in [('USB1','number'),('USB2','name')]:continue
    if entry['owner']=='USB1' and entry['lead']['side']!='left':continue
    box=entry['crop'];patch=rgb[box[1]:box[3],box[0]:box[2]];v=patch.astype(np.int16)
    mask=((v.max(2)-v.min(2)<35)&(v.mean(2)<230)).astype(np.uint8)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8);accepted=[]
    for i in range(1,n):
        x,y,cw,ch,area=map(int,stats[i]);cx,cy=centroids[i]
        if 2<=ch<=22 and area>=3 and abs(box[1]+cy-entry['lead']['row'])<=10:accepted.append(i)
    chosen=np.isin(labels,accepted);ys,xs=np.where(chosen);a,b,c,d=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
    isolated=np.full_like(patch,255);isolated[chosen]=patch[chosen]
    for pad in [4,8]:
        prepared=cv2.copyMakeBorder(isolated[b:d,a:c],pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(255,255,255));tokens=[]
        for scale in [3.,5.]:tokens.extend(reader(prepared,[0,0,prepared.shape[1],prepared.shape[0]],scale,0,True,f'row_grouped_{entry["owner"]}_{entry["field"]}_p{pad}_s{scale}'))
        value,support=vote(tokens,DEFAULT)
        out.append({'owner':entry['owner'],'field':entry['field'],'value':value,'support':support,'readings':tokens,'crop':box,'glyph_bbox':[box[0]+a,box[1]+b,box[0]+c,box[1]+d],'pad':pad})
        Image.fromarray(prepared).save(ROOT/f'group_{entry["owner"]}_{entry["field"]}_p{pad}.png')
(ROOT/'experiment_grouped.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('GROUPED',json.dumps(out,ensure_ascii=True))
