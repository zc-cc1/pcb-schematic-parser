"""Read the observed text-token bounds, rather than a padded half-column."""
from pathlib import Path
import runpy,statistics
scope=runpy.run_path(str(Path(__file__).with_name('experiment.py')))
globals().update({k:v for k,v in scope.items() if not k.startswith('__')})
result=[]
for row in rows:
    lead=row['lead'];x=lead['point'][0];side=lead['side'];mid=(frame[1]+frame[3])/2
    tokens=[]
    for p in document['pin_candidates']:
        if p.get('owner')!=owner:continue
        for t in p.get('text_candidates',{}).get('pinname',[]):
            cx=(t['bbox'][0]+t['bbox'][2])/2;cy=(t['bbox'][1]+t['bbox'][3])/2
            if t.get('confidence',0)>=.98 and abs(cx-x)<12 and frame[1]<cy<frame[3] and ((cy<mid) if side=='top' else (cy>mid)):
                tokens.append({**t,'view':t['source_view']})
    name,support=vote(tokens,DEFAULT)
    if not name:continue
    selected=[t for t in tokens if t['text']==name]
    bounds=[statistics.median(t['bbox'][i] for t in selected) for i in range(4)]
    box=[math.floor(bounds[0]-2),math.floor(bounds[1]-2),math.ceil(bounds[2]+2),math.ceil(bounds[3]+2)]
    patch=rgb[box[1]:box[3],box[0]:box[2]]
    for mode in ['raw','neutral']:
        isolated=patch.copy()
        if mode=='neutral':
            v=patch.astype(np.int16);mask=(v.max(2)-v.min(2)<35)&(v.mean(2)<230)
            isolated[:]=255;isolated[mask]=patch[mask]
        isolated=cv2.copyMakeBorder(isolated,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
        isolated=cv2.rotate(isolated,cv2.ROTATE_90_CLOCKWISE);fresh=[]
        for scale in [3.,5.]:fresh.extend(reader(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f'vertical_tight_{owner}_{side}_{x}_{mode}_s{scale}'))
        value,confirmed=vote(fresh,DEFAULT)
        result.append({'point':lead['point'],'source_name':name,'name':value,'support':confirmed,'mode':mode,'readings':fresh,'crop':box})
(ROOT/'experiment_tight_names.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print('TIGHT',json.dumps(result,ensure_ascii=False))
