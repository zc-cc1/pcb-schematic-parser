"""Use one interior name column rather than a whole component detector box."""
from pathlib import Path
import runpy
scope=runpy.run_path(str(Path(__file__).with_name('experiment.py')))
globals().update({k:v for k,v in scope.items() if not k.startswith('__')})
names=[]
for row in rows:
    lead=row['lead'];x=lead['point'][0];side=lead['side'];mid=(frame[1]+frame[3])/2
    box=[math.floor(x-14),math.floor(frame[1]+2 if side=='top' else mid-2),
         math.ceil(x+14),math.ceil(mid+2 if side=='top' else frame[3]+2)]
    patch=rgb[box[1]:box[3],box[0]:box[2]];v=patch.astype(np.int16)
    mask=(v.max(2)-v.min(2)<35)&(v.mean(2)<230);isolated=np.full_like(patch,255);isolated[mask]=patch[mask]
    isolated=cv2.copyMakeBorder(isolated,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
    isolated=cv2.rotate(isolated,cv2.ROTATE_90_CLOCKWISE);tokens=[]
    for scale in [3.,5.]:tokens.extend(reader(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f'vertical_name_{owner}_{side}_{x}_s{scale}'))
    value,support=vote(tokens,DEFAULT);names.append({'point':lead['point'],'name':value,'support':support,'readings':tokens,'crop':box})
    Image.fromarray(isolated).save(ROOT/f'name_{side}_{x}.png')
(ROOT/'experiment_names.json').write_text(json.dumps(names,ensure_ascii=False,indent=2),encoding='utf-8');print('NAMES',json.dumps(names,ensure_ascii=False))
