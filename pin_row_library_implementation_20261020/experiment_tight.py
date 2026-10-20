"""Corroborate row fields with whole grouped glyph crops at fixed scales."""
from pathlib import Path
import runpy,statistics
scope=runpy.run_path(str(Path(__file__).with_name('experiment.py')))
globals().update({k:v for k,v in scope.items() if not k.startswith('__')})
result=[]
for group in groups:
    for entry in group['rows']:
        lead=entry['lead'];owner=group['owner'];row=lead['row'];edge=lead['edge'];left=lead['side']=='left'
        if owner=='U1' and entry['name']!='VDD':continue
        if owner=='USB1' and (entry['number'] is not None and entry['name'] is not None):continue
        if owner=='USB2' and entry['name'] is not None:continue
        for field,source in [('number',entry['numbers']),('name',entry['names'])]:
            if field=='number' and owner=='USB2':continue
            if field=='name' and entry['name'] is not None:continue
            if not source:continue
            selected=[t for t in source if t.get('confidence',0)>=.75]
            bounds=[statistics.median(t['bbox'][i] for t in selected) for i in range(4)]
            box=[math.floor(bounds[0]-2),math.floor(bounds[1]-2),math.ceil(bounds[2]+2),math.ceil(bounds[3]+2)]
            patch=rgb[box[1]:box[3],box[0]:box[2]]
            for mode in ['raw','neutral']:
                isolated=patch.copy()
                if mode=='neutral':
                    v=patch.astype(np.int16);mask=(v.max(2)-v.min(2)<35)&(v.mean(2)<230)
                    isolated[:]=255;isolated[mask]=patch[mask]
                isolated=cv2.copyMakeBorder(isolated,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255));tokens=[]
                for scale in [3.,5.]:tokens.extend(reader(isolated,[0,0,isolated.shape[1],isolated.shape[0]],scale,0,True,f'row_tight_{owner}_{lead["side"]}_{row}_{field}_{mode}_s{scale}'))
                value,support=vote(tokens,DEFAULT)
                result.append({'owner':owner,'lead':lead,'field':field,'mode':mode,'value':value,'support':support,'readings':tokens,'crop':box})
                Image.fromarray(isolated).save(ROOT/f'{owner}_{row}_{field}_{mode}.png')
(ROOT/'experiment_tight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('TIGHT',json.dumps(result,ensure_ascii=True))
