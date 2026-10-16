"""Offline source-label audit and unapproved training-only review crops.

This script can read source annotations. The PNG inference entry never imports
it, and the generated review queue is not a training target until verified.
"""
from collections import Counter,defaultdict
import hashlib
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from run_pin_text_v2 import ROOT,read,save,sha
from pcb_parser.dataset_converter import load_source_annotation,Converter,parse_pin_numbers,raw_point,natural_key


def main():
    output=ROOT/'data/passive_number_review_v1';output.mkdir(parents=True,exist_ok=True)
    rows=[__import__('json').loads(s) for s in (ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    splits={r['case_id']:r['split'] for r in map(__import__('json').loads,(ROOT/'data/detection_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    policy=read(ROOT/'configs/conversion.json');counts=Counter();direction=defaultdict(Counter);pool=[];hashes=[]
    for row in rows:
        path=Path(row['raw_annotation_path']);digest=sha(path)
        if digest!=row['annotation_sha256']:raise ValueError('Source annotation changed: '+row['case_id'])
        source,_=load_source_annotation(path);converter=Converter(source,row['case_id'],policy)
        hashes.append({'case_id':row['case_id'],'source_annotation_sha256':digest})
        for cid,c in source['components'].items():
            if c.get('state')=='deleted' or c.get('category') not in {'R','C'}:continue
            pins=[]
            for pid,local in c.get('pins',{}).items():
                if local.get('state')=='deleted':continue
                raw={**source.get('pins',{}).get(pid,{}),**local};point=raw_point(raw.get('tip'))
                if point is None:continue
                number=converter.text_ref(raw.get('pinNumberByVLM'),c) or converter.text_ref(raw.get('pinNumberByOCR'),c)
                pins.append({'source_pin_id':pid,'point':list(point),'explicit_numbers':parse_pin_numbers(number)})
            counts['components']+=1
            if len(pins)!=2:counts['not_two_valid_tips']+=1;continue
            counts['two_tip_components']+=1
            explicit={tuple(p['explicit_numbers']) for p in pins}=={('1',),('2',)}
            counts['explicit_number_pair' if explicit else 'missing_or_noncanonical_pair']+=1
            if explicit:counts['explicit_pair_'+splits.get(row['case_id'],'excluded')]+=1
            pins.sort(key=lambda p:natural_key(p['source_pin_id']));a,b=pins
            ori='horizontal' if abs(a['point'][0]-b['point'][0])>abs(a['point'][1]-b['point'][1]) else 'vertical'
            axis=0 if ori=='horizontal' else 1;kind='explicit' if explicit else 'source_ID_auto'
            one=next((p for p in pins if p['explicit_numbers']==['1']),a) if explicit else a
            other=b if one is a else a
            direction[(c['category'],ori,kind)]['pin1_left_or_top' if one['point'][axis]<other['point'][axis] else 'pin1_right_or_bottom']+=1
            location=c.get('location',{});bbox=[location.get(k) for k in ['x1','y1','x2','y2']]
            if (not explicit and splits.get(row['case_id'])=='train' and all(isinstance(v,(int,float)) for v in bbox)
                    and bbox[2]>bbox[0] and bbox[3]>bbox[1]):
                pool.append({'case_id':row['case_id'],'source_component_id':cid,'category':c['category'],'orientation':ori,
                             'bbox_top_left':bbox,'source_annotation_sha256':digest,'split':'train','pins':pins})
    # Balance R/C and orientations, at most one of each group per source page.
    # Order is fixed by PNG digest; no validation outcomes influence selection.
    by_group=defaultdict(list)
    for item in pool:by_group[(item['category'],item['orientation'])].append(item)
    selected=[]
    for group,items in sorted(by_group.items()):
        used=set()
        for item in sorted(items,key=lambda r:(hashlib.sha256((r['case_id']+r['source_component_id']).encode()).hexdigest())):
            if item['case_id'] in used:continue
            used.add(item['case_id']);selected.append(item)
            if len(used)==10:break
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
    for index,item in enumerate(selected,1):
        image=ROOT/f"data/standardized/{item['case_id']}/{item['case_id']}.png"
        with Image.open(image) as im:
            rgb=im.convert('RGB');bbox=item['bbox_top_left'];points=[p['point'] for p in item['pins']]
            xs=[bbox[0],bbox[2]]+[p[0] for p in points];ys=[bbox[1],bbox[3]]+[p[1] for p in points]
            crop=[max(0,int(min(xs))-35),max(0,int(min(ys))-35),min(rgb.width,int(max(xs))+36),min(rgb.height,int(max(ys))+36)]
            patch=rgb.crop(crop);draw=ImageDraw.Draw(patch)
            for label,pin in zip(['A','B'],item['pins']):
                x,y=pin['point'][0]-crop[0],pin['point'][1]-crop[1]
                draw.ellipse((x-4,y-4,x+4,y+4),outline='#bf2222',width=1)
                draw.text((x+5,y+3),label,fill='#bf2222',font=font,stroke_width=1,stroke_fill='white')
                pin.update(review_number=None,number_visible_in_PNG=None,tip_verified=False,number_verified=False)
            scale=min(4,650/patch.width,380/patch.height);patch=patch.resize((round(patch.width*scale),round(patch.height*scale)),Image.Resampling.NEAREST)
            card=Image.new('RGB',(max(700,patch.width+20),patch.height+100),'white');card.paste(patch,(10,40));draw=ImageDraw.Draw(card)
            draw.text((10,8),f"{index:02d} | {item['case_id']} {item['source_component_id']} | {item['category']} {item['orientation']}",font=font,fill='#333')
            draw.text((10,patch.height+50),'A/B are source IDs, not pin 1/2. Verify tip locations first.',font=font,fill='#933')
            draw.text((10,patch.height+70),'If no number is visible or convention is unknown, leave review_number null.',font=font,fill='#333')
            name=f"review_{index:02d}_{item['case_id']}_{item['source_component_id']}.png";card.save(output/name)
            item.update(image_sha256=sha(image),crop_top_left=crop,review_card=name,review_status='unapproved',approved_for_training=False,
                        canonical_number_evidence=None,source_ID_auto_numbers_used_as_targets=False)
    audit={'status':'no_certified_passive_number_training_targets','counts':dict(counts),
           'source_ID_direction_counts':[{'category':k[0],'orientation':k[1],'number_source':k[2],**dict(v)} for k,v in sorted(direction.items())],
           'source_annotations':hashes,'source_annotations_modified':False,'automatic_ID_numbers_used_for_training':False,
           'explicit_source_number_is_visually_verified':False,'review_examples':len(selected),
           'review_splits':{'train':len(selected)},'approved_number_training_examples':0,
           'notes':'Audit raw pinNumberByOCR/VLM fields. Existing separately reviewed overrides are not counted as raw explicit pairs. Source-generated IDs are not visually observable.'}
    save(output/'audit.json',audit);save(output/'review_queue.json',{'scope':'offline training-only review; never inference input',
          'examples':selected,'approved_for_training':0,'automatic_number_training_disabled':True})
    print({'counts':dict(counts),'training_review_crops':len(selected),'approved_training_targets':0},flush=True)


if __name__=='__main__':main()
