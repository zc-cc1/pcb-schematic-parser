"""Prepare review crops, with no generated class labels or training changes."""
import hashlib
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from run_wire_graph import ROOT,read,save,sha


def main():
    output=ROOT/'data/topology_review_v1';output.mkdir(parents=True,exist_ok=True)
    queue=read(ROOT/'runs/s7_wire_graph_v1/review_queue.json')
    records=[]
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',13)
    for item in queue['items']:
        if item['kind'] not in {'junction','gap'}:continue
        case=item['case_id'];source=ROOT/f'data/standardized/{case}/{case}.png'
        with Image.open(source) as image:
            image=image.convert('RGB')
            center=item['point'] if item['kind']=='junction' else [(item['points'][0][i]+item['points'][1][i])/2 for i in range(2)]
            radius=48
            box=[max(0,round(center[0])-radius),max(0,round(center[1])-radius),min(image.width,round(center[0])+radius+1),min(image.height,round(center[1])+radius+1)]
            original=image.crop(box)
        name=f"{case}_{item['kind']}_{sum(r['case_id']==case and r['kind']==item['kind'] for r in records)+1:03d}"
        path=output/'crops'/f'{name}.png';path.parent.mkdir(exist_ok=True)
        original.save(path)
        enlarged=original.resize((192,192),Image.Resampling.NEAREST)
        draw=ImageDraw.Draw(enlarged)
        px=(center[0]-box[0])*192/original.width;py=(center[1]-box[1])*192/original.height
        draw.ellipse((px-8,py-8,px+8,py+8),outline='#ee7800',width=2)
        card=Image.new('RGB',(220,228),'white');card.paste(enlarged,(14,24))
        ImageDraw.Draw(card).text((5,3),name,font=font,fill='#333333')
        card.save(output/'crops'/f'{name}_preview.png')
        classes=['connected_junction','crossing_without_connection','segmentation_artifact','uncertain'] if item['kind']=='junction' else ['true_wire_break','should_remain_separate','segmentation_artifact','uncertain']
        records.append({'review_id':name,'case_id':case,'kind':item['kind'],'source_image_sha256':sha(source),
                        'center_top_left':center,'crop_top_left':box,'crop_file':f'crops/{name}.png','crop_sha256':sha(path),
                        'candidate':item,'allowed_labels':classes,'label':None,'reviewed':False,
                        'source_group':case,'reference_status':'model_candidate_from_existing_development_pages_not_new_external_data'})
    for kind in ['junction','gap']:
        selected=[r for r in records if r['kind']==kind]
        for offset in range(0,len(selected),24):
            page=selected[offset:offset+24]
            sheet=Image.new('RGB',(6*220,((len(page)+5)//6)*228),'white')
            for index,row in enumerate(page):
                with Image.open(output/'crops'/f"{row['review_id']}_preview.png") as tile:sheet.paste(tile,((index%6)*220,(index//6)*228))
            sheet.save(output/f'{kind}_sheet_{offset//24+1:02d}.png')
    save(output/'annotations.json',{'records':records,'total':len(records),'reviewed_count':0,'labels_generated_automatically':0,
         'training_dataset_modified':False,'source_scope':'Existing 26 development validation pages; not eligible as newly collected independent data',
         'split_rule':'If trained later, split by original schematic/source group, not by crop; old validation scores cease to be independent'})
    print(json.dumps({'junction_crops':sum(r['kind']=='junction' for r in records),'gap_crops':sum(r['kind']=='gap' for r in records),'labels':0,'output':str(output)},ensure_ascii=False))


if __name__=='__main__':main()
