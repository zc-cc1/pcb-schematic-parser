import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
item=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='4')
stage=Path(item['model_cache'])/'p'
model=read(stage/'export/engineering_pin_text_result.json');graph=read(stage/'export/engineering_graph_proposals.json')
pred=read(WORK/'pin_relocation_implementation_20261020/deployed/4_Pred.json')
record=next(r for r in read(stage/'export/completed_component_records.json')['records'] if r['component_id']=='U2')
print('STAGE',str(stage));print('SIZE',graph['size']);print('RECORD',json.dumps(record,ensure_ascii=False))
for p in model['pin_candidates']:
    if p.get('owner')!='U2':continue
    fields={k:v for k,v in p.items() if k not in {'text_candidates','text_evidence','owner_candidates'}}
    fields['text_candidates']={k:[{a:b for a,b in t.items() if a not in {'alternatives','polygon'}} for t in v] for k,v in p.get('text_candidates',{}).items()}
    print('CANDIDATE',json.dumps(fields,ensure_ascii=False))
    print('ATTACHMENT',json.dumps([a for a in graph['pin_attachments'] if a['candidate_id']==p['candidate_id']],ensure_ascii=False))
print('NETS',json.dumps({n:v for n,v in pred['nets'].items() if 'U2.' in v['hyperGraph']},ensure_ascii=False))
ocr=read('D:/pcb_open_gt_v37dots_b/logs/4_ocr_texts.json');a,b,c,d=record['bbox_top_left']
print('OCR',json.dumps([t for t in ocr['texts'] if a-20<(t['bbox'][0]+t['bbox'][2])/2<c+20 and b-10<(t['bbox'][1]+t['bbox'][3])/2<d+10],ensure_ascii=False))
