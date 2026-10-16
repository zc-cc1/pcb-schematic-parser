"""Offline text evaluation only. Raw references never enter PNG inference."""
from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from run_pin_text import ROOT,read,save,sha
from pcb_parser.component_text import merge_texts
from pcb_parser.pin_text import bind_pin_text


def component_matches(detections,truth):
    from pcb_parser.component_text import iou
    if not detections or not truth:
        return {}
    overlap=np.array([[iou(p['bbox'],g['bbox']) for g in truth] for p in detections])
    valid=(overlap>=.5)&np.array([[p['class_name']==g['class_name'] for g in truth] for p in detections])
    reward=valid*(min(len(detections),len(truth))+1)+np.where(valid,overlap,0)
    pi,gi=linear_sum_assignment(-reward)
    return {int(i):truth[j]['component_id'] for i,j in zip(pi,gi) if valid[i,j]}


def match(predicted,truth,owner_map,field=None,identity=False):
    if not predicted or not truth:
        return []
    distance=np.linalg.norm(np.array([p['point'] for p in predicted])[:,None]-np.array([t['point'] for t in truth])[None,:],axis=2)
    eligible=(distance<=5)&np.array([[owner_map.get(p['owner_detection_index'])==t['component_source_id'] for t in truth] for p in predicted])
    if identity:
        eligible &= np.array([[p['owner']==t['designator'] for t in truth] for p in predicted])
    if field:
        eligible &= np.array([[p.get(field) in t['expected_numbers'] if field=='number' else p.get(field)==t['expected_name'] for t in truth] for p in predicted])
    penalty=(min(len(predicted),len(truth))+1)*6
    pi,ti=linear_sum_assignment(np.where(eligible,distance,penalty))
    return [(int(i),int(j)) for i,j in zip(pi,ti) if eligible[i,j]]


def score(tp,emitted,expected):
    return {'correct':tp,'emitted':emitted,'expected':expected,'precision':tp/emitted if emitted else None,
            'recall':tp/expected if expected else None,'f1':2*tp/(emitted+expected) if emitted+expected else None}


def conditional_fields(predicted,truth,pairs):
    counts=Counter()
    for i,j in pairs:
        pin,reference=predicted[i],truth[j]
        for field in ['number','pinname']:
            expected=reference['expected_numbers'] if field=='number' else reference['expected_name']
            if not expected or field=='number' and len(expected)!=1:
                continue
            counts[field+'_eligible']+=1
            if pin.get(field) is not None:
                counts[field+'_emitted']+=1
                counts[field+'_correct']+=pin[field] in expected if field=='number' else pin[field]==expected
    return dict(counts)


def base_predictions(folder,result,cfg,records):
    observations=[]
    for row in read(folder/'texts.json')['texts']:
        readings=row.get('observations',[]) or [row]
        for old in readings:
            view=old.get('view',row.get('view','external'))
            if view.startswith('pin_local_'):
                continue
            x1,y1,x2,y2=old.get('bbox',row['bbox'])
            observations.append({'text':old['text'],'confidence':old['confidence'],'view':view,
                                 'polygon':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]})
    before,_=bind_pin_text(result,merge_texts(observations),cfg,records)
    return before


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-references',action='store_true',help='Replay the exported offline draft references, without loading raw dataset annotations')
    args=parser.parse_args()
    output=ROOT/'runs/s6_pin_text_v1'
    batch=read(output/'predictions/batch_summary.json')
    if args.reuse_references:
        replay=read(output/'draft_field_references.json')
        detector_manifest=replay['detector_references']
    else:
        from pcb_parser.dataset_converter import Converter,load_source_annotation,parse_pin_numbers
        raw_manifest={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
        detector_manifest={r['case_id']:r for r in map(json.loads,(ROOT/'data/detection_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
        policy=read(ROOT/'configs/conversion.json')
        overrides=read(ROOT/'configs/annotation_overrides.json')['cases']
    cfg=read(ROOT/'configs/pin_text_v1.json')
    records=[]
    review=[]
    manual=[]
    all_references=[]
    for row in batch['cases']:
        case=row['case_id']
        folder=output/'predictions'/case
        result=read(folder/'pin_text_result.json')
        assert result['image_sha256']==sha(ROOT/f'data/standardized/{case}/{case}.png')
        if args.reuse_references:
            assert replay['image_sha256'][case]==result['image_sha256']
            truth=[r for r in replay['references'] if r['case_id']==case]
        else:
            raw_path=Path(raw_manifest[case]['raw_annotation_path'])
            assert sha(raw_path)==raw_manifest[case]['annotation_sha256']
            source,_=load_source_annotation(raw_path)
            converter=Converter(source,case,policy,overrides.get(case))
            converter.convert_components()
            points=read(ROOT/f'data/segmentation_unet_v1/{case}/pin_points.json')
            truth=[]
            for point in points:
                cid,pid=point['component_source_id'],point['pin_source_id']
                component=source['components'][cid]
                raw=component.get('pins',{}).get(pid,source['pins'].get(pid,{}))
                correction=overrides.get(case,{}).get('pins',{}).get(cid,{}).get(pid,{})
                number=correction.get('number') or converter.text_ref(raw.get('pinNumberByVLM'),component) or converter.text_ref(raw.get('pinNumberByOCR'),component)
                name=correction.get('pinname') or converter.text_ref(raw.get('pinNameByVLM'),component) or converter.text_ref(raw.get('pinNameByOCR'),component) or ''
                truth.append({**point,'expected_numbers':parse_pin_numbers(number),'expected_name':name,
                              'designator':converter.components.get(cid,{}).get('designator'),
                              'number_source':'existing_image_review_override' if 'number' in correction else 'draft_source_text' if number else 'no_source_number_excluded',
                              'case_id':case})
        all_references.extend(truth)
        detections=[{'bbox':c['bbox_top_left'],'class_name':c['class_name']} for c in result['components']]
        owner_map=component_matches(detections,detector_manifest[case]['objects'])
        predicted=[p for p in result['pin_candidates'] if p['owner_status']=='assigned']
        pairs=match(predicted,truth,owner_map)
        named=[p for p in predicted if p['pinname'] is not None]
        named_truth=[t for t in truth if t['expected_name']]
        name_matches=match(named,named_truth,owner_map,'pinname')
        component_records=read(folder/'component_records.json')['records']
        before=result if row.get('timing',{}).get('local_OCR',{}).get('disabled') else base_predictions(folder,result,cfg,component_records)
        baseline=[p for p in before['pin_candidates'] if p['owner_status']=='assigned']
        base_named=[p for p in baseline if p['pinname'] is not None]
        baseline_names=match(base_named,named_truth,owner_map,'pinname')
        record={'case_id':case,'counts':row['summary'],'localized_owned_pairs':len(pairs),
                'conditional_on_localized_owned':conditional_fields(predicted,truth,pairs),
                'joint_name_and_location_draft':score(len(name_matches),len(named),len(named_truth)),
                'global_OCR_only':{'counts':before['pin_text_summary'],'conditional_on_localized_owned':conditional_fields(baseline,truth,match(baseline,truth,owner_map)),
                                  'joint_name_and_location_draft':score(len(baseline_names),len(base_named),len(named_truth))},
                'source_number_missing':sum(not t['expected_numbers'] for t in truth)}
        records.append(record)
        print(json.dumps({'case':case,'localized_owned_pairs':len(pairs),'named_pin_score':record['joint_name_and_location_draft']},ensure_ascii=False),flush=True)
        manual_truth=[t for t in truth if t['number_source']=='existing_image_review_override']
        if manual_truth:
            owners={t['component_source_id'] for t in manual_truth}
            manual_pred=[p for p in predicted if owner_map.get(p['owner_detection_index']) in owners and p['number'] is not None]
            manual_named=[p for p in named if owner_map.get(p['owner_detection_index']) in owners]
            manual_names=[t for t in manual_truth if t['expected_name']]
            manual.append({'case_id':case,'scope':'Existing visually reviewed pin overrides only; requires component ID, owner, point <=5px and exact text',
                           'number_owner_identity_location':score(len(match(manual_pred,manual_truth,owner_map,'number',True)),len(manual_pred),len(manual_truth)),
                           'name_owner_identity_location':score(len(match(manual_named,manual_names,owner_map,'pinname',True)),len(manual_named),len(manual_names)),
                           'references':manual_truth})
        matched={i:j for i,j in pairs}
        for i,pin in enumerate(predicted):
            reference=truth[matched[i]] if i in matched else None
            reasons=[]
            if pin['number_status']!='observed_OCR_text': reasons.append(pin['number_status'])
            if reference and reference['expected_name'] and pin['pinname']!=reference['expected_name']: reasons.append('draft_name_missing_or_disagrees')
            if pin['pinname'] and reference is None: reasons.append('named_pin_not_matched_by_owner_and_location_review_reference_or_detection')
            if reasons:
                review.append({'case_id':case,'candidate_id':pin['candidate_id'],'point':pin['point'],'owner':pin['owner'],
                               'number':pin['number'],'pinname':pin['pinname'],'reasons':reasons,'draft_reference':reference,
                               'label_change_applied':False})
    total_counts=Counter()
    conditional=Counter()
    before_conditional=Counter()
    for row in records:
        total_counts.update({k:row['counts'].get(k,0) for k in ['candidate_count','owned_candidates','number_observed','name_observed','number_and_name_observed','component_identity_resolved']})
        conditional.update(row['conditional_on_localized_owned'])
        before_conditional.update(row['global_OCR_only']['conditional_on_localized_owned'])
    totals={'counts':dict(total_counts),'conditional_on_localized_owned':dict(conditional),'global_OCR_only_conditional':dict(before_conditional),
            'source_number_missing':sum(r['source_number_missing'] for r in records)}
    for key,container in [('joint_name_and_location_draft',None),('global_OCR_only_joint_name', 'global_OCR_only')]:
        scores=[r['joint_name_and_location_draft'] if container is None else r[container]['joint_name_and_location_draft'] for r in records]
        totals[key]=score(*(sum(s[k] for s in scores) for k in ['correct','emitted','expected']))
    evaluation={'status':'pin_text_v1_draft_evaluation','unique_cases':len(records),'cases':records,'totals':totals,'existing_image_review_subset':manual,
                'config_sha256':sha(ROOT/'configs/pin_text_v1.json'),'official_PinF1':None,'engineering_test_independent':False,'label_changes_applied':0,
                'metric_limits':['Old validation pages already used during development; all results are development observations.',
                                 'Conditional field precision/coverage is only on matched owner+location pins with source text.',
                                 'Null/automatic source numbering is excluded. Multi-number pads are not supported in v1.',
                                 'Joint named-pin F1 uses draft nonempty names only; missing source pins/names can count genuine predictions as unmatched.',
                                 'This is not official PinF1: full numbering, pinname and nets export remain incomplete.']}
    save(output/'evaluation.json',evaluation)
    save(output/'text_review_queue.json',{'items':review,'count':len(review),'label_changes_applied':0})
    if not args.reuse_references:
        save(output/'draft_field_references.json',{'references':all_references,
             'detector_references':{row['case_id']:{'objects':detector_manifest[row['case_id']]['objects']} for row in records},
             'image_sha256':{row['case_id']:read(output/'predictions'/row['case_id']/'pin_text_result.json')['image_sha256'] for row in records},
             'purpose':'Offline evaluation only, excluded from inference and training','labels_modified':0})
    lines=['# 引脚编号与名称 v1 实测','',
           '输入为 PNG、冻结 YOLO/SAHI + U-Net 引脚归属结果，以及 RapidOCR 的图片文字预测。编号/名称分别绑定，不按位置排序编号，不移动引脚点。',
           '本轮 26 张图全部属于已有开发/验证数据，参考多数仍为草稿，没有独立工程测试或正式 PinF1。','',
           f"候选 {total_counts['candidate_count']} 个；已归属 {total_counts['owned_candidates']} 个；恢复编号 {total_counts['number_observed']} 个、名称 {total_counts['name_observed']} 个，同时有编号和名称 {total_counts['number_and_name_observed']} 个。恢复数量不等于正确数量。",'',
           '| 字段（限定已匹配归属和位置，且参考有文字） | 正确/已输出 | 正确/应输出 |','|---|---:|---:|']
    for field,label in [('number','编号'),('pinname','名称')]:
        correct,emitted,eligible=[conditional[field+'_'+k] for k in ['correct','emitted','eligible']]
        lines.append(f'| {label} | {correct}/{emitted} ({correct/emitted:.2%}) | {correct}/{eligible} ({correct/eligible:.2%}) |' if emitted and eligible else f'| {label} | {correct}/{emitted} | {correct}/{eligible} |')
    selected_f1=totals['joint_name_and_location_draft']['f1']
    experiment=output/'local_experiment/evaluation.json'
    local_note=''
    if experiment.exists():
        local_f1=read(experiment)['totals']['joint_name_and_location_draft']['f1']
        local_note=f' 局部复读实验为 {local_f1:.2%}，没有提高该指标，默认关闭；实验输出另存 local_experiment，可用 --enable-local 显式开启。'
    lines+=['',f"原参考中 {totals['source_number_missing']} 个引脚没有可用编号文字，未把自动编号当作真实物理编号评分。上述条件指标不是所有引脚的准确率。",'',
            f"默认的非空名称＋归属＋位置草稿联合 F1 为 {selected_f1:.2%}。该指标不含编号，不能当作正式 PinF1。"+local_note,'',
            '## 已有图像核对的小样本','']
    for row in manual:
        number,name=row['number_owner_identity_location'],row['name_owner_identity_location']
        lines.append(f"{row['case_id']} 页：编号＋位号＋归属＋位置正确 {number['correct']}/{number['expected']}；已输出 {number['emitted']} 个。名称正确 {name['correct']}/{name['expected']}；已输出 {name['emitted']} 个。范围只覆盖已有人工图像核对的引脚覆盖，不代表泛化能力。")
    lines+=['','## 逐页产物','','| 图片 | 编号/名称/两者都有 | 归属＋位置＋名称草稿 F1 | 叠加图 |','|---|---:|---:|---|']
    for row in records:
        c=row['counts']
        f=row['joint_name_and_location_draft']['f1']
        lines.append(f"| {row['case_id']} | {c['number_observed']}/{c['name_observed']}/{c['number_and_name_observed']} | {f:.2%} | [查看](<{(output/'predictions'/row['case_id']/'pin_text_overlay.png').as_posix()}>) |" if f is not None else f"| {row['case_id']} | {c['number_observed']}/{c['name_observed']}/{c['number_and_name_observed']} | 无可评分名称 | [查看](<{(output/'predictions'/row['case_id']/'pin_text_overlay.png').as_posix()}>) |")
    lines+=['',f'复核队列 {len(review)} 条，保留文字歧义、编号缺失和参考不一致建议，没有修改训练标注。','',
            '下一步：完善编号/名称复核，处理多编号 pad 与低覆盖符号，再从 U-Net 导线概率建立连通拓扑。交叉连接、跨线不连接、同名网络标签和正式 nets/edges 输出尚未实现。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'totals':totals,'manual_review':[{'case_id':r['case_id'],'number':r['number_owner_identity_location'],'name':r['name_owner_identity_location']} for r in manual],'review_items':len(review)},ensure_ascii=False))


if __name__=='__main__':
    main()
