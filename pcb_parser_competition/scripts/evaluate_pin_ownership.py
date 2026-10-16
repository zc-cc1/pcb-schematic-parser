"""Offline evaluation of pin ownership against draft reference annotations.

Inference artifacts are copied from frozen image-model predictions. References
are used only here for evaluation/review, never by run_pin_ownership.py.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.pin_ownership import normalized_components,associate_pins
from pcb_parser.pin_ownership_visual import draw_ownership
from evaluate_trained_unet import match_points,metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    total=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/total if total else 0.


def component_matches(detections,truth):
    result={}
    if not detections or not truth:
        return result
    overlap=np.array([[box_iou(p['bbox'],g['bbox']) for g in truth] for p in detections])
    valid=(overlap>=.5)&np.array([[p['class_name']==g['class_name'] for g in truth] for p in detections])
    reward=valid*(min(len(detections),len(truth))+1)+np.where(valid,overlap,0)
    pi,gi=linear_sum_assignment(-reward)
    for i,j in zip(pi,gi):
        if valid[i,j]:
            result[int(i)]=truth[j]['component_id']
    return result


def joint_match(assigned,truth,owner_map):
    if not assigned or not truth:
        return metrics(0,len(assigned),len(truth))
    p=np.array([item['point'] for item in assigned])
    g=np.array([item['point'] for item in truth])
    distance=np.linalg.norm(p[:,None]-g[None],axis=2)
    eligible=(distance<=5)&np.array([[owner_map.get(item['owner_detection_index'])==tip['component_source_id'] for tip in truth] for item in assigned])
    penalty=(min(len(assigned),len(truth))+1)*6
    pi,gi=linear_sum_assignment(np.where(eligible,distance,penalty))
    tp=int(eligible[pi,gi].sum())
    return metrics(tp,len(assigned)-tp,len(truth)-tp)


def main():
    output=ROOT/'runs/s5_pin_ownership_v1'
    split=read(ROOT/'runs/unet_inference_v2/split.json')
    unet_config=read(ROOT/'configs/unet_inference_v2.json')
    cfg=read(ROOT/'configs/pin_ownership_v1.json')
    detector_cfg=read(ROOT/'configs/baseline_v3_adopted.json')
    frozen=read(ROOT/'runs/detector_comparison_v3/adopted_fused_cpu.json')
    detector_manifest=ROOT/'data/detection_v1/manifest.jsonl'
    assert frozen['manifest_sha256']==sha(detector_manifest)
    assert frozen['weights_sha256']==sha(ROOT/detector_cfg['weights'])
    detections_by_case={r['case_id']:r['predictions'] for r in frozen['cases']}
    reference={r['case_id']:r for r in map(json.loads,detector_manifest.read_text(encoding='utf-8').splitlines())}
    status_counts=Counter()
    cases=[]
    review_queue=[]
    for row in split['rows']:
        case=row['case_id']
        image=Path(row['image'])
        assert sha(image)==row['image_sha256']
        detector_image=Path(reference[case]['image'])
        assert sha(detector_image)==reference[case]['image_sha256']
        # Source JPEG/TIFF and standardized PNG can have different byte hashes.
        # The frozen detector and U-Net must nevertheless see identical RGB pixels.
        with Image.open(image) as im:
            standard_rgb=np.asarray(im.convert('RGB'))
        with Image.open(detector_image) as im:
            detector_rgb=np.asarray(im.convert('RGB'))
        assert np.array_equal(standard_rgb,detector_rgb), f'Detector and U-Net pixels differ: {case}'
        detections=detections_by_case[case]
        candidates=read(ROOT/f'runs/unet_inference_v2/predictions/{case}/pin_candidates.json')
        probabilities=np.load(ROOT/f'runs/unet_inference_v2/cache/{case}/scale_1.0.npz')['probability']
        components=normalized_components(detections,detector_cfg['class_mapping'])
        predicted=associate_pins(candidates,components,cfg,[row['width'],row['height']],probabilities[0])
        predicted['image_sha256']=row['image_sha256']
        predicted['association_config_sha256']=sha(ROOT/'configs/pin_ownership_v1.json')
        folder=output/case
        save(folder/'pin_ownership.json',predicted)
        save(folder/'detector_predictions.json',{'image_sha256':row['image_sha256'],'size':[row['width'],row['height']],
             'coordinate_origin':'top_left','weights_sha256':frozen['weights_sha256'],'detections':detections,
             'source':'frozen_PNG_detector_predictions_no_reference_fields'})
        save(folder/'unet_predictions.json',{'image_sha256':row['image_sha256'],'size':[row['width'],row['height']],
             'coordinate_origin':'top_left','weights_sha256':unet_config['checkpoint_sha256'],
             'inference_config_sha256':sha(ROOT/'configs/unet_inference_v2.json'),'pin_candidates':candidates,
             'source':'frozen_PNG_UNet_v2_predictions_no_reference_fields'})
        status_counts.update(predicted['summary']['status_counts'])
        with Image.open(image) as source:
            rgb=np.asarray(source.convert('RGB'))
        draw_ownership(rgb,predicted,folder/'ownership_overlay.png')
        truth=read(Path(row['wire_mask']).parent/'pin_points.json')
        owner_map=component_matches(detections,reference[case]['objects'])
        raw_matches,raw=match_points([p['point'] for p in candidates],[g['point'] for g in truth])
        assigned=[p for p in predicted['pin_candidates'] if p['owner_status']=='assigned']
        _,assigned_position=match_points([p['point'] for p in assigned],[g['point'] for g in truth])
        joint=joint_match(assigned,truth,owner_map)
        owner_diagnostics=Counter()
        covered_owners=set(owner_map.values())
        for i,j,_ in raw_matches:
            source_owner=truth[j]['component_source_id']
            pin=predicted['pin_candidates'][i]
            if source_owner not in covered_owners:
                reason='reference_owner_not_matched_by_detector'
            elif pin['owner_status']!='assigned':
                reason='localized_but_owner_unresolved'
            elif owner_map.get(pin['owner_detection_index'])!=source_owner:
                reason='localized_but_assigned_owner_disagrees'
            else:
                reason='localized_and_owner_agrees'
            owner_diagnostics[reason]+=1
        record={'case_id':case,'subset':'development' if case in split['development'] else 'acceptance_holdout',
                'candidate_count':len(candidates),'assigned_count':len(assigned),'status_counts':predicted['summary']['status_counts'],
                'raw_localization':raw,'assigned_localization':assigned_position,'joint_owner_and_localization':joint,
                'matched_components':len(owner_map),'reference_components':len(reference[case]['objects']),
                'coordinate_match_owner_diagnostics':dict(owner_diagnostics)}
        cases.append(record)
        matched_ids={i for i,_,_ in raw_matches}
        truth_points=np.array([t['point'] for t in truth])
        for index,pin in enumerate(predicted['pin_candidates']):
            if index in matched_ids:
                continue
            distances=np.linalg.norm(truth_points-np.array(pin['point']),axis=1)
            nearest=int(distances.argmin())
            reason='tip_definition_or_offset_candidate' if distances[nearest]<=15 else 'missing_tip_annotation_or_spurious_prediction_needs_review'
            review_queue.append({'case_id':case,'candidate_id':pin['candidate_id'],'point':pin['point'],
                                 'owner':pin['owner'],'owner_status':pin['owner_status'],
                                 'nearest_source_tip':truth[nearest],'nearest_distance_px':float(distances[nearest]),
                                 'review_reason':reason,'label_change_applied':False})
        print(json.dumps({'case':case,'assigned':len(assigned),'candidate_count':len(candidates),'joint':joint}),flush=True)
    result={'status':'pin_ownership_prototype_evaluated_on_draft_references','config':cfg,
            'config_sha256':sha(ROOT/'configs/pin_ownership_v1.json'),'cases':cases,'candidate_status_counts':dict(status_counts),
            'engineering_test_independent':False,'official_score':None,
            'evaluation_scope':'Predicted components + learned pins; owner component match uses class + bbox IoU>=0.5, pin match <=5px. Number/name/nets not scored.',
            'same_pin_and_detector_models':True,'label_changes_applied':0}
    diagnostic_totals=Counter()
    for row in cases:
        diagnostic_totals.update(row['coordinate_match_owner_diagnostics'])
    result['coordinate_match_owner_diagnostics']=dict(diagnostic_totals)
    result['owner_diagnostic_scope']='Coordinate matching is tie-sensitive for coincident labeled pins; joint maximum matching is the primary metric.'
    for subset in ['development','acceptance_holdout','all_unique']:
        selected=[row for row in cases if subset=='all_unique' or row['subset']==subset]
        aggregate={'cases':len(selected),'assigned_candidates':sum(row['assigned_count'] for row in selected)}
        for field in ['raw_localization','assigned_localization','joint_owner_and_localization']:
            aggregate[field]=metrics(*(sum(row[field][key] for row in selected) for key in ['tp','fp','fn']))
        raw_tp=aggregate['raw_localization']['tp']
        localized_assigned_tp=aggregate['assigned_localization']['tp']
        joint_tp=aggregate['joint_owner_and_localization']['tp']
        aggregate['correct_owner_and_location_coverage_of_raw_TP']=joint_tp/raw_tp if raw_tp else None
        aggregate['owner_consistency_of_localized_assigned']=joint_tp/localized_assigned_tp if localized_assigned_tp else None
        result[subset]=aggregate
    save(output/'evaluation.json',result)
    save(output/'label_review_queue.json',{'status':'review_proposals_only_no_labels_modified','items':review_queue,
                                          'reason_counts':dict(Counter(item['review_reason'] for item in review_queue))})
    all_results=result['all_unique']
    lines=['# 引脚归属模块 v1 实测','',
           '输入为冻结 YOLO v3 元件预测和 U-Net v2 引脚预测。模型、坐标、阈值均不变；新增的是元件归属关系。所有候选保留，含歧义和无元件附近的点。',
           '配置在本轮首次评估前固定，未针对验收页继续调参。26 页仍属于已参与开发的旧验证集，参考为草稿。','',
           '| 范围 | 原引脚坐标 F1 | 已分配候选的坐标 F1 | 元件归属＋坐标联合 F1 |','|---|---:|---:|---:|']
    for key,name in [('development','13 张开发图'),('acceptance_holdout','13 张保留验收图'),('all_unique','全部 26 张')]:
        item=result[key]
        lines.append(f"| {name} | {item['raw_localization']['f1_dice']:.2%} | {item['assigned_localization']['f1_dice']:.2%} | {item['joint_owner_and_localization']['f1_dice']:.2%} |")
    lines+=['',f"候选状态：{dict(status_counts)}。",
            f"已定位且已分配候选的归属一致性：{all_results['owner_consistency_of_localized_assigned']:.2%}；正确归属且定位的覆盖率（原始坐标 TP 为分母）：{all_results['correct_owner_and_location_coverage_of_raw_TP']:.2%}。",'',
            '联合指标同时要求预测元件类别/框正确、引脚归属正确、引脚点≤5px；没有评价编号和 pinname，因此不是正式 PinF1。已分配候选的成绩不表示剩余候选可以删除。',
            '引脚编号、名称和电气网络仍待处理；当前只输出候选关系，不把未知编号换成排序编号。','',
            f'标注复核队列包含 {len(review_queue)} 条未匹配点；这是复核建议，没有改变参考标注或 approved_cases。','',
            '## 逐页结果','','| 图片 | 分配/全部候选 | 联合 F1 | 归属图 |','|---|---:|---:|---|']
    for row in cases:
        picture=(output/row['case_id']/'ownership_overlay.png').as_posix()
        lines.append(f"| {row['case_id']} | {row['assigned_count']}/{row['candidate_count']} | {row['joint_owner_and_localization']['f1_dice']:.2%} | [查看](<{picture}>) |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','config']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
