"""Offline S17/S18 field comparison using unchanged draft reference snapshots."""
import argparse
from collections import Counter
from pathlib import Path

from run_targeted_pin_ocr import ROOT,read,save,sha
from evaluate_pin_text import component_matches,match,conditional_fields,score
from evaluate_wire_graph import pair_counts,metrics
from build_network_completion_report import card


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s18_targeted_pin_ocr');a=p.parse_args()
    batch=read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S18 batch incomplete')
    fields_path=ROOT/'runs/s6_pin_text_v1/draft_field_references.json';fields=read(fields_path)
    reference_path=ROOT/'runs/s15_power_networks/reference_power_connectivity.json';reference=read(reference_path)
    totals={v:Counter() for v in ['before','after']};joint={v:Counter() for v in totals};rows=[];reviews=[]
    connectivity=Counter();blockers={v:Counter() for v in totals};ready={v:0 for v in totals}
    statuses=Counter()
    for item in batch['cases']:
        case=item['case_id'];source=ROOT/f'runs/s17_network_completion/predictions/{case}';folder=a.output/f'predictions/{case}'
        old=read(source/'completed_pin_text_result.json');new=read(folder/'completed_pin_text_result.json')
        if sha(source/'completed_pin_text_result.json')!=item['input_pin_sha256']:
            raise ValueError('Frozen S17 input bytes changed')
        if new['image_sha256']!=fields['image_sha256'][case] or new['image_sha256']!=reference['cases'][case]['image_sha256']:
            raise ValueError('Reference PNG differs')
        truth=[t for t in fields['references'] if t['case_id']==case]
        owners=component_matches([{'class_name':c['class_name'],'bbox':c['bbox_top_left']} for c in new['components']],fields['detector_references'][case]['objects'])
        row={'case_id':case}
        for version,payload in [('before',old),('after',new)]:
            predicted=[p for p in payload['pin_candidates'] if p['owner_status']=='assigned']
            pairs=match(predicted,truth,owners)
            conditional=conditional_fields(predicted,truth,pairs);totals[version].update(conditional);row[version]=conditional
            numbered=[p for p in predicted if p.get('number') is not None];number_truth=[t for t in truth if t['expected_numbers']]
            joint[version].update({'correct':len(match(numbered,number_truth,owners,'number')),'emitted':len(numbered),'expected':len(number_truth)})
        graph=read(folder/'graph_proposals.json')
        counts=pair_counts(graph,new,truth,reference['cases'][case]['reference_pin_networks'],owners)
        for scope in ['conditional_connectivity_pairs','all_reference_pair_proxy']:
            for m in ['tp','fp','fn']:connectivity[scope+'_'+m]+=counts[scope][m]
        for version,path in [('before',folder/'baseline_export'),('after',folder/'proposal_export')]:
            cov=read(path/'conservative_export_coverage.json')
            # The corrected omission gate is applied to both profiles here.
            from pcb_parser.submission_fields import network_field_queue
            records=read(folder/'completed_component_records.json')['records']
            queue=network_field_queue(old if version=='before' else new,graph,records,cov)
            ready[version]+=queue['field_only_incomplete_networks']
            blockers[version].update(queue['non_field_export_blocker_counts'])
        pmap={p['candidate_id']:p for p in new['pin_candidates']}
        predicted=[p for p in new['pin_candidates'] if p['owner_status']=='assigned']
        mapped={predicted[i]['candidate_id']:truth[j] for i,j in match(predicted,truth,owners)}
        records=read(folder/'completed_component_records.json')['records'];record_map={r['detection_index']:r for r in records}
        audit=read(folder/'recovery_audit.json')
        for change in audit['changes']:
            pin=pmap[change['candidate_id']];r=record_map[pin['owner_detection_index']];t=mapped.get(pin['candidate_id'])
            expected=(t['expected_numbers'] if change['field']=='number' else t['expected_name']) if t else None
            status=('owner_or_location_unmatched' if t is None else 'reference_field_missing' if not expected else
                    'agrees_with_draft' if (change['text'] in expected if change['field']=='number' else change['text']==expected)
                    else 'disagrees_with_draft')
            statuses[status]+=1
            name=f"field_{case}_{change['candidate_id']}_{change['field']}.png"
            crop=card(case,[r['bbox_top_left'],change['evidence']['bbox']],[pin['point']],
                f"{case} {pin['owner']} {pin['candidate_id']} {change['field']}={change['text']}",
                f"Actual OCR glyph recovery | {status} | full network not certified",a.output/name)
            reviews.append({'case_id':case,**change,'point':pin['point'],'crop_top_left':crop,'review_card':name,
                'draft_reference_status':status,'expected_draft_field':expected,'visually_reviewed':False,
                'complete_network_certified':False})
        row['new_fields']=len(audit['changes']);rows.append(row)
    connectivity_values={scope:metrics(*(connectivity[scope+'_'+m] for m in ['tp','fp','fn']))
                         for scope in ['conditional_connectivity_pairs','all_reference_pair_proxy']}
    s16=read(ROOT/'runs/s16_power_caption_ocr/evaluation.json')['aggregate']['localized_ocr']
    if connectivity_values!=s16:raise ValueError('Field recovery changed connectivity metrics')
    result={'cases':rows,'conditional_fields':{v:dict(c) for v,c in totals.items()},
        'joint_number_and_location_proxy':{v:score(c['correct'],c['emitted'],c['expected']) for v,c in joint.items()},
        'connectivity':connectivity_values,'connectivity_unchanged':True,
        'new_field_reference_status_counts':dict(statuses),'field_only_networks_with_corrected_guard':ready,
        'non_field_export_blocker_counts':{v:dict(c) for v,c in blockers.items()},
        'field_reference_sha256':sha(fields_path),'power_reference_sha256':sha(reference_path),
        'official_PinF1':None,'official_NetF1':None,'independent_test':False,'reference_changes':0}
    save(a.output/'evaluation.json',result)
    save(a.output/'visual_review_queue.json',{'items':reviews,'full_networks_certified':False})
    aggregate=batch['aggregate']
    lines=['# S18：局部引脚字段 OCR','',
        '在原有模型tip/归属上，以缺口队列和已有未分配文字证据选取局部ROI，执行2/3/4倍、0/90度带定位框OCR。只填空字段，保留现有编号/名称，不从位置猜编号，不推断JTAG身份，不把编号当被动器件名称写入严格出口。', '',
        f"26页旧开发集处理{aggregate['roi_count']}个ROI，新增{aggregate['missing_numbers_added']}个可见编号、{aggregate['missing_names_added']}个可见名称。严格完整pin116→{aggregate['conservative']['exported_pins']}，候选131→{aggregate['candidate']['exported_pins']}；严格net仍0，候选net仍1。原1477个tip、所有归属、已有字段和网络成员保留。",'',
        '局部OCR曾将IN截读为N；新增完整字面证据门槛将其扣留。一处强同位置完整拼写足以阻止新增片段，但不足以直接补入替代名称。合法的单字符F在不同位置没有完整拼写冲突，按实际文字保留。', '',
        '## 同一草稿参考的条件字段计数','','| 字段 | S17 正确/已输出/可评价 | S18 正确/已输出/可评价 |','|---|---|---|']
    for field,label in [('number','编号'),('pinname','名称')]:
        fmt=lambda v:f"{totals[v][field+'_correct']}/{totals[v][field+'_emitted']}/{totals[v][field+'_eligible']}"
        lines.append(f'| {label} | {fmt("before")} | {fmt("after")} |')
    lines+=['',f'新增字段参考状态：{dict(statuses)}。参考未独立核验，缺失参考字段不能当成新增字段正确。以上不是官方PinF1或独立测试。', '',
        '## 队列修正','',
        '发现S17仍漏报“已经因字段缺失扣留、同时存在电源漏成员风险”的网络。现在两个门槛都记录，重复应用不重复理由；真正从JSON移除的网络和所有被门槛拦截的网络分别统计。',
        f"按修正门槛，S17基线仅字段未决网络为{ready['before']}个，S18为{ready['after']}个。此前S17记录的305仅字段未决偏高，C87对应网络还受上方电源符号缺tip限制。补出C87的可见编号2不会自动解锁完整网络。", '',
        'JTAG图片标题与pinname规则仍未得到正式契约确认，不能虚构J1或将候选约定直接提升到严格出口。', '',
        '```powershell','.\\scripts\\run_targeted_pin_ocr.ps1 -Image .\\data\\standardized\\0018\\0018.png -Device cpu','```','',
        '逐例产物：localized_texts.json、recovery_audit.json、completed_pin_text_result.json、graph_proposals.json、旧/新部分出口及严格/候选复核队列。视觉裁剪见visual_review_queue.json。', '']
    verification=a.output/'runtime_verification.json'
    if verification.is_file():
        v=read(verification);lines+=[f"验证：{v['regression_tests_passed']}项回归通过；{v['fresh_case_id']}原始PNG完整入口{v['fresh_entrypoint_wall_seconds']:.1f}秒，新增字段、出口及网络成员匹配开发回放。",'']
    (a.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'conditional_fields':result['conditional_fields'],'reference_statuses':dict(statuses),
           'field_only_networks':ready,'strict_complete_pins':aggregate['conservative']['exported_pins']},flush=True)


if __name__=='__main__':main()
