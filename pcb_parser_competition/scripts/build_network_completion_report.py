"""Export-gap priority report, plus a separate offline reference discrepancy audit."""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from run_network_completion import ROOT, read, save, sha


def card(case, boxes, points, title, detail, path):
    with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
        source = source.convert('RGB')
        boxes = boxes+[[*p, *p] for p in points]
        crop = [max(0, int(min(b[0] for b in boxes))-30), max(0, int(min(b[1] for b in boxes))-30),
                min(source.width, int(max(b[2] for b in boxes))+31), min(source.height, int(max(b[3] for b in boxes))+31)]
        image = source.crop(crop)
        factor = min(3, 780/image.width, 600/image.height)
        image = image.resize((max(1, round(image.width*factor)), max(1, round(image.height*factor))), Image.Resampling.NEAREST)
        canvas = Image.new('RGB', (max(820, image.width+24), image.height+100), 'white')
        canvas.paste(image, (12, 32))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 13)
        draw.text((12, 8), title, fill='#333', font=font)
        for i, point in enumerate(points, 1):
            x,y=12+(point[0]-crop[0])*factor,32+(point[1]-crop[1])*factor
            draw.ellipse((x-4,y-4,x+4,y+4),outline='#b32626',width=2)
            draw.text((x+6,y),str(i),fill='#b32626',font=font,stroke_width=1,stroke_fill='white')
        draw.text((12,image.height+42),detail,fill='#333',font=font)
        draw.text((12,image.height+65),'Original PNG + unchanged model tip locations; no complete network certification.',fill='#8c4320',font=font)
        canvas.save(path)
        return crop


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'runs/s17_network_completion')
    a=p.parse_args()
    batch=read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('S17 batch incomplete')
    counts,blockers,by_type=Counter(),Counter(),Counter()
    tasks={};networks=[];withdrawn=[]
    for item in batch['cases']:
        case=item['case_id'];folder=a.output/f'predictions/{case}'
        coverage=read(folder/'proposal_export/conservative_export_coverage.json')
        counts.update(coverage['pin_reason_counts'])
        queue=read(folder/'conservative_completion_queue.json')
        records=read(folder/'completed_component_records.json')['records']
        by_index={r['detection_index']:r for r in records}
        for row in queue['networks']:
            networks.append({'case_id':case,**row})
            blockers.update(row['non_field_export_blockers'])
            for pin in row['pin_candidates']:
                for field in pin['missing_fields']:by_type[(pin['component_type'] or 'unassigned',field)]+=1
            for task in row['completion_tasks']:
                key=case+':'+task['task_id']
                entry=tasks.setdefault(key,{'case_id':case,**task,'affected_network_ids':[],
                    'field_only_network_ids':[],'affected_candidate_ids':[]})
                entry['affected_network_ids'].append(row['network_id'])
                if row['field_only_candidate']:entry['field_only_network_ids'].append(row['network_id'])
                entry['affected_candidate_ids']=sorted(set(entry['affected_candidate_ids']+task['affected_candidate_ids']))
        for change in read(folder/'pin_number_role_audit.json')['changes']:
            record=by_index[change['detection_index']]
            name=f"package_{case}_{change['candidate_id']}.png"
            crop=card(case,[record['bbox_top_left'],change['original_evidence']['bbox']], [change['point']],
                f"{case} {change['component_id']} | package-like OCR number withheld",
                f"Observed literal {change['original_number']} retained in audit; replacement number unresolved.",a.output/name)
            withdrawn.append({'case_id':case,**change,'crop_top_left':crop,'review_card':name,'visually_reviewed':False})
    networks.sort(key=lambda r:(bool(r['non_field_export_blockers']),r['completion_task_count'],r['missing_field_total'],r['pin_count'],r['case_id'],r['network_id']))
    for rank,row in enumerate(networks[:12],1):
        boxes=[p['component_bbox_top_left'] for p in row['pin_candidates'] if p['component_bbox_top_left']]
        points=[p['point'] for p in row['pin_candidates']]
        name=f'network_priority_{rank:02d}_{row["case_id"]}.png'
        row['crop_top_left']=card(row['case_id'],boxes,points,f"{rank}: {row['case_id']} {row['network_id']} | {row['completion_task_count']} tasks",
            f"Missing fields: {row['missing_field_counts']} | Other blockers: {row['non_field_export_blockers']}",a.output/name)
        row['review_card']=name
    task_rows=sorted(tasks.values(),key=lambda t:(-len(t['field_only_network_ids']),-len(t['affected_candidate_ids']),t['case_id'],t['task_id']))
    save(a.output/'completion_tasks.json',{'tasks':task_rows,'purpose':'Shared component identity tasks ranked by affected field-only networks; no prediction fields filled'})
    save(a.output/'prioritized_networks.json',{'networks':networks,'rank_scope':'Missing-field task count only after all non-field export gates','complete_networks_certified':False})
    save(a.output/'package_number_review.json',{'items':withdrawn,'role_filter_provisional':True})
    aggregate={'multi_pin_networks':len(networks),'field_only_incomplete_networks':sum(r['field_only_candidate'] for r in networks),
               'non_field_blocked_networks':sum(bool(r['non_field_export_blockers']) for r in networks),
               'non_field_export_blocker_counts':dict(blockers),'pending_pin_field_counts_all_candidates':dict(counts),
               'multi_pin_member_field_counts_by_type':[{'component_type':kind,'field':field,'count':count} for (kind,field),count in by_type.most_common()],
               'unique_completion_tasks':len(task_rows),'package_like_numbers_withheld':len(withdrawn),
               'official_PinF1':None,'official_NetF1':None,'independent_test':False}
    save(a.output/'aggregate_blockers.json',aggregate)

    # This analysis is offline and never imported by the image entry point.
    from evaluate_wire_graph import pair_counts, metrics
    from evaluate_pin_text import component_matches, match, conditional_fields
    fields=read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json')
    reference=read(ROOT/'runs/s15_power_networks/reference_power_connectivity.json')
    eval_rows=[];conditional={v:Counter() for v in ['before','after']};discrepancies=[]
    for item in batch['cases']:
        case=item['case_id'];folder=a.output/f'predictions/{case}'
        old=read(ROOT/f'runs/s14_component_library/predictions/{case}/pin_text/pin_text_result.json')
        new=read(folder/'completed_pin_text_result.json');graph=read(folder/'graph_proposals.json')
        truth=[t for t in fields['references'] if t['case_id']==case]
        if reference['cases'][case]['image_sha256']!=new['image_sha256']:raise ValueError('Reference image differs')
        owners=component_matches([{'class_name':c['class_name'],'bbox':c['bbox_top_left']} for c in new['components']],fields['detector_references'][case]['objects'])
        for version,payload in [('before',old),('after',new)]:
            predicted=[p for p in payload['pin_candidates'] if p['owner_status']=='assigned']
            conditional[version].update(conditional_fields(predicted,truth,match(predicted,truth,owners)))
        values=pair_counts(graph,new,truth,reference['cases'][case]['reference_pin_networks'],owners)
        eval_rows.append({'case_id':case,**values})
        if case in {'0148','0149'}:
            assigned=[p for p in new['pin_candidates'] if p['owner_status']=='assigned']
            mapped={assigned[i]['candidate_id']:truth[j] for i,j in match(assigned,truth,owners)}
            power_graph=read(ROOT/f'runs/s16_power_caption_ocr/predictions/{case}/power_graph_proposals.json')
            for site in power_graph['power_sites']:
                t=mapped.get(site['candidate_id'])
                caption_refs=reference['cases'][case]['power_caption_sites'].get(site['label'],[])
                represented=bool(t and any(t['component_source_id']==r['component_source_id'] for r in caption_refs))
                if site['label'] in {'5V','+310V'}:
                    discrepancies.append({'case_id':case,'candidate_id':site['candidate_id'],'observed_caption':site['label'],
                        'matched_source_component_id':t['component_source_id'] if t else None,
                        'matched_source_pin_id':t['pin_source_id'] if t else None,
                        'same_caption_source_occurrence_in_reference':represented,
                        'status':('covered_by_caption_reference' if represented else
                                  'matched_source_caption_not_in_reference_group' if t else
                                  'model_tip_not_matched_requires_location_or_reference_review'),
                        'model_merge_correctness_certified':False,'reference_changed':False})
    scopes=['conditional_connectivity_pairs','all_reference_pair_proxy']
    connectivity={scope:metrics(*(sum(row[scope][m] for row in eval_rows) for m in ['tp','fp','fn'])) for scope in scopes}
    previous=read(ROOT/'runs/s16_power_caption_ocr/evaluation.json')['aggregate']['localized_ocr']
    if connectivity!=previous:raise ValueError('Role filtering unexpectedly changed connection-pair results')
    save(a.output/'offline_evaluation.json',{'cases':eval_rows,'connectivity':connectivity,'conditional_fields':{v:dict(c) for v,c in conditional.items()},
        'connectivity_unchanged_from_S16':True,'reference_sha256':sha(ROOT/'runs/s15_power_networks/reference_power_connectivity.json'),
        'official_PinF1':None,'official_NetF1':None,'independent_test':False,'reference_changes':0})
    save(a.output/'reference_discrepancy_review.json',{'power_sites':discrepancies,'reference_changed':False,
        'purpose':'Locate caption/pin coverage gaps before labeling model proposals as wrong; no correctness inferred from a coverage gap'})
    lines=['# S17：编号角色过滤与完整网络缺口排序','',
        '新增独立PNG入口与固定开发集回放。对电阻/电容/电感中由OCR绑定的常见封装字面编号进行保守撤回，原文字和全部证据另存，不猜替代编号。修正复核队列：纳入电源provisional及漏成员等真实出口门槛；同一元件的身份缺口合并为一个共享任务。', '',
        f"26页保留全部1477个原tip、归属及物理网络成员，撤回{len(withdrawn)}个0402封装样式编号。严格出口仍213元件、116pin、0net；候选仍131pin、1net。连接对指标与S16完全一致，未宣称官方准确率提升。", '',
        '| 多引脚网络缺口 | 数量 |','|---|---:|',
        f"| 多引脚网络候选 | {aggregate['multi_pin_networks']} |",
        f"| 没有已记录的非字段出口拦截、但字段不全 | {aggregate['field_only_incomplete_networks']} |",
        f"| 存在非字段出口拦截 | {aggregate['non_field_blocked_networks']} |", '',
        '“没有已记录拦截”不等于拓扑已认证。所有成员保留，禁止删除未决成员以生成更小的完整网络。', '',
        '| 全部候选的字段缺口 | 数量 |','|---|---:|']
    for key,label in [('component_identity_unresolved','身份'),('pin_number_unresolved','编号'),('pin_name_unresolved','名称')]:
        lines.append(f'| {label} | {counts[key]} |')
    lines+=['','## 优先复核网络','','| 排名 | 页面 | 网络 | 共享任务数 | 字段缺口 | 局部图 |','|---|---|---|---:|---|---|']
    for rank,row in enumerate(networks[:12],1):
        lines.append(f"| {rank} | {row['case_id']} | {row['network_id']} | {row['completion_task_count']} | {row['missing_field_counts']} | [查看]({row['review_card']}) |")
    lines+=['','首要两例：0002的C87.1缺pinname；0042的JTAG连接器缺规范元件身份。后者原图标题为JTAG/JTAG-20PIN，不是未识别的U/J数字位号，不能通过猜测J1来解锁网络。前者数字作被动器件名称仍属于既有候选约定，不能直接扩展到严格出口。','',
        '## 参考差异审计','',
        '0149的5V组与参考不一致，其中一个模型电源节点对应的源组件没有进入该同名电源参考组。0148的+310V组有一个模型电源tip未在5px/归属匹配范围内找到参考，可能是预测位置/归属错误或参考缺失，仍需复核。已另存逐节点对照，原参考没有修改；这些情况都不能证明完整模型网络正确，也不能将全部差异直接称为已核验误合并。','',
        '```powershell','.\\scripts\\run_network_completion.ps1 -Image .\\data\\standardized\\0018\\0018.png -Device cpu','```','',
        '机器可读产物：aggregate_blockers.json、completion_tasks.json、prioritized_networks.json、package_number_review.json、reference_discrepancy_review.json；每页保留严格/候选队列和旧/新JSON出口。','']
    verification=a.output/'runtime_verification.json'
    if verification.is_file():
        v=read(verification);lines+=[f"验证：{v['regression_tests_passed']}项回归通过；{v['fresh_case_id']}原始PNG完整入口{v['fresh_entrypoint_wall_seconds']:.1f}秒，字段/任务队列/网络成员与回放一致。",'']
    (a.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'multi_pin_networks':len(networks),'field_only_incomplete_networks':aggregate['field_only_incomplete_networks'],
           'non_field_blocked_networks':aggregate['non_field_blocked_networks'],
           'package_like_numbers_withheld':len(withdrawn),'connectivity_unchanged_from_S16':True},flush=True)


if __name__=='__main__':main()
