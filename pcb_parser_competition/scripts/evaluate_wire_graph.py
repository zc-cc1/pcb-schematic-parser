"""Offline draft connectivity evaluation; never imported by PNG inference."""
from collections import Counter,defaultdict
import argparse
import itertools
import json
import math

from run_wire_graph import ROOT,read,save,sha
from evaluate_pin_text import component_matches,match


def metrics(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def uid(reference):return reference['component_source_id']+'/'+reference['pin_source_id']


def pair_counts(graph,pin_result,truth,reference_nets,owner_map):
    assigned=[p for p in pin_result['pin_candidates'] if p['owner_status']=='assigned']
    pairs=match(assigned,truth,owner_map)
    mapped={assigned[i]['candidate_id']:uid(truth[j]) for i,j in pairs}
    network={p['candidate_id']:p['network_id'] for p in graph['pin_attachments']}
    true_group={uid(t):reference_nets.get(uid(t),'isolated:'+uid(t)) for t in truth}
    tp=fp=fn=0
    examples=[]
    ids=sorted(mapped)
    for a,b in itertools.combinations(ids,2):
        expected=true_group[mapped[a]]==true_group[mapped[b]]
        predicted=network[a] is not None and network[a]==network[b]
        tp+=expected and predicted;fp+=not expected and predicted;fn+=expected and not predicted
        if predicted!=expected and len(examples)<20:
            examples.append({'pin_candidates':[a,b],'reference_pin_ids':[mapped[a],mapped[b]],
                             'reference_networks':[true_group[mapped[a]],true_group[mapped[b]]],
                             'predicted_networks':[network[a],network[b]],'error':'false_merge' if predicted else 'missed_connection'})
    predicted_pairs=sum(math.comb(len(n['pin_candidate_ids']),2) for n in graph['networks'])
    true_counts=Counter(true_group.values())
    expected_pairs=sum(math.comb(n,2) for n in true_counts.values())
    # The one-to-one owner/location map means a correctly predicted pair can be
    # counted only once. Unmatched predicted pins remain in the FP denominator.
    end_to_end=metrics(tp,predicted_pairs-tp,expected_pairs-tp)
    return {'localized_owned_pins':len(mapped),'conditional_connectivity_pairs':metrics(tp,fp,fn),
            'all_reference_pair_proxy':end_to_end,'error_examples':examples}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-references',action='store_true',help='Score the exported reference snapshot without opening raw annotations')
    args=parser.parse_args()
    output=ROOT/'runs/s7_wire_graph_v1'
    batch=read(output/'batch_summary.json')
    fields=read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json')
    if args.reuse_references:
        physical=read(output/'reference_connectivity.json')
    else:
        from pcb_parser.dataset_converter import Converter,load_source_annotation
        raw_manifest={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
        policy=read(ROOT/'configs/conversion.json')
        policy['merge_declared_edge_labels']=False;policy['merge_same_name_label_components']=False
        overrides=read(ROOT/'configs/annotation_overrides.json')['cases']
        physical={'cases':{},'purpose':'Offline draft connectivity only; no reference used in image inference',
                  'same_name_label_merging_disabled':True,'source_edge_grouping_independently_verified':False}
    cases=[]
    totals={variant:Counter() for variant in ['default','observed_connected']}
    structural=Counter();coverage=Counter();review=[]
    for row in batch['cases']:
        case=row['case_id']
        pin_result=read(ROOT/f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json')
        assert sha(ROOT/f'data/standardized/{case}/{case}.png')==pin_result['image_sha256']==fields['image_sha256'][case]
        truth=[r for r in fields['references'] if r['case_id']==case]
        if not args.reuse_references:
            from pathlib import Path
            raw_path=Path(raw_manifest[case]['raw_annotation_path'])
            assert sha(raw_path)==raw_manifest[case]['annotation_sha256']
            source,_=load_source_annotation(raw_path)
            converter=Converter(source,case,policy,overrides.get(case))
            converter.run()
            mapping={p:nid for nid,net in converter.net_trace.items() for p in net['source_pin_ids']}
            physical['cases'][case]={'image_sha256':pin_result['image_sha256'],'reference_pin_networks':mapping,
                                    'source_annotation_sha256':raw_manifest[case]['annotation_sha256']}
        reference=physical['cases'][case]
        assert reference['image_sha256']==pin_result['image_sha256']
        detections=[{'bbox':c['bbox_top_left'],'class_name':c['class_name']} for c in pin_result['components']]
        owner_map=component_matches(detections,fields['detector_references'][case]['objects'])
        record={'case_id':case}
        for variant,folder in [('default','predictions'),('observed_connected','observed_connected')]:
            graph=read(output/f'{folder}/{case}/graph_proposals.json')
            assert graph['image_sha256']==pin_result['image_sha256']
            statistics=pair_counts(graph,pin_result,truth,reference['reference_pin_networks'],owner_map)
            record[variant]=statistics
            for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']:
                for field in ['tp','fp','fn']:totals[variant][key+'_'+field]+=statistics[key][field]
            totals[variant]['localized_owned_pins']+=statistics['localized_owned_pins']
            if variant=='default':
                structural.update({k:graph['summary'][k] for k in ['attached_pins','network_proposals','multi_pin_network_proposals','uncertain_junctions','gap_candidates_not_filled','graph_nodes','graph_edges','skeleton_pixels']})
                structural['pin_candidates']+=len(graph['pin_attachments'])
                export=read(output/f'predictions/{case}/export_coverage.json')
                coverage.update({k:export[k] for k in ['exported_components','exported_pins','exported_nets']})
                queue=read(output/f'predictions/{case}/review_queue.json')
                review.extend({'case_id':case,**item,'label_change_applied':False} for item in queue['items'])
        cases.append(record)
    aggregate={}
    for variant,counts in totals.items():
        aggregate[variant]={'localized_owned_pins':counts['localized_owned_pins']}
        for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']:
            aggregate[variant][key]=metrics(*(counts[key+'_'+f] for f in ['tp','fp','fn']))
    result={'status':'learned_wire_graph_v1_draft_evaluated','cases':cases,'aggregate':aggregate,'structure_totals':dict(structural),
            'partial_export_totals':dict(coverage),'official_NetF1':None,'engineering_test_independent':False,'label_changes_applied':0,
            'conditional_metric_scope':'Pairs only between already owner/location-matched pins; unmatched and disconnected pairs remain FN where appropriate',
            'all_reference_metric_scope':'Pin-pair proxy, not NetF1; uses original draft edgeId groups, backreferences and coincident source tips. Same-name merges disabled.',
            'reference_limits':'Original edge grouping may itself combine logical/physical relations; missing source pins and incorrect tips remain. Reference has not been independently certified.',
            'junction_model_available':False,'logical_label_merges':0,'config_sha256':sha(ROOT/'configs/wire_graph_v1.json')}
    save(output/'evaluation.json',result)
    if not args.reuse_references:save(output/'reference_connectivity.json',physical)
    save(output/'review_queue.json',{'items':review,'count':len(review),'automatic_label_changes':0})
    lines=['# U-Net 导线图与 NetworkX 网络阶段 v1','',
           '输入为冻结的 U-Net 导线概率、YOLO 元件预测及引脚/文字模型预测。只对模型掩膜做骨架与图转换；不读取原图颜色来提取导线。',
           '所有结果来自已使用过的 26 张开发验证页，原参考为草稿，不能当作正式 NetF1 或独立工程测试。','',
           '| 产物 | 数量 |','|---|---:|',
           f"| 保留引脚候选 | {structural['pin_candidates']} |",f"| 唯一接入导线图的候选 | {structural['attached_pins']} |",
           f"| 多引脚网络候选 | {structural['multi_pin_network_proposals']} |",f"| 含零/一个引脚的线段碎片也计入的网络候选总数 | {structural['network_proposals']} |",
           f"| 交叉语义待判断 | {structural['uncertain_junctions']} |",f"| 断线候选（未自动补连） | {structural['gap_candidates_not_filled']} |",'',
           '## 已定位、已归属引脚间的连通代理指标','',
           '| 配置 | 精确率 | 召回率 | F1 |','|---|---:|---:|---:|']
    for variant,label in [('default','四向交叉分离假设（默认）'),('observed_connected','直接沿掩膜连通（对照）')]:
        metric=aggregate[variant]['conditional_connectivity_pairs']
        lines.append(f"| {label} | {metric['precision']:.2%} | {metric['recall']:.2%} | {metric['f1']:.2%} |")
    overall=aggregate['default']['all_reference_pair_proxy']
    lines+=['',f"计入未匹配预测引脚与所有参考连通引脚对后的整体配对代理指标：P={overall['precision']:.2%}，R={overall['recall']:.2%}，F1={overall['f1']:.2%}。这仍不是 NetF1，未评价正式编号、pinname、logical flywire 与全部 edges。",'',
            '参考依据为原始 edgeId 分组、连接回指和重合 tip；本轮关闭同名标签/VCC/GND 合并，但原始 edge 分组的物理/逻辑区分仍未经独立核验。',
            'T 形按模型掩膜暂作连通，四向/多向交叉保留多种解释，默认按可识别的直通方向分离。没有训练好的 junction/crossing 分类头，橙色候选不能当作已确认连接点。','',
            f"部分结构导出：{coverage['exported_components']} 个元件、{coverage['exported_pins']} 个编号/名称齐全的 pin、{coverage['exported_nets']} 个满足保守条件的 net。unknown 名称未猜为空字符串，含未决引脚/交叉/断线的网络被明确保留在 withheld_networks，文件名为 partial_submission.json，不能作为完整赛事提交。",'',
            '## 逐页可视化','', '| 图片 | 接入候选 | 多引脚网络候选 | 交叉/断线待判 | 图 |','|---|---:|---:|---:|---|']
    for row in batch['cases']:
        s=row['default']['summary'];case=row['case_id']
        lines.append(f"| {case} | {s['attached_pins']} | {s['multi_pin_network_proposals']} | {s['uncertain_junctions']}/{s['gap_candidates_not_filled']} | [查看](<{(output/'predictions'/case/'wire_graph_overlay.png').as_posix()}>) |")
    lines+=['',f'复核队列共 {len(review)} 条；候选状态和误连/漏连示例保存在 evaluation.json、review_queue.json。没有修改标注或进行新训练。','',
            '下一步：核对交叉/连接点候选并准备学习监督，补足导线断裂和符号边框误分割，再增加同名网络标签及电源符号语义。',
            '图转换使用 [scikit-image skeletonize](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize) 和 [NetworkX connected_components](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html)。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'structure':dict(structural),'aggregate':aggregate,'partial_export':dict(coverage),'review_items':len(review)},ensure_ascii=False))


if __name__=='__main__':main()
