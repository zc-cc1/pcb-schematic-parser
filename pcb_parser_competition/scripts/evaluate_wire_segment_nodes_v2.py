"""Audit source segmentation and test predicted node refinement with fixed raw scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.wire_segment_nodes import DEFAULTS, refine_segment_nodes
from pcb_parser.wire_segments import component_bridge_guard, export_candidate_edges
from audit_unet_competition_geometry import match_segments
from evaluate_wire_segments_v1 import total, unmatched_analysis

BASE=ROOT/'runs/wire_segments_v1_guarded_20261015'
OUT=ROOT/'runs/wire_segment_nodes_v2_20261015'
CACHE=ROOT/'runs/training/unet_v7_finetune_comparison_20261015/evaluation'
AUDIT=ROOT/'runs/unet_competition_requirements_20261015/geometry_audit.json'
PRESETS={
    'identity':DEFAULTS|dict(enabled=False),
    'snap3':DEFAULTS,
    'snap5':DEFAULTS|dict(snap_radius_px=5.),
    'pin_splits':DEFAULTS|dict(split_at_pins=True),
    'T_splits':DEFAULTS|dict(split_at_T_nodes=True),
    'pin_T_splits':DEFAULTS|dict(split_at_pins=True,split_at_T_nodes=True),
}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def overlap_union(refs):
    """Sensitivity view only: exact collinear overlap within one source edge.

    Merely touching segments stay separate. No short-length filtering and no
    changed masks/annotations. This view is NOT an official Golden conversion.
    """
    groups={};other=[]
    for row in refs:
        a,b=np.asarray(row['points'])
        if a[1]==b[1]:key=(row['edge_id'],'h',a[1]);lo,hi=sorted((a[0],b[0]))
        elif a[0]==b[0]:key=(row['edge_id'],'v',a[0]);lo,hi=sorted((a[1],b[1]))
        else:other.append(dict(row));continue
        groups.setdefault(key,[]).append((float(lo),float(hi)))
    for (edge,axis,level),intervals in groups.items():
        merged=[]
        for lo,hi in sorted(intervals):
            if merged and lo < merged[-1][1]-1e-6:
                merged[-1][1]=max(merged[-1][1],hi)
            else:merged.append([lo,hi])
        for lo,hi in merged:
            points=[[lo,float(level)],[hi,float(level)]] if axis=='h' else [[float(level),lo],[float(level),hi]]
            other.append(dict(edge_id=edge,points=points,length=hi-lo))
    return other


def profiles(audit):
    manifest={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    rows=[]
    for row in audit['variants']['reviewed_v7']:
        case=row['case_id']; annotation,_=load_source_annotation(manifest[case]['raw_annotation_path'])
        refs=row['segments'];short=[]
        for ref in refs:
            if ref['length']>5:continue
            edge=annotation['edges'][ref['edge_id']]
            connections=[c['location'] for c in edge.get('connect',{}).values() if isinstance(c,dict) and isinstance(c.get('location'),list) and len(c['location'])==2]
            if any(np.linalg.norm(np.asarray(point)-np.asarray(tip))<=3 for point in ref['points'] for tip in connections):
                short.append(dict(edge_id=ref['edge_id'],points=ref['points'],length=ref['length']))
        canonical=overlap_union(refs)
        examples=[]
        if case in {'0018','0170','0007'}:
            for eid in dict.fromkeys(s['edge_id'] for s in short):
                edge=annotation['edges'][eid]
                examples.append(dict(edge_id=eid,paths=edge.get('keyPointPaths'),connect=edge.get('connect'),key_points=edge.get('keyPoints')))
                if len(examples)>=2:break
        rows.append(dict(case_id=case,raw_segments=len(refs),segments_le_5px=sum(s['length']<=5 for s in refs),
                         short_segments_near_source_pin=len(short),overlap_union_segments=len(canonical),
                         overlapping_segment_reduction=len(refs)-len(canonical),short_pin_examples=short[:6],source_path_examples=examples))
    write(OUT/'segmentation_audit.json',dict(status='reference_audit_only_no_source_changes',cases=rows,
        total_raw_segments=sum(r['raw_segments'] for r in rows),total_short_pin_segments=sum(r['short_segments_near_source_pin'] for r in rows),
        overlap_union_segments=sum(r['overlap_union_segments'] for r in rows),official_Golden_verified=False,
        note='Raw 2266-segment reference remains the primary score. Overlap union is sensitivity only; no tiny edge is simply discarded.'))
    return rows


def context(variant,case):
    original=read(BASE/variant/case/'segments.json')
    pins=read(CACHE/variant/case/'pin_candidates.json')
    components=read(ROOT/f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json')
    guard=component_bridge_guard((original['size'][1],original['size'][0]),components['components'])
    return original,pins,guard


def score(variant,case,cfg,reference,cache):
    key=(variant,case)
    if key not in cache:cache[key]=context(variant,case)
    original,pins,guard=cache[key]
    decoded=refine_segment_nodes(original,pins,cfg,guard)
    points=[r['points'] for r in decoded['segments']]
    metric=match_segments(points,reference)
    record=dict(case_id=case,metrics=metric,error_hints=unmatched_analysis(points,reference,metric),
                snapped_endpoints=decoded['node_refinement']['snapped_endpoints'],interior_splits=decoded['node_refinement']['interior_splits'])
    return decoded,record


def render(case,old,new,refs,roi,new_title='当前保留v1（未改动）',suffix=''):
    rgb=np.asarray(Image.open(ROOT/f'data/standardized/{case}/{case}.png').convert('RGB'))
    x0,y0,x1,y1=roi
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4),dpi=150,layout='constrained')
    for ax,title,segments in zip(axes,('原图＋原始JSON分段','v1线段端点',new_title),([s['points'] for s in refs],old,new)):
        ax.imshow(rgb[y0:y1,x0:x1],interpolation='nearest')
        for a,b in np.asarray(segments):
            if max(a[0],b[0])<x0 or min(a[0],b[0])>x1 or max(a[1],b[1])<y0 or min(a[1],b[1])>y1:continue
            ax.plot([a[0]-x0,b[0]-x0],[a[1]-y0,b[1]-y0],color='#d96a12' if title.startswith('原图') else '#009557',lw=1.3,marker='o',ms=3)
        ax.set_xlim(0,x1-x0);ax.set_ylim(y1-y0,0);ax.axis('off');ax.set_title(title)
    fig.suptitle(f'{case} · 原参考分段保留；预测仅由模型tip与线段几何产生')
    fig.savefig(OUT/f'example_{case}{suffix}.png',facecolor='white');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=read(AUDIT);groups=read(ROOT/'runs/unet_inference_v2/split.json')
    dev=groups['development'];check=groups['acceptance_holdout'];all_cases=dev+check
    reference={r['case_id']:r['segments'] for r in audit['variants']['reviewed_v7']}
    protected=read(BASE/'protocol.json')['protected_sha256']|{str(AUDIT):sha(AUDIT)}
    for variant in audit['variants']:
        for case in all_cases:
            p=BASE/variant/case/'segments.json';protected[str(p)]=sha(p)
    write(OUT/'protocol.json',dict(presets=PRESETS,selection='highest raw-reference development endpoint F1; identity breaks ties',
        acceptance='check13 raw F1 and TP must not decrease; otherwise keep v1',groups=dict(development=dev,check13=check),
        raw_reference_unchanged=True,reference_union_used_for_selection=False,independent_test=False,
        official_NetLineF1=None,protected_sha256=protected))
    profile=profiles(audit);cache={};candidates={}
    for name,cfg in PRESETS.items():
        records=[score('reviewed_v7',case,cfg,reference[case],cache)[1] for case in dev]
        candidates[name]=dict(summary=total(records),cases=records)
        print(json.dumps(dict(preset=name,development=candidates[name]['summary'])),flush=True)
    selected=max(PRESETS,key=lambda n:candidates[n]['summary']['f1'])
    candidate_rows=[score('reviewed_v7',case,PRESETS[selected],reference[case],cache)[1] for case in check]
    identity_rows=[score('reviewed_v7',case,PRESETS['identity'],reference[case],cache)[1] for case in check]
    candidate_check=total(candidate_rows);baseline_check=total(identity_rows)
    accepted=candidate_check['f1']>=baseline_check['f1'] and candidate_check['tp']>=baseline_check['tp']
    adopted=selected if accepted else 'identity'
    cfg=PRESETS[adopted]
    write(OUT/'selection.json',dict(selected_on_development=selected,accepted_on_check=accepted,adopted=adopted,
        candidate_check=candidate_check,baseline_check=baseline_check,candidates=candidates))
    write(ROOT/'configs/wire_segment_nodes_v2.json',cfg|dict(version='wire-segment-nodes-v2',selected_preset=adopted,selection_record=str(OUT/'selection.json')))
    comparisons={}
    for variant in audit['variants']:
        before=[];after=[];union_before=[];union_after=[]
        for case in all_cases:
            if (variant,case) not in cache:cache[(variant,case)]=context(variant,case)
            original,pins,guard=cache[(variant,case)]
            decoded,record=score(variant,case,cfg,reference[case],cache)
            candidate,proposal=score(variant,case,PRESETS[selected],reference[case],cache)
            old_points=[s['points'] for s in original['segments']];new_points=[s['points'] for s in decoded['segments']]
            before.append(dict(case_id=case,metrics=match_segments(old_points,reference[case])))
            after.append(record)
            normalized=overlap_union(reference[case])
            union_before.append(dict(case_id=case,metrics=match_segments(old_points,normalized)))
            union_after.append(dict(case_id=case,metrics=match_segments(new_points,normalized)))
            folder=OUT/variant/case
            write(folder/'segments.json',decoded);write(folder/'candidate_edges.json',export_candidate_edges(decoded));write(folder/'evaluation.json',record)
            if adopted!=selected:write(folder/'rejected_candidate.json',dict(segments=candidate,evaluation=proposal))
            if variant=='reviewed_v7' and case in {'0018','0170','0007'}:
                roi={'0018':(565,250,735,310),'0170':(750,25,845,90),'0007':(125,440,245,485)}[case]
                render(case,old_points,new_points,reference[case],roi)
            print(json.dumps(dict(variant=variant,case=case,f1=record['metrics']['f1'],snaps=record['snapped_endpoints'],splits=record['interior_splits'])),flush=True)
        scores={}
        for group,members in dict(all26=all_cases,development=dev,check13=check).items():
            scores[group]=dict(before=total([r for r in before if r['case_id'] in members]),after=total([r for r in after if r['case_id'] in members]))
        comparisons[variant]=dict(groups=scores,cases=after,overlap_union_sensitivity=dict(before=total(union_before),after=total(union_after)))
    assert all(sha(p)==h for p,h in protected.items())
    result=dict(status='completed',selected_candidate=selected,adopted=adopted,accepted_on_check=accepted,
        comparisons=comparisons,raw_reference_unchanged=True,protected_inputs_unchanged=True,
        training_performed=False,default_weights_replaced=False,official_NetLineF1=None)
    write(OUT/'comparison.json',result)
    lines=['# 预测连接点切分与源分段审计','',f'开发组选择{selected}；另一13页检查后采用{adopted}。原始2266段参考保持，按5px端点一对一全图匹配；未按net限制，不是官方NetLineF1。历史验证页已被反复查看，不是独立测试。','',
        '| 模型 | 页面 | v1 F1 | 当前F1 | v1 TP/FP/FN | 当前TP/FP/FN |','|---|---|---:|---:|---|---|']
    for variant,data in comparisons.items():
        for group,s in data['groups'].items():
            a,b=s['before'],s['after'];lines.append(f"| {variant} | {group} | {a['f1']:.2%} | {b['f1']:.2%} | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} |")
    lines+=['','## 参考分段发现','',f"26页共有{sum(p['short_segments_near_source_pin'] for p in profile)}段≤5px的源几何靠近声明pin位置，可能是接入小折线，不能直接视为独立可见长导线。对同一source edge只做严格共线重叠合并后参考数量为{sum(p['overlap_union_segments'] for p in profile)}；相接但不重叠的段保持分开，没有简单删除短段。该解释尚未得到官方Golden确认。",'',
        '0018/E_3：主线(595,284)→(703,284)，另有(594,285)→(595,285)与(595,285)→(595,284)两段1px接入折线。0170/E_1：主线(776,50)→(818,50)，另有(820,50)→(778,50)和(778,50)→(776,50)，存在重叠；0007/E_1也有1px与3px接入折线。详见segmentation_audit.json。','',
        '这些短段/重叠可能是源数据的接入表示，未确认官方是否保留，不能宣称已证实标签错误。原始评分一直保留，重叠合并仅作敏感性对照，绝不用于选参数、重写标签或报告模型提升。','',
        '## 解码改动','',
        '仅使用预测tip及已有线段；校正近端点位置，在有充分证据的内侧tip/T型支路处切分，保留最小片段长度和歧义拒绝。相对两侧支路形成的疑似十字交叉不自动切分或推断电气连接，组件主体仍阻止新增连接。每次端点移动/切分均有审计记录。','',
        '## 参考解释敏感性（非模型提升）','','| 模型 | v1在合并解释下F1 | 当前在合并解释下F1 |','|---|---:|---:|']
    for variant,data in comparisons.items():
        s=data['overlap_union_sensitivity'];lines.append(f"| {variant} | {s['before']['f1']:.2%} | {s['after']['f1']:.2%} |")
    lines+=['','该表只改变离线参考解释，不能与原始分数相减后称为模型进步。完整赛事JSON仍需网络归属及逻辑关系。未重训或替换默认权重。','',
            '![0018](example_0018.png)','','![0170](example_0170.png)','','![0007](example_0007.png)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(selected=selected,adopted=adopted,reviewed_v7=comparisons['reviewed_v7']['groups']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split-only',action='store_true',help='Ablate cuts without moving endpoints after the first development experiment')
    args=parser.parse_args()
    if args.split_only:
        OUT=ROOT/'runs/wire_segment_nodes_v2_split_only_20261015'
        PRESETS={
            'identity':DEFAULTS|dict(enabled=False),
            'pin_splits_without_snap':DEFAULTS|dict(snap_radius_px=0.,split_at_pins=True),
            'T_splits_without_snap':DEFAULTS|dict(snap_radius_px=0.,split_at_T_nodes=True),
            'pin_T_splits_without_snap':DEFAULTS|dict(snap_radius_px=0.,split_at_pins=True,split_at_T_nodes=True),
        }
    main()
