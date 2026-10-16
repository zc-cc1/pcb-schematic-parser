"""Offline ordinary-label reference comparison. Never called by PNG inference."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path

import networkx as nx
from PIL import Image,ImageDraw,ImageFont

from run_network_labels import ROOT,read,save,sha
from evaluate_wire_graph import pair_counts,metrics
from evaluate_pin_text import component_matches
from pcb_parser.network_labels import power_name


def reference_snapshot(output,cases,cfg):
    from pcb_parser.dataset_converter import Converter,load_source_annotation
    import json
    manifests={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    policy=read(ROOT/'configs/conversion.json');policy.update(merge_declared_edge_labels=False,merge_same_name_label_components=False)
    overrides=read(ROOT/'configs/annotation_overrides.json')['cases']
    physical=read(ROOT/'runs/s7_wire_graph_v1/reference_connectivity.json')
    visual_path=ROOT/'data/reviews/network_labels_logical_v1.json';visual=read(visual_path)
    result={'cases':{},'scope':'Offline source-annotation logical-label draft, not official Golden',
            'power_merging':False,'source_edge_declared_label_merge':False,'cross_image_merging':False,
            'source_label_types':['PIN','Testpoint','NET_input','Net_output','Net_bidirection','Net_short'],
            'reference_independently_certified':False,'assistant_visual_supplement_sha256':sha(visual_path),
            'assistant_visual_supplement_applied':[]}
    for case in cases:
        row=manifests[case];source_path=Path(row['raw_annotation_path'])
        if sha(source_path)!=row['annotation_sha256']:raise ValueError('Source annotation changed')
        raw,_=load_source_annotation(source_path);converter=Converter(raw,case,policy,overrides.get(case));converter.run()
        base=physical['cases'][case];mapping=deepcopy(base['reference_pin_networks']);labels=defaultdict(list);network=nx.Graph()
        network.add_nodes_from(mapping.values())
        for cid,component in converter.components.items():
            name=component['logical_label']
            if component['source_category'] not in result['source_label_types'] or not name or component['is_power'] or power_name(name,cfg):continue
            pins=[f'{cid}/{pid}' for c,pid in converter.pins if c==cid]
            labels[name].append({'component_source_id':cid,'source_category':component['source_category'],'source_pin_ids':pins,
                                 'points':[list(converter.pins[cid,pid]['raw_tip']) for c,pid in converter.pins if c==cid]})
        def merged_reference(site_map):
            connections=nx.Graph();connections.add_nodes_from(mapping.values())
            for label,sites in site_map.items():
                if len(sites)<2:continue
                groups=sorted({mapping[p] for site in sites for p in site['source_pin_ids'] if p in mapping})
                for g in groups[1:]:connections.add_edge(groups[0],g,label=label)
            roots={old:f'REF_LOGICAL_{i+1:05d}' for i,group in enumerate(sorted(nx.connected_components(connections),key=lambda g:min(g))) for old in group}
            return {pin:roots[nid] for pin,nid in mapping.items()}
        raw_mapping=merged_reference(labels)
        visual_case=visual['cases'].get(case)
        if visual_case:
            if visual_case['image_sha256']!=base['image_sha256']:raise ValueError('Visual supplement image changed')
            for review in visual_case['component_label_roles']:
                cid=review['component_source_id'];name=review['label']
                if cid not in converter.components or not review['reviewed'] or power_name(name,cfg):raise ValueError('Invalid visual label role')
                source_text=raw['components'][cid]['associatedTexts'][review['source_text_id']]
                if source_text['content']!=name:raise ValueError('Supplement must use existing source text spelling')
                pins=[f'{cid}/{pid}' for c,pid in converter.pins if c==cid]
                if set(pins)!={f'{cid}/{pid}' for pid in review['source_pin_ids']}:raise ValueError('Reviewed source pin set changed')
                labels[name].append({'component_source_id':cid,'source_category':converter.components[cid]['source_category'],
                  'source_pin_ids':pins,'points':[list(converter.pins[cid,pid]['raw_tip']) for c,pid in converter.pins if c==cid],
                  'label_role_source':'assistant_visual_development_review'})
                result['assistant_visual_supplement_applied'].append({'case_id':case,'component_source_id':cid,'label':name})
        augmented_mapping=merged_reference(labels)
        result['cases'][case]={'image_sha256':base['image_sha256'],'source_annotation_sha256':row['annotation_sha256'],
              'reference_pin_networks':augmented_mapping,'raw_reference_pin_networks':raw_mapping,'ordinary_label_sites':dict(labels),
              'logical_label_groups':sum(len(sites)>=2 for sites in labels.values())}
    save(output/'reference_logical_connectivity.json',result);return result


def review_crops(output,cases):
    records=[];font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',12)
    visual=read(ROOT/'data/reviews/network_labels_logical_v1.json')
    for case in cases:
        logical=read(output/f'predictions/{case}/logical_graph_proposals.json')
        applied={sid for d in logical['label_merge_decisions'] if not d['withheld_reasons'] for sid in d['site_ids']}
        with Image.open(ROOT/f'data/standardized/{case}/{case}.png') as source:
            for site in logical['label_sites']:
                if site['status'] not in {'bound_ordinary_label','endpoint_text_role_unverified'}:continue
                bb=site['bbox'];point=site['point']
                crop=[max(0,int(min(bb[0],point[0])-25)),max(0,int(min(bb[1],point[1])-25)),
                      min(source.width,int(max(bb[2],point[0])+26)),min(source.height,int(max(bb[3],point[1])+26))]
                patch=source.crop(crop).convert('RGB');patch.thumbnail((225,165));card=Image.new('RGB',(240,210),'white')
                card.paste(patch,((240-patch.width)//2,25));draw=ImageDraw.Draw(card)
                draw.text((4,2),case+' '+site['site_id'],font=font,fill='#333')
                draw.text((4,176),site['label'],font=font,fill='#8d239f' if site['site_id'] in applied else '#a26200')
                draw.text((4,193),'MERGE PROPOSAL' if site['site_id'] in applied else 'BINDING ONLY',font=font,fill='#333')
                name=case+'_'+site['site_id'];path=output/'review_crops'/f'{name}.png';path.parent.mkdir(parents=True,exist_ok=True);card.save(path)
                reviewed=next((r for r in visual['cases'].get(case,{}).get('label_binding_reviews',[]) if r['site_id']==site['site_id']
                               and r['expected_text']==site['label'] and visual['cases'][case]['image_sha256']==logical['image_sha256']),None)
                records.append({'review_id':name,'case_id':case,'site_id':site['site_id'],'predicted_text':site['label'],'point':point,
                    'bbox_top_left':bb,'source_image_sha256':logical['image_sha256'],'candidate_card':f'review_crops/{name}.png',
                    'merge_applied':site['site_id'] in applied,'reviewed':bool(reviewed),'correct_text_and_role':reviewed['correct_text_and_role'] if reviewed else None,
                    'correct_physical_network':None,'reviewer':'assistant_visual_inspection' if reviewed else None,
                    'review_scope':'Text/role review on a deliberately selected development example; physical membership not exhaustively certified'})
    for offset in range(0,len(records),24):
        rows=records[offset:offset+24];sheet=Image.new('RGB',(6*240,((len(rows)+5)//6)*210),'white')
        for i,row in enumerate(rows):
            with Image.open(output/row['candidate_card']) as card:sheet.paste(card,((i%6)*240,(i//6)*210))
        sheet.save(output/f'label_review_{offset//24+1:02d}.png')
    save(output/'label_review_queue.json',{'records':records,'count':len(records),'reviewed':sum(r['reviewed'] for r in records),'automatic_label_changes':0})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/s9_network_labels_v1')
    parser.add_argument('--reuse-references',action='store_true');args=parser.parse_args();output=args.output
    batch=read(output/'batch_summary.json');case_ids=[r['case_id'] for r in batch['cases']];cfg=read(ROOT/'configs/network_labels_v1.json')
    reference=read(output/'reference_logical_connectivity.json') if args.reuse_references else reference_snapshot(output,case_ids,cfg)
    fields=read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json');totals=defaultdict(Counter);raw_totals=defaultdict(Counter);counts=Counter();decisions=[];cases=[]
    for case in case_ids:
        pin=read(ROOT/f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json');ref=reference['cases'][case]
        if ref['image_sha256']!=pin['image_sha256'] or fields['image_sha256'][case]!=pin['image_sha256']:raise ValueError('Reference image mismatch')
        truth=[r for r in fields['references'] if r['case_id']==case]
        owner_map=component_matches([{'bbox':c['bbox_top_left'],'class_name':c['class_name']} for c in pin['components']],fields['detector_references'][case]['objects'])
        record={'case_id':case}
        for variant,path in [('physical',ROOT/f'runs/s7_wire_graph_v1/predictions/{case}/graph_proposals.json'),('logical',output/f'predictions/{case}/logical_graph_proposals.json')]:
            graph=read(path)
            if graph['image_sha256']!=pin['image_sha256']:raise ValueError('Graph image mismatch')
            score=pair_counts(graph,pin,truth,ref['reference_pin_networks'],owner_map);record[variant]=score
            raw_score=pair_counts(graph,pin,truth,ref['raw_reference_pin_networks'],owner_map);record[variant+'_raw_draft_reference']=raw_score
            for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']:
                for field in ['tp','fp','fn']:totals[variant][key+'_'+field]+=score[key][field]
                for field in ['tp','fp','fn']:raw_totals[variant][key+'_'+field]+=raw_score[key][field]
            totals[variant]['localized_owned_pins']+=score['localized_owned_pins']
            if variant=='logical':
                counts.update(graph['summary']['label_site_status_counts'])
                for key in ['logical_label_merges','logical_networks_merged','flywire_network_proposals','network_proposals','attached_pins']:
                    counts[key]+=graph['summary'][key]
                counts['pin_candidates']+=len(graph['pin_attachments'])
                coverage=read(path.parent/'export_coverage.json')
                for key in ['exported_components','exported_pins','exported_nets']:counts[key]+=coverage[key]
                decisions.extend({'case_id':case,**d} for d in graph['label_merge_decisions'])
                base=read(ROOT/f'runs/s7_wire_graph_v1/predictions/{case}/graph_proposals.json')
                if graph['edges']!=base['edges'] or [(p['candidate_id'],p['point']) for p in graph['pin_attachments']]!=[(p['candidate_id'],p['point']) for p in base['pin_attachments']]:
                    raise ValueError('Logical label step changed physical edges or pin tips')
        cases.append(record)
    aggregate={}
    for variant,c in totals.items():
        aggregate[variant]={'localized_owned_pins':c['localized_owned_pins']}
        for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']:aggregate[variant][key]=metrics(*(c[key+'_'+f] for f in ['tp','fp','fn']))
    raw_aggregate={}
    for variant,c in raw_totals.items():
        raw_aggregate[variant]={key:metrics(*(c[key+'_'+f] for f in ['tp','fp','fn'])) for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']}
    result={'cases':cases,'aggregate':aggregate,'raw_source_reference_aggregate':raw_aggregate,'counts':dict(counts),'decisions':decisions,'official_NetF1':None,
            'assistant_visual_supplement':reference['assistant_visual_supplement_applied'],
            'metric_scope':'Both physical and logical graphs scored against the SAME new ordinary-label draft reference; not directly comparable to the earlier physical-only reference score.',
            'reference_limits':'Source labels, edge grouping and pin tips are draft and may be incomplete. Not official Golden or independent engineering testing.',
            'engineering_test_independent':False,'power_merging':0,'cross_image_merging':0,'fuzzy_label_matching':0,
            'physical_edges_unchanged':True,'all_pin_coordinates_preserved':True,'config_sha256':sha(ROOT/'configs/network_labels_v1.json')}
    save(output/'evaluation.json',result);review_crops(output,case_ids)
    def pct(x):return '—' if x is None else f'{x*100:.2f}%'
    lines=['# 普通网络标签关联与逻辑飞线 v1','',
        '新增流程：冻结物理导线图＋RapidOCR 原图文字 → 网络标签与连接位置关联 → 同名普通标签逻辑网络候选。默认基于 v1 物理图，以便单独观察标签变化；交点 CNN 仍是可选试验。','',
        '赛题第21–24页明确：同名分散普通标签可形成逻辑网络；hyperGraph需要包含标签自身和相关元件引脚；示例使用标签连接处的零长度edges。VCC/GND等电源网络不按普通飞线处理。',
        '原PDF SHA256：45eedbc195ad71bbb03044c4606ea85ccf3eccebbb3d9b3304d6e325e1acc5e8。示例几何已核对，但具体分段与完整出口仍需官方Golden校准。','',
        '## 本轮完成','',
        f"26页：保留 {counts['pin_candidates']} 个原引脚候选和全部物理线段；带符号证据的普通标签绑定候选 {counts['bound_ordinary_label']}，端点旁文字角色未决 {counts['endpoint_text_role_unverified']}，电源关联待处理 {counts['power_semantics_pending']}。",
        f"自动应用 {counts['logical_label_merges']} 组普通同名逻辑候选，减少 {counts['logical_networks_merged']} 个物理网络分组。没有新增远距离物理导线、移动引脚、跨图合并或模糊名称替换。",'',
        '普通文字＋端点只能形成候选，不能自动合并。初版把0026的说明词Differential误认作同名网名，视觉检查后新增符号证据门槛，当前已阻止该误合并。',
        'Net_short候选只在模型预测有两个相对的引脚簇、邻近实体元件能确定外部文字侧时使用；属于局部语义假设，须继续复核。重复OCR裁剪不算多个标签位置。',
        'OCR拼写保持原样和备选。存在争议时，至少3种已有OCR视图支持某个完整拼写、且领先其他拼写至少2个视图，才选用该已出现的拼写。视图来自同一个OCR模型，不能视为统计独立证据。空格、下划线或大小写不同的网名不自动合并。','',
        '## 草稿连接指标','',
        '**本表两种方案均使用普通标签逻辑草稿参考，并加入0170一处助手视觉角色复核；不能直接与上一阶段物理参考指标混比，也不是官方NetF1。**',
        '0170的源C_114关联T_42已经写有ADC_IN12，但details为空，转换器不能把它当作正式netLabel。原PNG复核确认它与底部C_72同名相连，因此另存视觉角色补充。没有更改原数据、源tip、原训练标注，也没有把参考输入预测流程。','',
        '| 范围 | 方案 | P | R | F1 | TP / FP / FN |','|---|---|---:|---:|---:|---|']
    for key,label in [('conditional_connectivity_pairs','已匹配归属/位置引脚对'),('all_reference_pair_proxy','全量参考引脚对代理')]:
        for variant,title in [('physical','仅物理图'),('logical','增加普通标签')]:
            s=aggregate[variant][key];lines.append(f"| {label} | {title} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} | {s['tp']} / {s['fp']} / {s['fn']} |")
    lines+=['','为保留原始证据，未补充标签角色时的分数也保存在 `raw_source_reference_aggregate`：原参考把新增连接计为14个误配对，条件F1从75.43%到74.74%。该变化反映此例标签角色缺失，不能据此判断真实飞线错误或泛化提升。视觉补充只核对这一例，不能当成全量标签标注认证。']
    lines+=['','## 当前合并与未决项','']
    for d in decisions:
        if not d['withheld_reasons']:lines.append(f"- {d['case_id']}：`{d['label']}`，标签位置 {len(d['site_ids'])} 个，物理分组 {len(d['physical_network_ids'])} 个。当前为模型候选，不是已认证比赛网络。")
    lines+=['',f"严格部分出口：{counts['exported_components']} 个元件、{counts['exported_pins']} 个字段齐全引脚、{counts['exported_nets']} 条网络。不会删除未决标签pin来拼出看似完整的hyperGraph。",'',
        '主要缺口：OCR仍会丢失下划线；标签检测和引脚/导线关联有漏项；电源逻辑尚未合并；完整标签组件身份、编号和pinname未恢复；交点和断线还有未决项。',
        '零长度飞线几何只在严格出口资料齐全时生成。中间结果保留原物理edges并单列logical_label_sites供调试。','',
        '## 文件与运行','',
        '`label_bindings.json` 保存原始OCR、拼写票数、锚点和候选；`logical_graph_proposals.json` 保存物理网络与逻辑网络对应、每次合并原因；叠加图紫框为已应用逻辑候选，橙框为绑定待决，灰框为电源待决。',
        '`label_review_queue.json` 和 `label_review_*.png` 可逐点复核；它们不能自动成为训练标注或官方参考。',
        '在桌面项目运行 `powershell -File scripts/run_network_labels.ps1 -Image data/standardized/0170/0170.png -Device cuda:0`。默认完整PNG入口，不读取CAD/标注。',
        '离线重放：`python scripts/batch_network_labels.py`，再执行 `python scripts/evaluate_network_labels.py --reuse-references`。评估才读取参考快照；生产推理不导入评估脚本。','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'counts':dict(counts),'aggregate':aggregate})


if __name__=='__main__':main()
