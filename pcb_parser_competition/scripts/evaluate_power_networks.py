"""Offline power-caption draft reference; never imported by PNG inference."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from pathlib import Path
import json

import networkx as nx

from run_power_networks import ROOT,read,save,sha
from evaluate_wire_graph import pair_counts,metrics
from evaluate_pin_text import component_matches
from pcb_parser.power_networks import qualified_name


def make_reference(output,cases):
    from pcb_parser.dataset_converter import load_source_annotation
    base=read(ROOT/'runs/s9_network_labels_v1/reference_logical_connectivity.json')
    manifest={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    cfg=read(ROOT/'configs/power_networks_v1.json');result={'cases':{},'scope':'Ordinary-label draft plus exact explicitly captioned source Gnd/V groups. Not official Golden.',
         'uncaptioned_ground_category_not_merged':True,'base_reference_sha256':sha(ROOT/'runs/s9_network_labels_v1/reference_logical_connectivity.json'),
         'inference_reads_this_reference':False,'independently_certified':False}
    for case in cases:
        raw_path=Path(manifest[case]['raw_annotation_path'])
        if sha(raw_path)!=manifest[case]['annotation_sha256']:raise ValueError('Source annotation changed')
        source,_=load_source_annotation(raw_path)
        mapping=deepcopy(base['cases'][case]['reference_pin_networks']);by_name=defaultdict(list);held=[]
        for cid,c in source['components'].items():
            kind={'Gnd':'gnd','V':'vcc'}.get(c.get('category'))
            if not kind:continue
            names={str(t.get('content','')) for t in c.get('associatedTexts',{}).values()
                   if qualified_name(str(t.get('content','')),kind,cfg)}
            pids=[f'{cid}/{pid}' for pid in c.get('pins',{}) if f'{cid}/{pid}' in mapping]
            if len(names)!=1 or len(pids)!=1:
                held.append({'component_source_id':cid,'caption_names':sorted(names),'mapped_pin_count':len(pids)});continue
            name=next(iter(names));by_name[name].append({'component_source_id':cid,'type':kind,'source_pin_ids':pids,
                                                       'caption':name,'caption_from_explicit_associated_text':True})
        connected=nx.Graph();connected.add_nodes_from(mapping.values())
        for name,sites in by_name.items():
            nodes=sorted({mapping[p] for s in sites for p in s['source_pin_ids']})
            if len({s['type'] for s in sites})>1:continue
            for n in nodes[1:]:connected.add_edge(nodes[0],n,label=name)
        roots={n:f'REF_POWER_{i+1:05d}' for i,group in enumerate(sorted(nx.connected_components(connected),key=lambda g:min(g))) for n in group}
        result['cases'][case]={'image_sha256':base['cases'][case]['image_sha256'],'source_annotation_sha256':sha(raw_path),
                'reference_pin_networks':{p:roots[n] for p,n in mapping.items()},'power_caption_sites':dict(by_name),
                'held_power_components':held,'power_label_groups':sum(len(s)>=2 for s in by_name.values())}
    save(output/'reference_power_connectivity.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s15_power_networks');p.add_argument('--reuse-references',action='store_true');a=p.parse_args()
    batch=read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('Incomplete power model batch')
    cases=[r['case_id'] for r in batch['cases']]
    ref=read(a.output/'reference_power_connectivity.json') if a.reuse_references else make_reference(a.output,cases)
    fields=read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json');old_ref=read(ROOT/'runs/s9_network_labels_v1/reference_logical_connectivity.json')
    totals={v:Counter() for v in ['baseline_s14','power']};original_totals={v:Counter() for v in totals};rows=[];group_reviews=[]
    for case in cases:
        f=ROOT/f'runs/s14_component_library/predictions/{case}';pins=read(f/'pin_text/pin_text_result.json')
        old=read(f/'network/logical_graph_proposals.json');new=read(a.output/f'predictions/{case}/power_graph_proposals.json')
        if pins['image_sha256']!=ref['cases'][case]['image_sha256'] or new['image_sha256']!=pins['image_sha256']:
            raise ValueError('Reference PNG differs')
        truth=[t for t in fields['references'] if t['case_id']==case]
        owner_map=component_matches([{'class_name':c['class_name'],'bbox':c['bbox_top_left']} for c in pins['components']],fields['detector_references'][case]['objects'])
        record={'case_id':case}
        for version,graph in [('baseline_s14',old),('power',new)]:
            values=pair_counts(graph,pins,truth,ref['cases'][case]['reference_pin_networks'],owner_map)
            raw=pair_counts(graph,pins,truth,old_ref['cases'][case]['reference_pin_networks'],owner_map)
            record[version]=values
            for key in ['conditional_connectivity_pairs','all_reference_pair_proxy']:
                for metric in ['tp','fp','fn']:
                    totals[version][key+'_'+metric]+=values[key][metric];original_totals[version][key+'_'+metric]+=raw[key][metric]
        rows.append(record)
        for d in new['power_merge_decisions']:
            if not d['withheld_reasons']:group_reviews.append({'case_id':case,**d,'visually_reviewed':False,'complete_network_reference_verified':False})
    aggregate=lambda ts:{v:{k:metrics(*(c[k+'_'+m] for m in ['tp','fp','fn'])) for k in ['conditional_connectivity_pairs','all_reference_pair_proxy']} for v,c in ts.items()}
    result={'cases':rows,'aggregate':aggregate(totals),'old_ordinary_only_reference_aggregate':aggregate(original_totals),
            'applied_power_groups':group_reviews,'reference_sha256':sha(a.output/'reference_power_connectivity.json'),
            'metric_scope':'Both S14 and S15 scored against the same new power-caption draft reference; not directly comparable to earlier ordinary/physical-only scores.',
            'reference_limits':'Only explicit unique source power captions are merged; uncaptained grounds excluded from power merging. Source tips/wires are draft and not independently certified.',
            'engineering_test_independent':False,'official_NetF1':None,'new_training':False}
    save(a.output/'evaluation.json',result);print(result['aggregate'],flush=True)


if __name__=='__main__':main()
