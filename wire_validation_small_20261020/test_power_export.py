"""Frozen-cache small-set test; transformations never receive GT annotations."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'wire_validation_20261020'
sys.path.insert(0,str(PREVIOUS))
from validate import read,metrics,agent_module,TRUTH
from compare import geometry_counts,robust_wire
agent=agent_module(PREVIOUS/'runtime')
from adapter import validate_target


def cached_inputs(case):
    digest=hashlib.sha256(Path(f'D:/pcb_open_gt_in/{case}.png').read_bytes()).hexdigest()
    for work in (PREVIOUS/'candidate_path/work').iterdir():
        path=work/'p/export/engineering_graph_proposals.json'
        graph=read(path)
        if graph['image_sha256']==digest:
            pins=read(work/'p/export/engineering_pin_text_result.json')
            pred=read(PREVIOUS/f'candidate_path/final_results/{case}_Pred.json')
            return pred,graph,pins
    raise ValueError('Missing frozen case '+case)


def restore_power_nets(base,graph,pins,add_power_contacts=False,split_physical=False):
    """Restore only existing accepted same-caption power groups and model edges."""
    out=copy.deepcopy(base)
    pmap={p['candidate_id']:p for p in pins['pin_candidates']}
    emap={e['edge_id']:e for e in graph['edges']}
    sites={s['site_id']:s for s in graph.get('power_sites',[])}
    attachments={a['candidate_id']:a for a in graph['pin_attachments']}
    accepted_labels={d['label'] for d in graph.get('power_merge_decisions',[])
                     if not d['withheld_reasons'] and d['status']=='power_logical_connectivity_proposal'}
    height=graph['size'][1]
    audit=[]
    candidates=[]
    for net in graph['networks']:
        if not split_physical or not net.get('power_logical_merge'):
            candidates.append(net)
            continue
        edge_ids=set(net['edge_ids'])
        for physical in graph['physical_networks']:
            if not physical['edge_ids'] or not set(physical['edge_ids'])<=edge_ids:
                continue
            site_ids=[sid for sid in net.get('power_site_ids',[]) if sites[sid]['candidate_id'] in physical['pin_candidate_ids']]
            if not site_ids:
                continue
            candidates.append({**physical,'network_id':'RESTORED_'+physical['network_id'],
                'power_logical_merge':True,'power_names':net['power_names'],'power_site_ids':site_ids})
    for net in candidates:
        names=net.get('power_names',[])
        if (not net.get('power_logical_merge') or net['network_id'] in out['nets']
                or len(names)!=1 or names[0] not in accepted_labels or net['unresolved_junction_ids']):
            continue
        pending={}
        if add_power_contacts:
            for sid in net.get('power_site_ids',[]):
                site=sites[sid];pin=pmap[site['candidate_id']]
                owner=pin.get('owner');component=out['components'].get(owner,{})
                if (site['withheld_reasons'] or site['label']!=names[0]
                        or pin.get('owner_status')!='assigned'
                        or site['component_id']!=owner
                        or component.get('type') not in {'gnd','v','vcc'}
                        or str(component.get('value') or '').upper()!=names[0].upper()
                        or pin.get('number') is None
                        or attachments[pin['candidate_id']]['attachment_status']!='attached'):
                    continue
                key='pin_'+str(pin['number'])
                if key not in out['pins'].get(owner,{}):
                    x,y=pin['point']
                    pending[(owner,key)]={'pinname':pin.get('pinname') or '',
                                          'point':{'x':float(x),'y':float(height-y)}}
        refs=[]
        vertices=set()
        for cid in net['pin_candidate_ids']:
            pin=pmap[cid];owner=pin.get('owner');number=pin.get('number')
            if owner not in out['components'] or number is None:
                continue
            key='pin_'+str(number)
            if key not in out['pins'].get(owner,{}) and (owner,key) not in pending:
                continue
            refs.append(f'{owner}.{number}')
            component=out['components'][owner]
            if component['type'] in {'box','ic','connector','raw_block','block'}:
                vertices.add((owner,str(number)))
            elif component['type'] in {'gnd','v','vcc'}:
                vertices.add((component['type'],str(component.get('value'))))
            else:
                vertices.add((owner,))
        if len(vertices)<2:
            continue
        for (owner,key),pin in pending.items():
            out['pins'].setdefault(owner,{})[key]=pin
        edges={}
        for eid in net['edge_ids']:
            polyline=emap[eid]['polyline']
            for a,b in zip(polyline,polyline[1:]):
                edges[f'edge_{len(edges)+1}']=[{'x':float(a[0]),'y':float(height-a[1])},
                                             {'x':float(b[0]),'y':float(height-b[1])}]
        restored={'hyperGraph':'('+','.join(sorted(set(refs)))+')','edges':edges}
        local={'nets':{net['network_id']:restored}}
        agent.cleanup_wire_edges(local)
        out['nets'][net['network_id']]=local['nets'][net['network_id']]
        audit.append({'network_id':net['network_id'],'observed_rail':names[0],
                      'members':len(set(refs)),'added_power_contacts':len(pending),
                      'edges':len(out['nets'][net['network_id']]['edges']),
                      'source_power_sites':net['power_site_ids']})
    assert out['components']==base['components']
    for owner,old in base['pins'].items():
        assert all(out['pins'][owner][k]==v for k,v in old.items())
    assert all(out['nets'][k]==v for k,v in base['nets'].items())
    validate_target(out)
    return out,audit


def evaluate(cases,report):
    rows=[]
    for case in cases:
        baseline,graph,pins=cached_inputs(case)
        variants={'baseline':(baseline,[])}
        for name,contacts,physical in [('restore_existing_members',False,False),
                ('restore_observed_power_contacts',True,False),
                ('restore_physical_existing',False,True),('restore_physical_contacts',True,True)]:
            variants[name]=restore_power_nets(baseline,graph,pins,contacts,physical)
        # Ground truth enters only here, after every candidate has been constructed.
        truth=read(next((TRUTH/case).glob('*.json')))
        record={'case':case,'variants':{}}
        for name,(prediction,audit) in variants.items():
            destination=ROOT/name
            destination.mkdir(exist_ok=True)
            (destination/f'{case}_Pred.json').write_text(json.dumps(prediction,ensure_ascii=False,indent=2),encoding='utf-8')
            scored=metrics.score_case(prediction,truth)
            record['variants'][name]={'metrics':{k:scored[k]['f1'] for k in metrics.WEIGHTS},
                'score_100':scored['final_score']*100,'geometry_f1':geometry_counts(prediction,truth)['f1'],
                'one_to_one_network_wire_f1':robust_wire(prediction,truth),'restored':audit}
        rows.append(record)
    names=rows[0]['variants']
    macro={n:{'metrics':{k:sum(r['variants'][n]['metrics'][k] for r in rows)/len(rows) for k in metrics.WEIGHTS},
              'score_100':sum(r['variants'][n]['score_100'] for r in rows)/len(rows),
              'geometry_f1':sum(r['variants'][n]['geometry_f1'] for r in rows)/len(rows),
              'one_to_one_network_wire_f1':sum(r['variants'][n]['one_to_one_network_wire_f1'] for r in rows)/len(rows)}
           for n in names}
    result={'cases':cases,'macro':macro,'per_case':rows,
            'starting_point':'Previously validated path-preserving test copy; deployment unchanged.',
            'neural_inference_calls':0,'new_training':False,'gt_used_by_transformation':False}
    (ROOT/report).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':cases,'macro':macro,'per_case':[{'case':r['case'],'variants':{n:{'wire':v['metrics']['wire_f1'],'conn':v['metrics']['connectivity_f1'],'score':v['score_100'],'restored_nets':len(v['restored'])} for n,v in r['variants'].items()}} for r in rows]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='+',default=['5','6','13'])
    parser.add_argument('--report',default='small_set_results.json')
    args=parser.parse_args();evaluate(args.cases,args.report)
