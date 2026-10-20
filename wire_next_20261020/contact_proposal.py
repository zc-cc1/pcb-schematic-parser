"""Isolated candidate: recover observed power contacts on existing model nets."""
from collections import defaultdict
from copy import deepcopy


def propose(base, graph, pins, allow_explicit_duplicates=False):
    if (graph.get('image_sha256') != pins.get('image_sha256') or not graph.get('image_sha256')
            or graph.get('size') != pins.get('size')):
        raise ValueError('Mismatched current-image artifacts')
    out=deepcopy(base)
    owners=defaultdict(list)
    for pin in pins['pin_candidates']:
        if pin.get('owner_status')=='assigned':owners[pin.get('owner')].append(pin)
    attachments={a['candidate_id']:a for a in graph['pin_attachments']}
    networks={n['network_id']:n for n in graph['networks']}
    height=graph['size'][1]
    audit=[]
    for owner,candidates in owners.items():
        component=out['components'].get(owner,{})
        if component.get('type') not in {'gnd','v','vcc'} or not component.get('value'):
            continue
        primary=[p for p in candidates if not p.get('text_hypothesis_primary_candidate_id')]
        if len(primary)!=1:continue
        pin=primary[0]
        cid=pin['candidate_id']
        if len(candidates)>1:
            if not allow_explicit_duplicates:continue
            if any(p is not pin and (p.get('pinname_status')!='duplicate_tip_hypothesis'
                   or p.get('text_hypothesis_primary_candidate_id')!=cid) for p in candidates):continue
        attachment=attachments.get(cid,{})
        nid=attachment.get('network_id')
        if (pin.get('number') is None or attachment.get('attachment_status')!='attached'
                or nid not in out['nets'] or attachment.get('point')!=pin['point']
                or cid not in networks.get(nid,{}).get('pin_candidate_ids',[])):
            continue
        ref=f"{owner}.{pin['number']}"
        members=lambda n:{m.strip() for m in n['hyperGraph'].strip('()').split(',') if m.strip()}
        if any(ref in members(n) for n in out['nets'].values()):continue
        key='pin_'+str(pin['number'])
        x,y=pin['point']
        out['pins'].setdefault(owner,{}).setdefault(key,{'pinname':pin.get('pinname') or '',
                                                      'point':{'x':float(x),'y':float(height-y)}})
        out['nets'][nid]['hyperGraph']='('+','.join(sorted(members(out['nets'][nid])|{ref}))+')'
        audit.append({'network_id':nid,'owner':owner,'number':str(pin['number']),
                      'rail':component['value'],'candidate_id':cid,'duplicate_count':len(candidates)-1})
    return out,audit
