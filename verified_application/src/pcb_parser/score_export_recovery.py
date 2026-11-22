"""Late additive contact export. No labels, GT, model or OCR calls."""
from collections import defaultdict
from copy import deepcopy
import math
import re
from .export_resilience import valid_point

IC_TYPES = {'box','ic','connector','block','raw_block'}
PASSIVE_TYPES = {'r','c','l','resistor','capacitor','inductor'}


def _number(value):
    return str(value) if value is not None and re.fullmatch(r'[^.,()\s]+',str(value)) else None


def _confidence(value):
    return float(value) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else 0.0


def recover_export_contacts(target, pins, graph, profile='none'):
    if profile not in {'none','ic','passive','combined'}:raise ValueError('Unknown score export profile: '+str(profile))
    out = deepcopy(target)
    audit = {'profile':profile,'added':[],'skipped':[],'existing_fields_changed':0,'networks_merged':0}
    if profile == 'none':return out,audit
    size = pins['size']
    pmap = {p['candidate_id']:p for p in pins['pin_candidates']}
    membership = defaultdict(set)
    for nid, net in target['nets'].items():
        for ref in net['hyperGraph'].strip('()').split(','):
            if ref.strip():membership[ref.strip()].add(nid)
    raw_owners = defaultdict(set)
    for net in graph['networks']:
        for cid in net['pin_candidate_ids']:raw_owners[cid].add(net['network_id'])
    reference_raw_nets = defaultdict(set)
    for cid,p in pmap.items():
        number=_number(p.get('number'))
        if number is not None:
            reference_raw_nets[f"{p.get('owner')}.{number}"].update(raw_owners[cid])
    net_targets = {}
    for net in graph['networks']:
        anchors = set()
        for cid in net['pin_candidate_ids']:
            p=pmap.get(cid,{})
            number=_number(p.get('number'))
            ref=f"{p.get('owner')}.{number}"
            if number is not None and len(reference_raw_nets[ref])==1:
                anchors.update(membership.get(ref,set()))
        if len(anchors)==1:net_targets[net['network_id']]=next(iter(anchors))
    def skip(p,reason):audit['skipped'].append({'candidate_id':p['candidate_id'],'reason':reason})
    groups = defaultdict(list)
    reserved = defaultdict(set)
    for owner, existing in target['pins'].items():reserved[owner].update(key[4:] for key in existing)
    for p in pins['pin_candidates']:
        if _number(p.get('number')) is not None:reserved[p.get('owner')].add(_number(p['number']))
    for p in sorted(pins['pin_candidates'],key=lambda p:p['candidate_id']):
        owner=p.get('owner');kind=str(target['components'].get(owner,{}).get('type',''))
        number=_number(p.get('number'))
        is_ic=kind in IC_TYPES
        if not ((is_ic and profile in {'ic','combined'}) or (kind in PASSIVE_TYPES and profile in {'passive','combined'})):continue
        if number is not None and 'pin_'+number in target['pins'].get(owner,{}):continue
        if p.get('owner_status')!='assigned' or not valid_point(p.get('point'),size) or _confidence(p.get('confidence'))<.85:
            skip(p,'unreliable_owner_point_or_confidence');continue
        if number is None and is_ic:
            skip(p,'ic_number_unknown');continue
        raw=raw_owners[p['candidate_id']]
        if len(raw)!=1 or next(iter(raw)) not in net_targets:
            skip(p,'ambiguous_or_unanchored_network');continue
        nid=net_targets[next(iter(raw))]
        generated=number is None
        if generated:
            number=str(next(n for n in range(1,len(reserved[owner])+2) if str(n) not in reserved[owner]))
            reserved[owner].add(number)
        groups[(owner,number)].append((p,nid,generated))
    for (owner,number),candidates in sorted(groups.items()):
        # Same-number candidates on different final nets cannot safely be resolved here.
        if len({nid for _,nid,_ in candidates}) != 1:
            for p,_,_ in candidates:skip(p,'conflicting_duplicate_number')
            continue
        def quality(item):
            p=item[0];e=p.get('text_evidence',{})
            return (bool(p.get('pinname')), _confidence(e.get('number',{}).get('confidence')), _confidence(e.get('pinname',{}).get('confidence')), _confidence(p.get('confidence')))
        candidates.sort(key=lambda item:item[0]['candidate_id'])
        chosen=max(candidates,key=quality)
        p,nid,generated=chosen
        for other in candidates:
            if other is not chosen:skip(other[0],'weaker_duplicate_number')
        ref=f'{owner}.{number}'
        if membership.get(ref):skip(p,'reference_already_connected');continue
        x,y=p['point']
        out['pins'].setdefault(owner,{})['pin_'+number]={'pinname':str(p.get('pinname') or ''),'point':{'x':x,'y':size[1]-y}}
        members=[r.strip() for r in out['nets'][nid]['hyperGraph'].strip('()').split(',') if r.strip()]
        members.append(ref);out['nets'][nid]['hyperGraph']='('+','.join(members)+')'
        membership[ref].add(nid)
        audit['added'].append({'candidate_id':p['candidate_id'],'reference':ref,'network_id':nid,'provisional_passive_number':generated,'pinname_known':bool(p.get('pinname'))})
    return out,audit
