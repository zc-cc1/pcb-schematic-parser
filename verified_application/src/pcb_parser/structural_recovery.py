"""Register image-verified dependent structural patterns as one atomic rule."""
from copy import deepcopy
import math
import re

from .repair_engine import Edit, Proposal, RuleSpec, RepairContext, RepairPipeline, SAFE_CAPABILITIES
from .structural_patterns import StructuralDraft, members, NC, RAILS, guarded_named_supply
from .numeric_power_caption import numeric_voltage
from .pin_numeric_connector import observations
from .terminal_topology import ground_branch_proof, no_connect_marker, isolated_nc_net
from .terminal_identity_recovery import capacitor_candidates, ground_proof
from .red_resistor_contacts import resistor_contact_proof
from .single_led_identity import single_led_proof

PATTERNS = ('opposed_inductor_endpoint', 'arrow_caption_identity', 'independent_led_symbols',
            'visible_terminal_pair', 'typed_power_contact', 'colored_wire_network',
            'outward_red_stem', 'zero_wire_fragment', 'numeric_power_caption',
            'unidentified_capacitor_identity', 'ground_owner_arbitration',
            'mixed_color_ground_branch', 'image_no_connect_marker', 'observed_resistor_terminals',
            'single_filled_led_identity')


class ImageStructurePolicy:
    capabilities = ('image_verified_structure',)

    def required_capability(self, edit):
        if len(edit.path)==2 and edit.path[0] in {'components','pins','nets'}:
            return 'image_verified_structure'
        raise ValueError('unsupported_structural_edit')

    def validate(self, before, after, context, proposal):
        events=proposal.evidence.get('pattern_changes',[])
        if not events:raise ValueError('missing_pattern_evidence')
        supported={e['owner'] for e in events if 'owner' in e}
        def finite_point(p):
            return isinstance(p,dict) and all(type(p.get(k)) in (int,float) and math.isfinite(p[k])
                and 0 <= p[k] <= bound for k,bound in [('x',context.width),('y',context.height)])
        if set(before['components'])-set(after['components']):raise ValueError('component_identity_deleted')
        for owner,comp in after['components'].items():
            old=before['components'].get(owner)
            if old and old['type'] in {'box','ic','switch','pin'}:
                if comp!=old or after['pins'].get(owner)!=before['pins'].get(owner):
                    raise ValueError('protected_component_or_pin_changed')
            changed=old!=comp or after['pins'].get(owner)!=before['pins'].get(owner)
            if not changed:continue
            if owner not in supported:raise ValueError('unexplained_component_change')
            if old and comp['type']!=old['type']:raise ValueError('existing_type_changed')
            if not old and comp['type'] not in {'led','net_bidirection'}:
                if comp['type']!='c' or not any(e['rule']=='unidentified_capacitor_identity' and e.get('owner')==owner for e in events):
                    raise ValueError('unsupported_new_identity')
            box=comp.get('bbox')
            if not isinstance(box,list) or len(box)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box):
                raise ValueError('invalid_component_box')
            if not (0<=box[0]<=box[2]<=context.width and 0<=box[1]<=box[3]<=context.height):
                raise ValueError('component_box_outside_image')
            pins=after['pins'].get(owner,{})
            if not pins or not all(finite_point(p.get('point')) for p in pins.values()):raise ValueError('invalid_structural_pin')
            if comp['type'] in {'r','c','l','led'} and set(pins)!={'pin_1','pin_2'}:
                raise ValueError('two_terminal_identity_not_proven')
        all_refs=set()
        for nid,net in after['nets'].items():
            refs=members(net)
            for ref in refs:
                try:
                    owner,num=ref.rsplit('.',1);pin=after['pins'][owner]['pin_'+num]
                    after['components'][owner]
                except (KeyError,ValueError):raise ValueError('dangling_structural_member') from None
                if str(pin.get('pinname','')).strip().upper() in NC:raise ValueError('explicit_nc_connected')
                all_refs.add(ref)
            if net==before['nets'].get(nid):continue
            for edge in net.get('edges',{}).values():
                if not isinstance(edge,list) or len(edge)<2 or not all(finite_point(p) for p in edge):
                    raise ValueError('invalid_structural_wire')
        for event in events:
            kind=event['rule']
            if kind=='single_filled_led_identity':
                data=context.evidence
                source=StructuralDraft(before,data['rgb'],data['records'],data['pins'],data['ocr'],data['wire_cfg'],data['config'])
                record=source.rmap.get(event.get('source_record'))
                proof=single_led_proof(source,record) if record else None
                owner=event.get('owner');readings=event.get('caption_observations',[])
                qualified=[t for t in readings if t.get('confidence',0)>=.98 and t.get('view')
                           and re.fullmatch(r'LED[1-9]\d*',str(t.get('text','')).strip().upper())]
                names={str(t['text']).strip().upper() for t in qualified}
                views={t['view'] for t in qualified if str(t['text']).strip().upper()==owner}
                if (not proof or owner in before['components'] or names!={owner} or len(views)<2
                        or any(event.get(k)!=v for k,v in proof.items())
                        or any(t.get('bbox')!=proof['caption_box'] for t in qualified)):
                    raise ValueError('single_led_without_source_shape_and_local_caption')
                a,b,c,d=proof['body_box']
                expected={'type':'led','Name':owner,'value':None,'bbox':[a,context.height-d,c,context.height-b]}
                pins={'pin_'+number:source.export_pin(point,number)
                      for number,point in zip(proof['numbering'],proof['terminals'])}
                if after['components'].get(owner)!=expected or after['pins'].get(owner)!=pins:
                    raise ValueError('single_led_output_differs_from_observed_proof')
            if kind=='observed_resistor_terminals':
                data=context.evidence
                record=next((r for r in data['records']['records'] if r['component_id']==event.get('source_record')),None)
                proof=resistor_contact_proof(data['rgb'],record) if record else None
                owner=event.get('owner')
                expected={'pin_'+str(index):{'pinname':str(index),
                          'point':{'x':point[0],'y':context.height-point[1]}}
                          for index,point in enumerate(proof['terminals'],1)} if proof else None
                if (not proof or record['component_id']!=owner or before['pins'].get(owner)
                        or before['components'].get(owner,{}).get('type')!='r'
                        or any(event.get(k)!=v for k,v in proof.items())
                        or after['pins'].get(owner)!=expected):
                    raise ValueError('resistor_contacts_without_rectangle_and_stems')
            if kind in {'arrow_caption_identity','independent_led_symbols'} and len(set(event.get('views',[])))<2:
                raise ValueError('identity_without_multiview_text')
            if (kind=='typed_power_contact' and event.get('caption')!='GND'
                    and event.get('caption_kind')!='guarded_named_supply'
                    and len(set(event.get('views',[])))<3):
                raise ValueError('supply_without_multiview_text')
            if kind=='typed_power_contact' and event.get('caption_kind')=='guarded_named_supply':
                data=context.evidence
                records={r['component_id']:r for r in data['records']['records']}
                candidates={p['candidate_id']:p for p in data['pins']['pin_candidates']}
                candidate=candidates.get(event.get('candidate'))
                record=records.get(event.get('owner'))
                proof=(guarded_named_supply(observations(data['ocr']),data['rgb'],
                                            record['bbox_top_left'],candidate['point'])
                       if record and candidate and candidate.get('owner')==event.get('owner')
                       and candidate.get('owner_status')=='assigned' else None)
                if (not proof or proof['caption']!=event.get('caption')
                        or proof['views']!=event.get('views')
                        or proof['outward_wire']!=event.get('outward_wire')
                        or event.get('point')!=candidate['point']):
                    raise ValueError('guarded_supply_without_local_evidence')
            if kind=='typed_power_contact' and event.get('caption_kind')=='numeric_voltage':
                caption=numeric_voltage(event.get('caption'))
                readings=event.get('caption_observations',[])
                names={numeric_voltage(t.get('text')) for t in readings if t.get('confidence',0)>=.9}
                views={t.get('view') for t in readings if t.get('view') and t.get('confidence',0)>=.9
                       and numeric_voltage(t.get('text'))==caption}
                if (not caption or names!={caption} or len(views)<3
                        or event.get('candidate') not in context.evidence['candidate_ids']
                        or event.get('source_confidence',0)<.9 or not event.get('outward_wire')):
                    raise ValueError('numeric_supply_without_local_evidence')
            if kind in {'unidentified_capacitor_identity','ground_owner_arbitration'}:
                data=context.evidence
                source=StructuralDraft(before,data['rgb'],data['records'],data['pins'],data['ocr'],data['wire_cfg'],data['config'])
                if kind=='unidentified_capacitor_identity':
                    record=source.rmap.get(event.get('source_record'))
                    proof=capacitor_candidates(source,record) if record else None
                    readings=event.get('caption_observations',[]);name=event.get('owner')
                    names={str(t.get('text','')).strip().upper() for t in readings if t.get('confidence',0)>=.98
                           and re.fullmatch(r'C[1-9]\d*',str(t.get('text','')).strip().upper())}
                    views={t.get('view') for t in readings if t.get('confidence',0)>=.98 and t.get('view')
                           and str(t.get('text','')).strip().upper()==name}
                    if (not proof or names!={name} or len(views)<2 or name in before['components']
                        or event.get('shape')!=proof[1]
                        or set(event.get('source_candidates',[]))!={p['candidate_id'] for p in proof[0]}):
                        raise ValueError('capacitor_identity_without_source_evidence')
                    a,b,c,d=record['bbox_top_left'];expected={'type':'c','Name':name,'value':record.get('value') or '',
                                                          'bbox':[a,context.height-d,c,context.height-b]}
                    if after['components'].get(name)!=expected:raise ValueError('capacitor_identity_differs_from_source')
                    source.out['components'][name]=expected;source.rmap[name]=record
                    source.source_points[name]=[p['point'] for p in proof[0]]
                    pair=source.terminal_pair(name,record)
                    if not pair or event.get('terminals')!=pair[1]:raise ValueError('capacitor_terminal_geometry_unproven')
                    axis=proof[1]['axis'];ordered=sorted(proof[0],key=lambda p:p['point'][axis])
                    expected_pins={'pin_'+str(p['number']):source.export_pin(q,str(p['number']))
                                   for p,q in zip(ordered,event.get('terminals',[]))}
                    if after['pins'].get(name)!=expected_pins:raise ValueError('capacitor_terminals_differ_from_proof')
                else:
                    record=source.rmap.get(event.get('owner'))
                    candidate=next((p for p in source.pin_document['pin_candidates'] if p['candidate_id']==event.get('candidate')),None)
                    proof=ground_proof(source,record,candidate) if record and candidate else None
                    if not proof or any(event.get(k)!=v for k,v in proof.items()):
                        raise ValueError('ground_arbitration_without_source_evidence')
                    pins={'pin_1':source.export_pin(candidate['point'])}
                    if after['pins'].get(event['owner'])!=pins:raise ValueError('ground_contact_differs_from_source')
            if kind in {'mixed_color_ground_branch','image_no_connect_marker'}:
                data=context.evidence
                draft=StructuralDraft(after if kind=='mixed_color_ground_branch' else before,
                    data['rgb'],data['records'],data['pins'],data['ocr'],data['wire_cfg'],data['config'])
                if kind=='mixed_color_ground_branch':
                    pair=event.get('members',[])
                    proof=ground_branch_proof(draft,*pair) if len(pair)==2 else None
                    net=after['nets'].get(event.get('network_id'))
                    if (not proof or any(event.get(k)!=v for k,v in proof.items()) or not net or members(net)!=set(pair)
                        or proof['source_candidate'] not in data['candidate_ids']):
                        raise ValueError('mixed_ground_branch_without_source_evidence')
                else:
                    ref=event.get('member');proof=no_connect_marker(draft,ref) if ref else None
                    ids=event.get('network_ids',[])
                    if (not proof or any(event.get(k)!=v for k,v in proof.items()) or event.get('members')!=[ref]
                        or not ids or any(nid not in before['nets'] or nid in after['nets']
                           or not isolated_nc_net(draft,before['nets'][nid],proof) for nid in ids)
                        or any(ref in members(n) for n in after['nets'].values())):
                        raise ValueError('no_connect_removal_without_image_evidence')
            if kind=='colored_wire_network':
                if len(event.get('members',[]))<2 or not proposal.evidence.get('color_trace'):
                    raise ValueError('network_without_image_trace')
            if kind=='opposed_inductor_endpoint':
                if event.get('source_candidate') not in context.evidence['candidate_ids']:
                    raise ValueError('unobserved_inductor_contact')
        fragment_refs={r for e in events if e['rule'] in {'zero_wire_fragment','image_no_connect_marker'} for r in e['members']}
        for net in before['nets'].values():
            for ref in members(net)-all_refs-fragment_refs:
                owner,num=ref.rsplit('.',1)
                if before['components'][owner]['type'] in {'box','ic','switch','pin'}:
                    raise ValueError('protected_connection_lost')


class ImageStructuralRule:
    spec=RuleSpec('image_structural_patterns','1.3',
        '按形态识别端点、独立LED和标签，并由原图导线统一重建连接',
        tuple(sorted(SAFE_CAPABILITIES))+('image_verified_structure',))

    def propose(self, context):
        data=context.evidence
        draft=StructuralDraft(context.target,data['rgb'],data['records'],data['pins'],data['ocr'],
                              data['wire_cfg'],data['config'],data.get('reader'))
        after=draft.run()
        if data.get('cleanup'):
            for nid,net in after['nets'].items():
                if net.get('edges')==context.target['nets'].get(nid,{}).get('edges'):
                    continue
                # Stub removal can expose another degree-two joint. Normalize
                # newly reconstructed geometry to a fixed point once.
                observed_edges=deepcopy(net.get('edges',{}))
                for _ in range(len(net.get('edges',{}))+2):
                    old_edges=deepcopy(net.get('edges',{}))
                    data['cleanup']({'nets':{nid:net}})
                    if not net.get('edges') and observed_edges:
                        # A wholly observed short two-contact conductor is real
                        # evidence, even when all its segments are below 8 px.
                        net['edges']=observed_edges
                        break
                    if net.get('edges',{})==old_edges:break
                else:raise ValueError('wire_cleanup_did_not_converge')
        edits=[]
        for section in ('components','pins','nets'):
            for key in sorted(set(context.target[section])|set(after[section])):
                if context.target[section].get(key)==after[section].get(key):continue
                path=(section,key)
                edits.append(Edit.set(context.target,path,after[section][key]) if key in after[section]
                             else Edit.remove(context.target,path))
        evidence={'pattern_changes':draft.events,'withheld_patterns':draft.withheld,
                  'color_trace':getattr(draft,'graph_evidence',{}),'caption_reader_calls':draft.ocr_calls}
        rejection='structural_edit_budget' if len(edits)>data['config']['maximum_changes'] else ''
        yield Proposal(self.spec.rule_id,tuple(edits),evidence,
                       'dependent_image_evidence_patterns_committed_together',
                       rejection=(rejection if edits else 'no_effect'))


def recover_image_structures(target, rgb, records, pins, ocr, wire_cfg, *, cfg=None, cleanup=None, reader=None):
    config={'enabled':True,'maximum_caption_views':8,'maximum_color_networks':128,
            'maximum_changes':512,'patterns':{},**(cfg or {})}
    if type(config['enabled']) is not bool:raise ValueError('structural enabled must be boolean')
    if not config['enabled']:return deepcopy(target),{'status':'disabled','enabled':False}
    if set(config)-{'enabled','maximum_caption_views','maximum_color_networks','maximum_changes','patterns'}:
        raise ValueError('unknown_structural_configuration')
    for key in ('maximum_caption_views','maximum_color_networks','maximum_changes'):
        if type(config[key]) is not int or config[key]<0:raise ValueError('invalid structural budget')
    patterns=config['patterns']
    if not isinstance(patterns,dict) or set(patterns)-set(PATTERNS) or any(type(v) is not bool for v in patterns.values()):
        raise ValueError('unknown_or_invalid_pattern_configuration')
    if config['maximum_color_networks']==0 and patterns.get('colored_wire_network',True):
        raise ValueError('topology_budget_must_be_positive')
    if not patterns.get('colored_wire_network',True) and any(patterns.get(k,True) for k in (*PATTERNS[:5],'numeric_power_caption','unidentified_capacitor_identity','ground_owner_arbitration','mixed_color_ground_branch')):
        raise ValueError('terminal_and_identity_patterns_require_topology_stage')
    height,width=rgb.shape[:2]
    digest=pins.get('image_sha256')
    for doc in (records,pins,ocr):
        if (not digest or doc.get('image_sha256')!=digest or doc.get('size')!=[width,height]
                or doc.get('coordinate_origin')!='top_left' or doc.get('prediction_only') is not True):
            raise ValueError('structural evidence differs from current image')
    data={'rgb':rgb,'records':records,'pins':pins,'ocr':ocr,'wire_cfg':wire_cfg,
          'config':config,'reader':reader,'cleanup':cleanup,
          'candidate_ids':{p['candidate_id'] for p in pins['pin_candidates']}}
    context=RepairContext(target,data,width,height)
    result,audit=RepairPipeline([ImageStructuralRule()],extension=ImageStructurePolicy()).run(context,
        maximum_new_contacts=config['maximum_changes'],maximum_new_branches=config['maximum_changes'])
    audit['enabled']=True
    audit['pattern_registry']=[{'rule_id':k,'enabled':patterns.get(k,True),
                               'version':'1.1' if k=='numeric_power_caption' else '1.0'} for k in PATTERNS]
    return result,audit
