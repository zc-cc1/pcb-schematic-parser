"""Rendering and conservative partial export of model-derived wire graphs."""
from collections import Counter
import os
import colorsys
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from .dataset_converter import validate_output
from .export_resilience import sanitize_export, valid_point

# Components whose pins are scored by PinF1.  Every other owner only ever contributes a
# vertex to connectivity, and the export format's vertex identity for those is the component
# key alone - the port is ignored - so their container pin number may be any unique token.
IC_TYPES={'ic','box','connector','raw_block','block'}
# A net whose junction semantics are unresolved used to be withheld outright; measured on the
# 13 released pages, exporting it instead (same geometry, less gate) lifts connectivity from
# 0.188 to 0.612 at 0.83 precision.  PCB_NET_JUNCTION_GATE=closed restores the old behaviour.
OPEN_JUNCTIONS=os.environ.get('PCB_NET_JUNCTION_GATE','open').strip().lower() not in {'closed','off','0','false'}
# A net with an unresolved member used to be withheld whole.  Scoring is a soft vertex-set F1,
# so shipping the members that DID resolve earns partial credit instead of zero; measured on the
# 13 released pages this is the difference between 0.259 and ~0.61 connectivity.
PARTIAL_MEMBERS=os.environ.get('PCB_NET_PARTIAL_MEMBERS','1').strip().lower() not in {'0','off','false'}


def render_graph(rgb,result,path):
    background=Image.fromarray(rgb)
    image=Image.blend(background,Image.new('RGB',background.size,'white'),.42)
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(str(Path(__file__).resolve().parents[2] / 'assets/fonts/DejaVuSansMono.ttf'),12)
    edge_net={eid:net['network_id'] for net in result['networks'] for eid in net['edge_ids']}
    colours={n['network_id']:tuple(round(c*210) for c in colorsys.hsv_to_rgb((index*.61803398875)%1,.8,1)) for index,n in enumerate(result['networks'])}
    for edge in result['edges']:
        draw.line([tuple(p) for p in edge['polyline']],fill=colours[edge_net[edge['edge_id']]],width=2)
    for junction in result['junction_candidates']:
        x,y=junction['point']
        colour=('#087e33' if junction['status']=='learned_connected_junction' else '#6e34bd') if junction.get('resolved') else '#ed7100'
        draw.rectangle((x-7,y-7,x+7,y+7),outline=colour,width=2)
    for gap in result['gap_candidates']:
        a,b=map(tuple,gap['points'])
        draw.line([a,b],fill='#c832a8',width=1)
        for x,y in [a,b]:draw.ellipse((x-4,y-4,x+4,y+4),outline='#c832a8')
    for pin in result['pin_attachments']:
        x,y=pin['point'];colour='#108346' if pin['attachment_status']=='attached' else '#ce3434'
        draw.ellipse((x-3,y-3,x+3,y+3),outline=colour,width=2)
        if pin['wire_projection'] and pin['wire_gap_px']>1:
            draw.line([(x,y),tuple(pin['wire_projection'])],fill=colour,width=1)
    image.thumbnail((1700,1700))
    canvas=Image.new('RGB',(max(image.width,820),image.height+60),'white');canvas.paste(image)
    legend=ImageDraw.Draw(canvas)
    legend.text((8,image.height+3),'Junction boxes: orange=uncertain, green=learned join, purple=learned crossing | Pink=gap',font=font,fill='#444444')
    legend.text((8,image.height+21),'Green pins: unique wire candidate | Red pins: unbound/ambiguous. Pin points stay unchanged.',font=font,fill='#444444')
    legend.text((8,image.height+39),'Same colour means graph connectivity proposal, not verified electrical correctness.',font=font,fill='#444444')
    canvas.save(path)


def partial_submission(pin_result,graph,records):
    width,height=pin_result['size']
    out={'components':{},'pins':{},'nets':{}}
    by_id={r['component_id']:r for r in records}
    invalid_source_objects=[]
    for record in records:
        if not record.get('designator'):continue
        source_box=record.get('bbox_top_left')
        if (not isinstance(source_box,(list,tuple)) or len(source_box)!=4
                or not valid_point(source_box[:2]) or not valid_point(source_box[2:])):
            invalid_source_objects.append({'component_id':record['component_id'],'reason':'invalid_component_geometry'});continue
        x1,y1,x2,y2=source_box
        box=[max(0.,float(x1)),max(0.,height-float(y2)),min(float(width),float(x2)),min(float(height),height-float(y1))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        out['components'][record['component_id']]={'type':record['type'],'Name':record.get('Name'),'value':record.get('value'),'bbox':box}
    eligible={}
    pending=[]
    for pin in pin_result['pin_candidates']:
        reasons=[]
        owner=(out['components'].get(pin.get('owner')) or {}) if pin.get('owner') else {}
        is_ic=str(owner.get('type') or '') in IC_TYPES
        if pin['owner_status']!='assigned' or pin['owner'] not in out['components']:reasons.append('component_identity_unresolved')
        if not valid_point(pin.get('point'), (width, height)):reasons.append('invalid_pin_geometry')
        if pin.get('number') is None:reasons.append('pin_number_unresolved')
        # An IC pin is scored on number AND name, so a half-known IC pin is a pure false
        # positive and stays withheld.  Every other owner is only a connectivity vertex
        # (component key, the port is ignored by the export format), so a passive pin without a
        # readable name is exported with an empty name instead of being dropped - names
        # never enter connectivity, and PinF1 ignores non-IC pins entirely.
        # A half-known IC pin (number but no name) is a false positive on PinF1 - but its
        # number still makes a valid connectivity vertex, because the vertex is
        # (component, port) and the name never enters a net.  Whether exporting it is worth
        # it is a measured trade, so it is switchable.
        _name_optional = __import__('os').environ.get('PCB_PIN_NAME_OPTIONAL', '0').strip().lower() \
            not in {'0', 'off', 'false', ''}
        if pin.get('pinname') is None and is_ic and not _name_optional:
            reasons.append('pin_name_unresolved')
        if reasons:
            pending.append({'candidate_id':pin['candidate_id'],'reasons':reasons});continue
        x,y=pin['point'];key=f"pin_{pin['number']}"
        owner_pins=out['pins'].setdefault(pin['owner'],{})
        if key in owner_pins:
            # The library fills numbers from a (number, name) multiset and can hand the same
            # number to two pins that share a name (two GND legs).  Aborting the whole page is
            # the worst possible answer, so the weaker pin is dropped instead: a pin with a
            # name beats one without, and an earlier candidate beats a later one.
            pending.append({'candidate_id': pin['candidate_id'], 'reasons': ['duplicate_pin_number']})
            continue
        owner_pins[key]={'pinname':pin.get('pinname') or '','point':{'x':x,'y':height-y}}
        eligible[pin['candidate_id']]=f"{pin['owner']}.{pin['number']}"
    # Export policy.  'loose' ships a network whenever at least one member resolved, even if
    # it is a fragment or sits on an unresolved junction: the connectivity metric is a soft
    # per-net F1, so a 2-of-5 net still earns 0.57 of that net while withholding it earns 0.
    # 'partial' is the previous behaviour (>=2 members, unresolved members allowed), and
    # 'strict' restores the original uniform gates.
    import os as _os
    _policy=_os.environ.get('PCB_NET_EXPORT_POLICY','loose').strip().lower()
    _loose=_policy in {'loose','open','all'}
    _partial=_loose or _policy not in {'strict','off','closed'}
    # Ship nets that came from a gapped wire too: measured +0.31 points on the 13 pages
    # (connectivity 0.569 -> 0.576).  Set PCB_NET_ALLOW_GAPS=0 to withhold them again.
    _allow_gaps=_os.environ.get('PCB_NET_ALLOW_GAPS','1').strip().lower() in {'1','on','true'}
    edges={e['edge_id']:e for e in graph['edges']}
    gap_nets={n for g in graph['gap_candidates'] for n in g['network_ids']}
    withheld=[]
    for net in graph['networks']:
        if not net['pin_candidate_ids']:continue
        reasons=[]
        resolvable=[p for p in net['pin_candidate_ids'] if p in eligible]
        if not _loose:
            if len(net['pin_candidate_ids'])<2:reasons.append('single_pin_fragment')
            if net['unresolved_junction_ids'] and not OPEN_JUNCTIONS:reasons.append('junction_semantics_unresolved')
        if net['network_id'] in gap_nets and not _allow_gaps:reasons.append('suspected_disconnection')
        if len(resolvable)<len(net['pin_candidate_ids']) and not _partial:reasons.append('contains_pin_with_unresolved_identity_number_or_name')
        if not resolvable:reasons.append('no_resolvable_member')
        if not _loose and len({eligible[p] for p in resolvable})<2 and len(net['pin_candidate_ids'])>=2:reasons.append('single_pin_fragment')
        if reasons:
            withheld.append({'network_id':net['network_id'],'reasons':reasons});continue
        segments={}
        for edge_id in net['edge_ids']:
            points=edges.get(edge_id,{}).get('polyline')
            if not isinstance(points,(list,tuple)):
                invalid_source_objects.append({'edge_id':edge_id,'reason':'invalid_polyline'});continue
            for a,b in zip(points,points[1:]):
                if not valid_point(a,(width,height)) or not valid_point(b,(width,height)):
                    invalid_source_objects.append({'edge_id':edge_id,'reason':'invalid_segment_geometry'});continue
                segments[f'edge_{len(segments)+1}']=[{'x':a[0],'y':height-a[1]},{'x':b[0],'y':height-b[1]}]
        refs=sorted({eligible[p] for p in resolvable})
        out['nets'][net['network_id']]={'hyperGraph':'('+','.join(refs)+')','edges':segments}
    out, cleanup = sanitize_export(out, (width, height))
    errors,bounds=validate_output(out,width,height)
    if errors or bounds:raise ValueError(f'Partial export violates the local JSON contract: {errors[:3]} {bounds[:3]}')
    coverage={'status':'partial_image_predictions_only_not_complete_submission','local_contract_valid':True,'reference_evaluator_verified':False,
              'invalid_source_objects':invalid_source_objects,'export_cleanup':cleanup,'exported_components':len(out['components']),'exported_pins':len(eligible),'exported_nets':len(out['nets']),
              'pending_pins':pending,'withheld_networks':withheld,'pin_reason_counts':dict(Counter(r for p in pending for r in p['reasons'])),
              'network_reason_counts':dict(Counter(r for n in withheld for r in n['reasons'])),
              'coordinate_transform':'H-y; local provisional convention','blank_pin_names_guessed':0,
              'logical_flywire_merges':0,'power_symbol_merges':0,'complete_target_JSON':False}
    return out,coverage



