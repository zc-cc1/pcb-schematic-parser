"""Rendering and conservative partial export of model-derived wire graphs."""
from collections import Counter
import colorsys
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from .dataset_converter import validate_output


def render_graph(rgb,result,path):
    background=Image.fromarray(rgb)
    image=Image.blend(background,Image.new('RGB',background.size,'white'),.42)
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',12)
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


def partial_submission(pin_result,graph,records,auto_id_for_unnamed=False,auto_id_prefix='DET',gates=None):
    """Export the structurally valid subset.

    `auto_id_for_unnamed`: a component whose designator could not be read is still
    exported under a deterministic id (`<prefix>_<detection_index>`), so its bbox,
    type, value, pins and nets reach the submission instead of being dropped whole.
    Those components cannot match a reference key that uses the real designator, so
    the option only buys coverage of everything attached to them, never a
    component match.  `coverage['auto_id_components']` lists them for audit.

    `gates`: which confidence gates may withhold a network - single_pin, junction,
    disconnection.  Defaults to all on; the ablation script turns them off one at a
    time to price each gate.  The identity gate is never optional: a net whose members
    are not exportable cannot be written at all.
    """
    gates = {'single_pin': True, 'junction': True, 'disconnection': True, **(gates or {})}
    width,height=pin_result['size']
    out={'components':{},'pins':{},'nets':{}}
    by_id={r['component_id']:r for r in records}
    auto_ids={}
    for record in records:
        designator=record.get('designator')
        key=str(designator) if designator else None
        if not key:
            if not auto_id_for_unnamed:continue
            key=f"{auto_id_prefix}_{int(record.get('detection_index',0)):04d}"
            auto_ids[key]=record.get('detection_index')
        x1,y1,x2,y2=record['bbox_top_left']
        box=[max(0.,float(x1)),max(0.,height-float(y2)),min(float(width),float(x2)),min(float(height),height-float(y1))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        # Name stays null for an auto id: inventing a designator would claim an
        # identity that was not observed.
        out['components'][key]={'type':record['type'],'Name':record.get('Name') if designator else None,'value':record.get('value'),'bbox':box}
    # pins address their owner by the same key the component was exported with
    owner_keys={}
    for record in records:
        designator=record.get('designator')
        owner_keys[record.get('component_id')]=str(designator) if designator else f"{auto_id_prefix}_{int(record.get('detection_index',0)):04d}"
    eligible={}
    pending=[]
    for pin in pin_result['pin_candidates']:
        reasons=[]
        owner_key=owner_keys.get(pin.get('owner'))
        if pin['owner_status']!='assigned' or owner_key is None or owner_key not in out['components']:reasons.append('component_identity_unresolved')
        if pin.get('number') is None:reasons.append('pin_number_unresolved')
        if pin.get('pinname') is None:reasons.append('pin_name_unresolved')
        if reasons:
            pending.append({'candidate_id':pin['candidate_id'],'reasons':reasons});continue
        x,y=pin['point'];key=f"pin_{pin['number']}"
        owner_pins=out['pins'].setdefault(owner_key,{})
        if key in owner_pins:raise ValueError('Duplicate physical pin key in partial export')
        owner_pins[key]={'pinname':pin['pinname'],'point':{'x':x,'y':height-y}}
        eligible[pin['candidate_id']]=f"{owner_key}.{pin['number']}"
    edges={e['edge_id']:e for e in graph['edges']}
    gap_nets={n for g in graph['gap_candidates'] for n in g['network_ids']}
    withheld=[]
    for net in graph['networks']:
        if not net['pin_candidate_ids']:continue
        reasons=[]
        if len(net['pin_candidate_ids'])<2 and gates.get('single_pin',True):reasons.append('single_pin_fragment')
        if net['unresolved_junction_ids'] and gates.get('junction',True):reasons.append('junction_semantics_unresolved')
        if net['network_id'] in gap_nets and gates.get('disconnection',True):reasons.append('suspected_disconnection')
        if any(p not in eligible for p in net['pin_candidate_ids']):reasons.append('contains_pin_with_unresolved_identity_number_or_name')
        if reasons:
            withheld.append({'network_id':net['network_id'],'reasons':reasons});continue
        segments={}
        for edge_id in net['edge_ids']:
            points=edges[edge_id]['polyline']
            for a,b in zip(points,points[1:]):
                segments[f'edge_{len(segments)+1}']=[{'x':a[0],'y':height-a[1]},{'x':b[0],'y':height-b[1]}]
        refs=sorted({eligible[p] for p in net['pin_candidate_ids']})
        out['nets'][net['network_id']]={'hyperGraph':'('+','.join(refs)+')','edges':segments}
    errors,bounds=validate_output(out,width,height)
    if errors or bounds:raise ValueError(f'Partial export violates the local JSON contract: {errors[:3]} {bounds[:3]}')
    coverage={'status':'partial_image_predictions_only_not_complete_submission','local_contract_valid':True,'official_evaluator_verified':False,
              'exported_components':len(out['components']),'exported_pins':len(eligible),'exported_nets':len(out['nets']),
              'auto_id_components':sorted(auto_ids),'auto_id_count':len(auto_ids),
              'pending_pins':pending,'withheld_networks':withheld,'pin_reason_counts':dict(Counter(r for p in pending for r in p['reasons'])),
              'network_reason_counts':dict(Counter(r for n in withheld for r in n['reasons'])),
              'coordinate_transform':'H-y; local provisional convention','blank_pin_names_guessed':0,
              'logical_flywire_merges':0,'power_symbol_merges':0,'complete_competition_JSON':False}
    return out,coverage
