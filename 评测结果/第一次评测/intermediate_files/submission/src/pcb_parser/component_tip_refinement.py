"""Add only model peaks with cross-scale, owner and wire-context agreement.

No box-edge snapping or raw-image segmentation. Expected symbol pin counts
limit proposals; they do not manufacture a location or infer a pin number.
The prior candidate list and semantic fields are preserved without deletions.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np

from .unet_inference import peak_candidates, select_candidates
from .pin_ownership import associate_pins


def distance(a,b):
    return math.hypot(a[0]-b[0],a[1]-b[1])


def pair_is_plausible(points,box,cfg):
    if len(points)!=2:return False
    w,h=box[2]-box[0],box[3]-box[1]
    axis=0 if w>=h else 1
    span=abs(points[0][axis]-points[1][axis])
    tangent=abs(points[0][1-axis]-points[1][1-axis])
    return (span>=cfg['pair_min_axis_fraction']*max(w,h)
            and tangent<=max(cfg['pair_max_tangent_min_px'],cfg['pair_max_tangent_fraction']*min(w,h)))


def refine_component_tips(pins,native,zoom,cfg,owner_cfg):
    width,height=pins['size']
    for probability in [native,zoom]:
        if probability.shape!=(2,height,width) or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
            raise ValueError('Probability map must be finite, bounded and match original PNG dimensions')
    if cfg['new_numbers_policy']!='keep_unresolved':raise ValueError('Unobserved pin numbers must remain unresolved')
    components={c['detection_index']:c for c in pins['components']}
    if len(components)!=len(pins['components']):raise ValueError('Duplicate component detection index')
    ids={p['candidate_id'] for p in pins['pin_candidates']}
    if len(ids)!=len(pins['pin_candidates']):raise ValueError('Duplicate original candidate ID')
    owned=defaultdict(list)
    for pin in pins['pin_candidates']:
        if pin['owner_status']=='assigned':owned[pin['owner_detection_index']].append(pin)
    deficits={i:cfg['expected_tip_counts'][c['type']]-len(owned[i]) for i,c in components.items()
              if c['type'] in cfg['expected_tip_counts'] and len(owned[i])<cfg['expected_tip_counts'][c['type']]}
    native_peaks=peak_candidates(native[1],cfg['native_threshold'],cfg['peak_kernel'],cfg['peak_refinement'])
    zoom_peaks=peak_candidates(zoom[1],cfg['zoom_threshold'],cfg['peak_kernel'],cfg['peak_refinement'])
    native_owned=associate_pins(native_peaks,list(components.values()),owner_cfg,[width,height],native[0])['pin_candidates']
    zoom_owned=associate_pins(zoom_peaks,list(components.values()),owner_cfg,[width,height],zoom[0])['pin_candidates']
    zoom_by_owner=defaultdict(list)
    for pin in zoom_owned:
        if pin['owner_status']=='assigned':zoom_by_owner[pin['owner_detection_index']].append(pin)
    proposals=defaultdict(list);rejected=[]
    existing=[p['point'] for p in pins['pin_candidates']]
    for peak in native_owned:
        index=peak.get('owner_detection_index')
        if peak['owner_status']!='assigned' or index not in deficits:continue
        reasons=[]
        if any(distance(peak['point'],p)<cfg['original_tip_exclusion_radius_px'] for p in existing):
            continue
        support=[p for p in zoom_by_owner[index] if distance(p['point'],peak['point'])<=cfg['cross_scale_distance_px']]
        if not support:reasons.append('cross_scale_peak_or_owner_disagrees')
        x,y=map(round,peak['point']);r=cfg['wire_context_radius_px']
        context=native[0,max(0,y-r):min(height,y+r+1),max(0,x-r):min(width,x+r+1)]
        wire_max=float(context.max()) if context.size else 0.0
        if wire_max<cfg['wire_min_probability']:reasons.append('no_confident_learned_wire_context')
        record={'point':peak['point'],'confidence':peak['confidence'],'owner_detection_index':index,
                'owner':peak['owner'],'owner_candidates':peak['owner_candidates'],'wire_context_max':wire_max,
                'native_pin_probability':peak['confidence'],'zoom_support':support,
                'withheld_reasons':reasons}
        if reasons:rejected.append(record)
        else:proposals[index].append(record)
    result=deepcopy(pins);added=[];component_reviews=[];sequence=len(ids)+1
    for index,deficit in sorted(deficits.items()):
        proposed=select_candidates(proposals[index],0,cfg['proposal_nms_radius_px'],max(1,len(proposals[index])))
        component=components[index];reasons=[]
        if len(proposed)>deficit:reasons.append('more_plausible_peaks_than_missing_tip_slots')
        if component['type'] in {'resistor','capacitor'}:
            combined=[p['point'] for p in owned[index]]+[p['point'] for p in proposed]
            if not pair_is_plausible(combined,component['bbox_top_left'],cfg):
                reasons.append('incomplete_or_nonopposed_two_terminal_pair')
        accepted=proposed if proposed and not reasons else []
        for proposal in accepted:
            # Appended IDs cannot rename or reorder any prior model candidate.
            while f'PIN_CAND_{sequence:04d}' in ids:sequence+=1
            cid=f'PIN_CAND_{sequence:04d}';sequence+=1;ids.add(cid)
            support=max(proposal['zoom_support'],key=lambda p:p['confidence'])
            evidence={'source':'same_frozen_UNet_native_and_zoom15_peaks','native_pin_probability':proposal['confidence'],
                      'zoom_pin_probability':support['confidence'],'zoom_point':support['point'],
                      'cross_scale_distance_px':distance(proposal['point'],support['point']),
                      'native_wire_context_max':proposal['wire_context_max'],
                      'coordinate_snapped_to_component_box':False,'number_or_name_inferred':False,
                      'model_peak_below_old_global_threshold':proposal['confidence']<cfg['original_global_threshold']}
            pin={'candidate_id':cid,'point':proposal['point'],'confidence':proposal['confidence'],
                 'owner':proposal['owner'],'owner_detection_index':index,'owner_status':'assigned',
                 'owner_candidates':proposal['owner_candidates'],'number':None,'pinname':None,
                 'number_status':'requires_OCR_or_verified_symbol_evidence','pinname_status':'requires_OCR_or_verified_symbol_evidence',
                 'text_candidates':{},'text_evidence':{},'localization_evidence':evidence}
            result['pin_candidates'].append(pin);added.append({'candidate_id':cid,'component_type':component['type'],**pin})
        component_reviews.append({'detection_index':index,'component_id':component['component_id'],'component_type':component['type'],
                                  'bbox_top_left':component['bbox_top_left'],'prior_owned_count':len(owned[index]),
                                  'maximum_missing_slots':deficit,'supported_proposal_count':len(proposed),
                                  'added_count':len(accepted),'withheld_reasons':reasons})
    if result['pin_candidates'][:len(pins['pin_candidates'])]!=pins['pin_candidates']:
        raise ValueError('Refinement changed existing tips or semantic fields')
    summary={'prior_candidates':len(pins['pin_candidates']),'added_candidates':len(added),'total_candidates':len(result['pin_candidates']),
             'added_by_type':dict(Counter(p['component_type'] for p in added)),
             'components_with_missing_tip_slots':len(deficits),'original_candidates_modified_or_removed':0,
             'tip_points_generated_from_bbox':0,'numbers_guessed':0,'names_guessed':0,
             'rejected_native_proposal_reasons':dict(Counter(r for p in rejected for r in p['withheld_reasons'])),
             'component_withheld_reasons':dict(Counter(r for c in component_reviews for r in c['withheld_reasons']))}
    result['component_tip_refinement_summary']=summary
    return result,{'summary':summary,'added_candidates':added,'component_reviews':component_reviews,'rejected_proposals':rejected}
