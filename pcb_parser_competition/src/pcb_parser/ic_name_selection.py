"""Choose IC Name from retained OCR: model, observed designator, description.

This is a final component-field policy. It neither changes pin/model-library
decisions upstream nor invents spellings from a reference dataset or part list.
"""
from copy import deepcopy
import math
import re

import numpy as np

from .component_text import REF, center, choose_global, gap, inside, iou, tokens_from_texts, value_kind


POLICY = 'ic_observed_model_reference_description_v1'
PIN_OR_RAIL = re.compile(
    r'(?:[AD]?GND|PGND|VCC|VDD|VSS|VEE|VBUS|VBAT|VIN|VOUT|AVDD|DVDD|VREF|'
    r'GPIO\d*|GPI\d+|IO\d+|P[A-H]\d{1,2}|ADC\d{1,2}|AIN\d*|PIN\d*|'
    r'RESET|RST|EN|ENABLE|SDA|SCL|SCLK|CLK|MOSI|MISO|TXD?|RXD?|NC|DNC|EP|'
    r'IN[+-]?|OUT[+-]?|CS|SS|FB|SW|INT|IRQ|BOOT\d*)(?:[_/.-][A-Za-z0-9]+)*', re.I)


def category(text, own_reference=None):
    text = text.strip()
    if own_reference and text == own_reference:
        return 'reference'
    if not text or not re.search(r'[^\W\d_]', text, re.UNICODE):
        return 'excluded_numeric_or_empty'
    if REF.fullmatch(text):
        return 'excluded_other_reference'
    if PIN_OR_RAIL.fullmatch(text):
        return 'excluded_pin_or_rail'
    if value_kind(text) is not None or re.search(r'\d\s*PIN\b', text, re.I):
        return 'description'
    if (re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+/\-]*', text)
            and re.search(r'[A-Za-z]', text) and re.search(r'\d', text)):
        return 'model'
    return 'description'


def _reference(record, key):
    observed = record.get('fields', {}).get('designator', {})
    if record.get('designator') == key and observed.get('text') == key and observed.get('bbox'):
        return observed
    return None


def _retained_tokens(texts):
    tokens = tokens_from_texts(texts)
    # Keep a full multiword caption in addition to split words. Never rebuild
    # a string by concatenating separate OCR boxes or guessing missing letters.
    seen=set()
    for row in texts:
        for observation in [row]+row.get('observations',[]):
            text=str(observation.get('text','')).strip()
            b=observation.get('bbox',row['bbox'])
            identity=(text,tuple(b),observation.get('view',row.get('view')))
            if len(text.split())<2 or identity in seen:
                continue
            seen.add(identity)
            tokens.append({'token_id':f'ic_full_{len(tokens):05d}','text_id':row['text_id'],
                'text':text,'bbox':b,'confidence':float(observation['confidence']),
                'source_view':observation.get('view',row.get('view')),'role':'full_caption',
                'source_text':observation['text'],'approximate_split_geometry':False})
    return tokens


def _same_glyph(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    small=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
    large=max((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
    return iou(a,b)>.45 or (small>0 and overlap/small>=.9 and large/small<=3)


def _geometry(token, box):
    b = token['bbox'];cx,cy = center(b);w,h = box[2]-box[0],box[3]-box[1]
    if token['confidence'] < .9:
        return None, 'low_confidence'
    if b[2]-b[0] > 2.5*w or b[3]-b[1] > .3*h:
        return None, 'not_a_local_component_caption'
    interior = (box[0]+.2*w < cx < box[2]-.2*w
                and box[1]+.15*h < cy < box[3]-.15*h)
    edge = (box[0]-.1*w <= cx <= box[2]+.1*w
            and (cy <= box[1] or cy >= box[3]) and gap(b,box) <= 40)
    if not (interior or edge):
        return None, 'side_pin_region_or_distant_caption'
    score = (min(35., .1*abs(cy-center(box)[1])) if interior else .7*gap(b,box))
    score += .2*abs(cx-center(box)[0]) + 12*(1-token['confidence'])
    return score, 'central_body_caption' if interior else 'top_or_bottom_caption'


def select_ic_names(payload, records_document, ocr_document):
    """Return a copy with IC Name selected; all other fields remain identical."""
    if (not records_document.get('prediction_only') or records_document.get('coordinate_origin')!='top_left'
            or not ocr_document.get('prediction_only') or ocr_document.get('coordinate_origin')!='top_left'
            or not isinstance(records_document.get('image_sha256'),str) or len(records_document['image_sha256'])!=64
            or records_document.get('image_sha256')!=ocr_document.get('image_sha256')
            or records_document.get('size')!=ocr_document.get('size')):
        raise ValueError('IC Name selection requires matching current-image OCR and records')
    records = records_document['records']
    record_map = {r['component_id']:r for r in records}
    if len(record_map)!=len(records):
        raise ValueError('Duplicate component records')
    result = deepcopy(payload)
    tokens = _retained_tokens(ocr_document.get('texts', []))
    # Claimed glyphs stay with their current physical owner. Identical strings
    # at separate positions are independent candidates and can name two ICs.
    claimed = [(r['component_id'],f,v['bbox']) for r in records
               for f,v in r.get('fields',{}).items() if f in {'Name','value','designator'} and isinstance(v,dict) and v.get('bbox')]
    entries = []
    for key,component in result['components'].items():
        if component['type'] not in {'ic','box'}:
            continue
        if key not in record_map:
            raise ValueError('Missing current-image IC record: '+key)
        r = record_map[key];ref = _reference(r,key);box=r['bbox_top_left']
        before = component.get('Name')
        existing_category = category(before, key if ref else None) if before is not None else None
        entry = {'component_id':key,'before_Name':before,'after_Name':None,
                 'selected_category':None,'selected_evidence':None,'candidates':[]}
        # Existing model assignments already passed the upstream ownership
        # checks. Preserve them instead of replacing a working model with U1.
        if before is not None and existing_category=='model':
            entry.update(after_Name=before,selected_category='model',
                         selected_evidence={'source':'existing_observed_model',**r.get('fields',{}).get('Name',{})})
        for token in tokens:
            if gap(token['bbox'],box)>100:
                continue
            cat = category(token['text'], key if ref else None)
            score,reason = _geometry(token,box)
            if cat.startswith('excluded_') or cat=='reference':
                score,reason = None,cat
            owners = {owner for owner,field,b in claimed if iou(token['bbox'],b)>=.45
                      and not (owner==key and field=='Name')}
            if owners:
                score,reason = None,'glyph_already_claimed_by_field'
            # A central caption cannot be inside a different detected part.
            if any(other['component_id']!=key and other.get('duplicate_of_detection_index') is None
                   and inside(center(token['bbox']),other['bbox_top_left'],2) for other in records):
                score,reason = None,'inside_another_component'
            entry['candidates'].append({**token,'category':cat,'eligible':score is not None,
                                        'selection_score':score,'reason':reason})
        if before is not None:
            entry['previous_Name_category'] = existing_category
            entry['previous_Name_evidence'] = deepcopy(r.get('fields',{}).get('Name',{}))
        if ref:
            entry['observed_reference'] = deepcopy(ref)
        entries.append(entry)

    rejections = []
    for tier in ['model','description']:
        if tier=='description':
            # The second priority is this IC's already observed identifier.
            # Technical UNRESOLVED_DET IDs are never used as a fallback name.
            for entry in entries:
                if entry['after_Name'] is None and entry.get('observed_reference'):
                    entry.update(after_Name=entry['observed_reference']['text'],selected_category='reference',
                                 selected_evidence={'source':'observed_designator_fallback',**entry['observed_reference']})
                if (entry['after_Name'] is None and entry.get('previous_Name_category')=='description'
                        and entry['before_Name'] is not None):
                    entry.update(after_Name=entry['before_Name'],selected_category='description',
                                 selected_evidence={'source':'existing_observed_description',**entry['previous_Name_evidence']})
        active = [e for e in entries if e['after_Name'] is None]
        for entry in active:
            for candidate in entry['candidates']:
                if any(other['component_id']!=entry['component_id'] and (other.get('selected_evidence') or {}).get('bbox')
                       and _same_glyph(candidate['bbox'],other['selected_evidence']['bbox']) for other in entries):
                    candidate.update(eligible=False,selection_score=None,reason='glyph_selected_for_another_IC')
        groups = []
        for token in tokens:
            if not any(any(c['token_id']==token['token_id'] and c['eligible'] and c['category']==tier
                           for c in e['candidates']) for e in active):
                continue
            for group in groups:
                if any(_same_glyph(token['bbox'],old['bbox']) for old in group):
                    group.append(token);break
            else:
                groups.append([token])
        costs = np.full((len(active),len(groups)),1e6);chosen={}
        for i,entry in enumerate(active):
            eligible = {c['token_id']:c for c in entry['candidates'] if c['eligible'] and c['category']==tier}
            for j,group in enumerate(groups):
                options = [eligible[t['token_id']] for t in group if t['token_id'] in eligible]
                if options:
                    candidate=max(options,key=lambda c:(c['confidence'],not c['approximate_split_geometry'],-c['selection_score']))
                    if any(c['text']!=candidate['text'] and not c['approximate_split_geometry']
                           and abs(c['confidence']-candidate['confidence'])<.02 for c in options):
                        rejections.append({'field':'IC_Name_'+tier,'component_id':entry['component_id'],
                                           'reason':'conflicting_same_glyph_spellings',
                                           'spellings':sorted({c['text'] for c in options})})
                        continue
                    costs[i,j]=candidate['selection_score'];chosen[i,j]=candidate
        details=[]
        for i,(j,score,margin) in choose_global(costs,90,3,details,'IC_Name_'+tier).items():
            c=chosen[i,j];active[i].update(after_Name=c['text'],selected_category=tier,
                selected_evidence={**c,'source':'retained_OCR_'+tier,'assignment_margin':margin})
        rejections.extend(details)
    for entry in entries:
        result['components'][entry['component_id']]['Name']=entry['after_Name']
        entry['changed']=entry['before_Name']!=entry['after_Name']
    audit={'policy':POLICY,'priority':['model','reference','description'],
           'image_sha256':ocr_document['image_sha256'],'size':ocr_document['size'],
           'components':entries,'assignment_rejections':rejections,
           'changed_Name_count':sum(e['changed'] for e in entries),
           'full_OCR_log_required':True,'OCR_row_count':len(ocr_document.get('texts',[])),
           'only_IC_Name_changed':True,'reference_labels_or_model_dictionary_used':False}
    return result,audit
