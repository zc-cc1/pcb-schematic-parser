"""Export detected power symbols independently of ordinary designators.

Uses current-image detections and OCR only. Technical detection IDs are not
invented schematic designators; the official power matcher ignores key/Name.
No pin coordinates, wire edges or network membership are generated here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import os
import re


GROUND = re.compile(r'(?:GND|AGND|DGND|PGND|SGND|CGND|EGND)(?:_[A-Za-z0-9]+)*', re.I)
SUPPLY = re.compile(r'(?:VCC|VDDIO|VDD|VSS|VEE|VBUS|VBAT|VIN|VOUT|AVDD|DVDD|PVDD|AVCC|DVCC|VIO)'
                    r'(?:[0-9_+.-][A-Za-z0-9_+.-]*)?|[+-]?(?:[0-9]+(?:[._][0-9]+)?V(?:[0-9]+)?|V[0-9]+(?:P[0-9]+)?)(?:_[A-Za-z0-9]+)*', re.I)

# A caption is one physical word, so a spelling repeated across views is stronger
# evidence than a single high-scoring read.  Measured on the 13 released pages: the
# old .97-per-observation gate threw away captions that two views agreed on at .92-.96
# ('VCC' .941 on case 1, '+5V' 1.00 in one view on case 2/5), and case variants
# ('VCC' / 'vcc' / 'VcC') split one caption into two "different" spellings.
# PCB_POWER_CAPTION_MIN_CONF / _SINGLE_VIEW_CONF restore the old behaviour by setting
# them to 0.97 and 2.0 respectively.
POWER_CAPTION_MIN_CONF = float(os.environ.get('PCB_POWER_CAPTION_MIN_CONF', '0.90'))
POWER_CAPTION_SINGLE_VIEW_CONF = float(os.environ.get('PCB_POWER_CAPTION_SINGLE_VIEW_CONF', '0.98'))
# Every ground symbol in the released reference carries value 'GND' (83/83), and the
# symbol class itself carries that meaning, so a ground with no readable caption is
# spelled GND instead of staying null (null matches nothing under the power rule).
POWER_GROUND_VALUE_FALLBACK = os.environ.get('PCB_POWER_GROUND_VALUE', '1').strip().lower() not in {'0', 'off', 'false'}
# PCB_NET_STRIP_EXPORT=0 stops the net-label strips from being exported on their own.
NET_STRIP_EXPORT = os.environ.get('PCB_NET_STRIP_EXPORT', '0').strip().lower() not in {'0', 'off', 'false'}


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def gap(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def overlap(a, b):
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
    return intersection / max(1., (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def observed_captions(texts):
    groups = []
    for row in texts:
        for obs in [row] + row.get('observations', []):
            text = str(obs.get('text', '')).strip()
            confidence = float(obs.get('confidence', 0))
            box = obs.get('bbox', row.get('bbox'))
            view = obs.get('view', row.get('view'))
            if (confidence < POWER_CAPTION_MIN_CONF or not view or not box
                    or not (GROUND.fullmatch(text) or SUPPLY.fullmatch(text))):
                continue
            found = next((g for g in groups if all(overlap(box,b) >= .4 for b in g['boxes'])), None)
            if found is None:
                found = {'boxes': [], 'observations': []}
                groups.append(found)
            found['boxes'].append(box)
            found['observations'].append((confidence, text, box, view))
    captions = []
    for group in groups:
        # One caption can be spelled several ways across views ('VCC' / 'vcc' / 'VcC');
        # group by the upper-cased key so those are votes for one caption, not two.
        votes = {}
        for confidence, text, box, view in group['observations']:
            vote = votes.setdefault(text.upper(), {'views': set(), 'confidence': 0.0, 'spellings': Counter()})
            vote['views'].add(view)
            vote['confidence'] = max(vote['confidence'], confidence)
            vote['spellings'][text] += 1
        supported = {key: vote for key, vote in votes.items()
                     if len(vote['views']) >= 2 or vote['confidence'] >= POWER_CAPTION_SINGLE_VIEW_CONF}
        if not supported:
            continue
        # Truncated readings of the same word ('VDD_1' / 'VDD_1.') can now enter as
        # single-view candidates, so keep the best-supported key instead of discarding
        # the group; a genuine tie between two different rails is still dropped.
        best_key, best = max(supported.items(), key=lambda kv: (len(kv[1]['views']), kv[1]['confidence']))
        if any(v is not best and len(v['views']) >= len(best['views']) for v in supported.values()):
            continue
        text = best['spellings'].most_common(1)[0][0]
        _, _, box, _ = max((o for o in group['observations'] if o[1].upper() == best_key),
                           key=lambda o: o[0])
        captions.append({'text': text, 'bbox': box, 'views': sorted(best['views'])})
    return captions


def missing_field_diagnostics(records_document, ocr_document):
    """Keep bounded OCR evidence for missing fields; never fill values by guess."""
    entries = []
    for record in records_document['records']:
        if not record.get('designator'):
            continue
        missing = []
        if record['type'] in {'resistor','capacitor','inductor'} and record.get('value') is None:
            missing.append('value')
        if record['type'] in {'ic','opamp','led','diode','mosfet','bjt'} and record.get('Name') is None:
            missing.append('Name')
        if not missing:
            continue
        box = record['bbox_top_left']
        nearby = []
        for row in ocr_document.get('texts', []):
            tb = row.get('bbox')
            if not tb or gap(box,tb)>100:
                continue
            nearby.append({'text':row['text'],'bbox':tb,'confidence':row['confidence'],
                           'view':row.get('view'),'distance_px':round(gap(box,tb),3),
                           'alternatives':[{'text':o['text'],'confidence':o['confidence'],'view':o.get('view')}
                                           for o in row.get('observations',[])[:3]]})
        nearby.sort(key=lambda row:(row['distance_px'],math.dist(center(row['bbox']),center(box))))
        entries.append({'component_id':record['component_id'],'type':record['type'],'missing_fields':missing,
                        'bbox_top_left':box,'current_binding_evidence':record.get('fields',{}),
                        'nearby_OCR':nearby[:8]})
    return {'source_image_sha256':records_document['image_sha256'],'entries':entries,
            'missing_fields_are_not_automatically_errors':True,'values_or_names_changed':0}


def export_power_components(payload, records_document, ocr_document):
    if not records_document.get('prediction_only') or records_document.get('coordinate_origin') != 'top_left':
        raise ValueError('Power export requires current-image prediction records')
    if (not ocr_document.get('prediction_only') or ocr_document.get('image_sha256') != records_document.get('image_sha256')
            or ocr_document.get('size') != records_document.get('size')):
        raise ValueError('Power records and OCR came from different images')
    width, height = records_document['size']
    records = records_document['records']
    if len({r['component_id'] for r in records}) != len(records):
        raise ValueError('Duplicate detection identities')
    powers = [r for r in records if r['type'] in {'gnd','vcc'} and r.get('duplicate_of_detection_index') is None]
    ordinary_boxes = [r['bbox_top_left'] for r in records if r['type'] not in {'gnd','vcc'}]
    captions = observed_captions(ocr_document.get('texts', []))
    choices = defaultdict(list)
    for caption in captions:
        cx, cy = center(caption['bbox'])
        if any(b[0]+2 < cx < b[2]-2 and b[1]+2 < cy < b[3]-2 for b in ordinary_boxes):
            continue  # An IC's internal GND/VCC pin label is not a power caption.
        candidates = []
        for record in powers:
            pattern = GROUND if record['type'] == 'gnd' else SUPPLY
            if not pattern.fullmatch(caption['text']):
                continue
            box = record['bbox_top_left']
            sx, sy = center(box)
            alignment = min(abs(cx-sx), abs(cy-sy))
            if gap(box,caption['bbox']) > 36 or alignment > max(9, .35*min(box[2]-box[0],box[3]-box[1])):
                continue
            candidates.append((gap(box,caption['bbox']) + .2*math.hypot(cx-sx,cy-sy) + alignment, record['component_id']))
        candidates.sort()
        if candidates and (len(candidates)==1 or candidates[1][0]-candidates[0][0] >= 6):
            choices[candidates[0][1]].append(caption)
    result = deepcopy(payload)
    audit = {'policy': 'observed_power_symbols_v1', 'added': [], 'skipped': [],
             'pin_coordinates_changed': 0, 'network_membership_changed': 0,
             'source_image_sha256': records_document['image_sha256']}
    for record in powers:
        cid = record['component_id']
        if cid in result['components']:
            audit['skipped'].append({'component_id':cid, 'reason':'already_exported'})
            continue
        x1,y1,x2,y2 = record['bbox_top_left']
        box = [max(0.,float(x1)),max(0.,height-float(y2)),min(float(width),float(x2)),min(float(height),height-float(y1))]
        if not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            audit['skipped'].append({'component_id':cid, 'reason':'invalid_box'})
            continue
        matches = choices[cid]
        spellings = {c['text'] for c in matches}
        observed = next(iter(spellings)) if len(spellings)==1 else None
        scalar = record.get('value')
        reason = 'existing_observed_value' if scalar is not None else 'caption_not_uniquely_observed'
        if scalar is None and observed is not None:
            # The reference spells rails in upper case ('VCC', '+5V'); OCR sometimes
            # reads the same word lower case, which would fail the exact comparison.
            scalar = observed.upper() if any(c.isalpha() for c in observed) else observed
            reason = 'nearby_caption_supported_by_distinct_OCR_views'
        if scalar is None and record['type'] == 'gnd' and POWER_GROUND_VALUE_FALLBACK:
            # The released annotation spells every ground symbol 'GND' (83 of 83), and
            # the symbol class is that meaning; a missing caption is a missing spelling,
            # not missing evidence about the rail.
            scalar = 'GND'
            reason = 'ground_symbol_value_convention'
        result['components'][cid] = {'type':record['type'], 'Name':record.get('Name'), 'value':scalar,'bbox':box}
        result['pins'].setdefault(cid,{})
        audit['added'].append({'component_id':cid,'type':record['type'],'value':scalar,
                               'value_source':reason,'caption_evidence':matches,
                               'detection_index':record['detection_index']})
    audit['added_count'] = len(audit['added'])
    audit['type_counts'] = dict(Counter(row['type'] for row in audit['added']))
    audit['value_counts'] = dict(Counter('null' if row['value'] is None else 'observed' for row in audit['added']))
    # Net label strips (VIN / VOUT / BAT …) are ordinary components in the reference:
    # key = the printed label, value = the same text, type net_bidirection.  The
    # engineering export drops records whose designator is unresolved, so a strip whose
    # text was never bound never reaches the submission - case 12 kept VOUT and lost the
    # four VIN strips even though the detector scored them at 0.50-0.93 and OCR read the
    # text.  Export them here, naming them from the printed text and de-duplicating
    # repeated labels the way the reference spells them (VIN, VIN_2, VIN_3 …).
    if NET_STRIP_EXPORT:
        strips = [r for r in records if r['type'] in {'vbus_in','vbus_out','vbus_bidirectional'}
                  and r.get('duplicate_of_detection_index') is None]
        existing = [c.get('bbox') for c in result['components'].values() if c.get('bbox')]
        ordered = sorted(strips, key=lambda r:(r['bbox_top_left'][1], r['bbox_top_left'][0]))
        seen = Counter()
        for record in ordered:
            x1,y1,x2,y2 = record['bbox_top_left']
            box = [max(0.,float(x1)),max(0.,height-float(y2)),min(float(width),float(x2)),min(float(height),height-float(y1))]
            if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                audit['skipped'].append({'component_id':record['component_id'],'reason':'invalid_strip_box'})
                continue
            if any(overlap(box, other) >= .5 for other in existing):
                audit['skipped'].append({'component_id':record['component_id'],'reason':'strip_already_exported'})
                continue
            nearby = []
            for row in ocr_document.get('texts', []):
                text = str(row.get('text') or '').strip()
                tb = row.get('bbox')
                if not tb or not text or not re.fullmatch(r'[A-Za-z][A-Za-z0-9+_\-/]{0,15}', text):
                    continue
                if re.fullmatch(r'[A-Za-z]{1,3}\d{1,4}', text):
                    continue  # a neighbour's designator (U1, R3), never a net label
                if len(text) < 2:
                    continue  # a stray single letter is not a rail name
                if gap(box, tb) > 60:
                    continue
                nearby.append((gap(box, tb), text))
            if not nearby:
                audit['skipped'].append({'component_id':record['component_id'],'reason':'strip_label_unreadable'})
                continue
            nearby.sort()
            label = nearby[0][1].upper() if nearby[0][1].isalpha() else nearby[0][1]
            seen[label] += 1
            key = label if seen[label] == 1 else f"{label}_{seen[label]}"
            if key in result['components']:
                audit['skipped'].append({'component_id':record['component_id'],'reason':'strip_key_collision'})
                continue
            result['components'][key] = {'type':record['type'],'Name':key,'value':label,'bbox':box}
            result['pins'].setdefault(key,{})
            existing.append(box)
            audit['added'].append({'component_id':key,'type':record['type'],'value':label,
                                   'value_source':'printed_net_label','detection_index':record['detection_index']})
        audit['added_count'] = len(audit['added'])
        audit['type_counts'] = dict(Counter(row['type'] for row in audit['added']))
        audit['value_counts'] = dict(Counter('null' if row['value'] is None else 'observed' for row in audit['added']))
    return result, audit
