"""Offline paired draft scoring. This module is never imported by inference."""
import argparse
from collections import Counter
from pathlib import Path

from run_component_pin_library import ROOT, read, save, sha
from evaluate_pin_text import component_matches, match, conditional_fields, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'runs/s14_component_library')
    args = parser.parse_args()
    batch = read(args.output/'predictions/batch_summary.json')
    if not batch.get('completed'):
        raise ValueError('Incomplete predictions')
    reference_path = ROOT/'runs/s6_pin_text_v1/draft_field_references.json'
    refs = read(reference_path)
    aggregate = {key:Counter() for key in ['baseline_s11', 'library']}
    cases, changes = [], []
    for row in batch['cases']:
        case = row['case_id']
        old = read(ROOT/f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json')
        new = read(args.output/f'predictions/{case}/pin_text/pin_text_result.json')
        if old['image_sha256'] != refs['image_sha256'][case] or new['image_sha256'] != old['image_sha256']:
            raise ValueError('Reference PNG differs')
        truth = [t for t in refs['references'] if t['case_id']==case]
        owner_map = component_matches([{'class_name':c['class_name'], 'bbox':c['bbox_top_left']} for c in old['components']],
                                      refs['detector_references'][case]['objects'])
        record = {'case_id':case}
        for version, data in [('baseline_s11',old),('library',new)]:
            predicted = [p for p in data['pin_candidates'] if p['owner_status']=='assigned']
            pairs = match(predicted, truth, owner_map)
            counts = conditional_fields(predicted, truth, pairs)
            aggregate[version].update(counts)
            values = {'conditional':{field:score(counts.get(field+'_correct',0), counts.get(field+'_emitted',0), counts.get(field+'_eligible',0)) for field in ['number','pinname']}}
            for field in ['number','pinname']:
                emitted = [p for p in predicted if p.get(field) is not None]
                expected = [t for t in truth if t['expected_numbers']] if field=='number' else [t for t in truth if t['expected_name']]
                matches = match(emitted, expected, owner_map, field)
                values['joint_'+field+'_location'] = score(len(matches),len(emitted),len(expected))
                aggregate[version].update({field+'_joint_tp':len(matches), field+'_joint_emitted':len(emitted), field+'_joint_expected':len(expected)})
            record[version] = values
        original_owned = [p for p in old['pin_candidates'] if p['owner_status']=='assigned']
        matched_truth = {original_owned[i]['candidate_id']:truth[j] for i,j in match(original_owned,truth,owner_map)}
        audit = read(args.output/f'predictions/{case}/library_binding_audit.json')
        for change in audit['changes']:
            target = matched_truth.get(change['candidate_id'])
            field = change['field']
            expected = target['expected_numbers'] if target and field=='number' else target['expected_name'] if target else None
            scorable = bool(expected) and (field!='number' or len(expected)==1)
            correct = (change['after'] in expected if field=='number' else change['after']==expected) if scorable else None
            changes.append({'case_id':case, 'candidate_id':change['candidate_id'], 'field':field,
                            'before':change['before'], 'after':change['after'], 'owner_location_draft_match':target is not None,
                            'draft_expected':expected, 'draft_field_scorable':scorable, 'after_matches_draft':correct})
        cases.append(record)
    totals = {}
    for version,c in aggregate.items():
        totals[version] = {'conditional':{field:score(c[field+'_correct'],c[field+'_emitted'],c[field+'_eligible']) for field in ['number','pinname']},
                           **{'joint_'+field+'_location':score(c[field+'_joint_tp'],c[field+'_joint_emitted'],c[field+'_joint_expected']) for field in ['number','pinname']}}
    numbers = [c for c in changes if c['field']=='number' and c['before'] is None and c['after'] is not None]
    save(args.output/'evaluation.json', {'cases':cases, 'totals':totals, 'changes':changes,
         'new_library_numbers':{'emitted':len(numbers), 'owner_location_draft_matched':sum(c['owner_location_draft_match'] for c in numbers),
           'conditional_draft_scorable':sum(c['draft_field_scorable'] for c in numbers),
           'conditional_draft_correct':sum(c['after_matches_draft'] is True for c in numbers)},
         'reference_sha256':sha(reference_path), 'official_PinF1':None, 'official_NetF1':None,
         'engineering_test_independent':False, 'metric_scope':'Same 26 development pages; draft fields after owner IoU>=0.5 and point distance<=5px. Not competition PinF1.',
         'reference_limits':'Missing number/name fields excluded. Some draft strings visibly disagree with PNG. No reference labels modified.'})
    print(totals,flush=True)


if __name__=='__main__':
    main()
