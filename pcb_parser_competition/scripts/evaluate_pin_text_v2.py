"""Paired offline evaluation. No reference imports in PNG inference."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from run_pin_text_v2 import ROOT,read,save,sha
from evaluate_pin_text import component_matches,match,conditional_fields,score


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/s11_pin_text_v2');a=p.parse_args()
    refs=read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json');aggregate={key:Counter() for key in ['v1','v2','full_v2_experiment']};cases=[]
    batch=read(a.output/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('Batch incomplete; wait for predictions to finish before scoring')
    config_hash=hashlib.sha256(json.dumps(batch['config'],sort_keys=True).encode()).hexdigest()
    for row in batch['cases']:
        case=row['case_id'];new=read(a.output/f'predictions/{case}/pin_text_result.json');old=read(ROOT/f'runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json')
        if new['pin_text_config_sha256']!=config_hash:raise ValueError('Mixed pin-text configurations in batch')
        if new['image_sha256']!=refs['image_sha256'][case] or old['image_sha256']!=new['image_sha256']:raise ValueError('Reference/image mismatch')
        truth=[r for r in refs['references'] if r['case_id']==case];named_truth=[r for r in truth if r['expected_name']]
        objects=refs['detector_references'][case]['objects'];om=component_matches([{'class_name':c['class_name'],'bbox':c['bbox_top_left']} for c in old['components']],objects)
        record={'case_id':case}
        experimental=read(a.output/f'predictions/{case}/experimental_pin_text_result.json')
        for version,data in [('v1',old),('v2',new),('full_v2_experiment',experimental)]:
            predicted=[p for p in data['pin_candidates'] if p['owner_status']=='assigned'];pairs=match(predicted,truth,om)
            conditional=conditional_fields(predicted,truth,pairs);aggregate[version].update(conditional)
            named=[p for p in predicted if p['pinname'] is not None];matches=match(named,named_truth,om,'pinname')
            aggregate[version].update(name_tp=len(matches),name_emitted=len(named),name_expected=len(named_truth))
            record[version]={'conditional':conditional,'joint_name_location':score(len(matches),len(named),len(named_truth)),
                             'localized_owned':len(pairs),'summary':data['pin_text_summary']}
            for key in ['number_observed','name_observed','number_and_name_observed','component_identity_resolved','new_header_identities','new_model_names','secondary_tip_hypotheses_text_withheld']:
                aggregate[version][key]+=data['pin_text_summary'].get(key,0)
        if [(p['candidate_id'],p['point'],p.get('owner_detection_index'),p['owner_status']) for p in old['pin_candidates']] != [(p['candidate_id'],p['point'],p.get('owner_detection_index'),p['owner_status']) for p in new['pin_candidates']]:
            raise ValueError('Localization/ownership changed')
        cases.append(record)
    totals={}
    for version,c in aggregate.items():
        totals[version]={'counts':{k:c[k] for k in ['number_observed','name_observed','number_and_name_observed','component_identity_resolved','new_header_identities','new_model_names','secondary_tip_hypotheses_text_withheld']},
                         'joint_name_location':score(c['name_tp'],c['name_emitted'],c['name_expected']),
                         'conditional':{field:score(c[field+'_correct'],c[field+'_emitted'],c[field+'_eligible']) for field in ['number','pinname']}}
    result={'cases':cases,'totals':totals,'official_PinF1':None,'engineering_test_independent':False,
            'all_original_pin_points_preserved':True,'ownership_assignments_preserved':True,
            'metric_scope':'Same frozen 26 development pages and draft reference, matched owner/point <=5px; not official number-key-based PinF1',
            'reference_limits':'Some source strings and identities are missing or visibly disagree with image glyphs. Empty source fields excluded; no source labels changed.',
            'config_sha256':config_hash}
    save(a.output/'evaluation.json',result)
    print(totals,flush=True)


if __name__=='__main__':main()
