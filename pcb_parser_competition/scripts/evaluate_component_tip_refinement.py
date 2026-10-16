"""Offline draft position+owner evaluation. Never imports references into inference."""
from collections import Counter
from run_component_tip_refinement import ROOT,read,save
from evaluate_pin_text import component_matches,match,score


def main():
    stage=ROOT/'runs/s13_component_tips';batch=read(stage/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('Refinement batch incomplete')
    refs=read(ROOT/'runs/s6_pin_text_v1/draft_field_references.json');split=read(ROOT/'configs/unet_inference_v2.json')
    rows=[];all_added=[]
    for row in batch['cases']:
        case=row['case_id'];old=read(ROOT/f'runs/s11_pin_text_v2/predictions/{case}/pin_text_result.json')
        new=read(stage/f'predictions/{case}/pin_text_result.json');audit=read(stage/f'predictions/{case}/refinement_audit.json')
        truth=[r for r in refs['references'] if r['case_id']==case]
        cm=component_matches([{'bbox':c['bbox_top_left'],'class_name':c['class_name']} for c in old['components']],refs['detector_references'][case]['objects'])
        before=[p for p in old['pin_candidates'] if p['owner_status']=='assigned'];after=[p for p in new['pin_candidates'] if p['owner_status']=='assigned']
        bp=match(before,truth,cm);ap=match(after,truth,cm)
        # Report marginal new matches using a one-to-one match that first
        # reserves all original matches. Never give two predictions one truth.
        used={j for _,j in bp};remaining=[t for j,t in enumerate(truth) if j not in used]
        added=new['pin_candidates'][len(old['pin_candidates']):];marginal=match(added,remaining,cm)
        good={i for i,_ in marginal}
        graph=read(stage/f'predictions/{case}/physical/graph_proposals.json');attachments={p['candidate_id']:p for p in graph['pin_attachments']}
        for i,p in enumerate(added):
            typ=next(c['type'] for c in old['components'] if c['detection_index']==p['owner_detection_index'])
            all_added.append({'case_id':case,'candidate_id':p['candidate_id'],'component_type':typ,'point':p['point'],
                              'draft_incremental_owner_location_match':i in good,'attachment_status':attachments[p['candidate_id']]['attachment_status'],
                              'wire_gap_px':attachments[p['candidate_id']]['wire_gap_px'],'visual_reviewed':False,
                              'unmatched_can_mean_draft_missing_or_wrong_prediction':True})
        rows.append({'case_id':case,'subset':'prior_calibration' if case in split['calibration_cases'] else 'prior_acceptance',
            'before':score(len(bp),len(before),len(truth)),'after':score(len(ap),len(after),len(truth)),
            'added':len(added),'marginal_correct':len(marginal),'graph_edges_changed':row['graph_edges_changed'],
            'body_guard_mask_changed_pixels':row['body_guard_mask_changed_pixels'],
            'before_power_missing_owned_tip':read(ROOT/f'runs/s12_pin_output_rules/predictions/{case}/summary.json')['power_guard']['power_without_owned_tip'],
            'after_power_missing_owned_tip':row['power_guard_after']['power_without_owned_tip']})
    totals={}
    for subset in ['prior_calibration','prior_acceptance','all']:
        selected=rows if subset=='all' else [r for r in rows if r['subset']==subset]
        result={}
        for version in ['before','after']:
            result[version]=score(sum(r[version]['correct'] for r in selected),sum(r[version]['emitted'] for r in selected),sum(r[version]['expected'] for r in selected))
        result.update(added=sum(r['added'] for r in selected),marginal_correct=sum(r['marginal_correct'] for r in selected))
        totals[subset]=result
    totals['added_by_type']=dict(Counter(p['component_type'] for p in all_added))
    totals['incremental_matches_by_type']=dict(Counter(p['component_type'] for p in all_added if p['draft_incremental_owner_location_match']))
    totals['added_attachment_status']=dict(Counter(p['attachment_status'] for p in all_added))
    totals['power_missing_owned_tip_before']=sum(r['before_power_missing_owned_tip'] for r in rows)
    totals['power_missing_owned_tip_after']=sum(r['after_power_missing_owned_tip'] for r in rows)
    save(stage/'evaluation.json',{'scope':'26 old development pages, draft references, <=5px AND matched component owner; not official PinF1 or independent test',
          'aggregate':totals,'cases':rows,'added_review_queue':all_added,'default_pipeline_adopted':False,
          'reference_annotations_modified':False,'new_training':False,'official_PinF1':None,'official_NetF1':None})
    print(totals,flush=True)


if __name__=='__main__':main()
