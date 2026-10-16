"""Materialize assistant image reviews; preserve the original review queue."""
from collections import Counter
from copy import deepcopy

from run_wire_graph import ROOT,read,save,sha


def main():
    original=ROOT/'data/topology_review_v1/annotations.json'
    records=[deepcopy(r) for r in read(original)['records'] if r['kind']=='junction']
    train={'0018','0022','0026','0049','0131','0170','0240'}
    calibration={'0033','0042','0105'}
    test={'0002','0096','0148','0149','0227','0236','0264'}
    connected={f'0018_junction_{i:03d}' for i in range(1,5)}
    connected|={f'0049_junction_{i:03d}' for i in range(1,4)}
    connected|={f'0131_junction_{i:03d}' for i in range(1,4)}
    connected|={f'0148_junction_{i:03d}' for i in range(1,4)}
    connected|={f'0170_junction_{i:03d}' for i in range(1,6)}
    connected|={'0042_junction_010','0105_junction_001','0149_junction_001','0227_junction_001','0236_junction_001'}
    for row in records:
        name=row['review_id']
        if name=='0012_junction_001':
            label='uncertain';note='Candidate is offset from a dot inside highlighted, closely spaced pin tracks; exact arm correspondence needs review.'
        elif name=='0105_junction_002':
            label='segmentation_artifact';note='The candidate center falls in the 15k(NC) text, not at a wire crossing.'
        elif name in connected:
            label='connected_junction';note='A filled connection dot is visible at the candidate center; some actual T connections produce four skeleton arms.'
        else:
            label='crossing_without_connection';note='The center has crossing strokes without a dot; nearby intended connections on this page have visible dots.'
        split='train' if row['case_id'] in train else 'calibration' if row['case_id'] in calibration else 'test' if row['case_id'] in test else 'excluded'
        row.update(label=label,reviewed=True,reviewer='assistant_visual_inspection',review_note=note,
                   split=split,training_eligible=label in {'connected_junction','crossing_without_connection'},
                   reference_status='assistant_visual_development_reference_not_user_or_official_Golden')
    assert not train&calibration and not train&test and not calibration&test
    assert len({r['source_image_sha256'] for r in records})==len({r['case_id'] for r in records})
    result={'records':records,'original_queue_sha256':sha(original),
            'reviewer':'assistant_visual_inspection_of_three_contact_sheets_and_four_original_crops',
            'labels_inferred_from_reference_nets':False,'original_annotations_modified':False,
            'split_groups':{'train':sorted(train),'calibration':sorted(calibration),'test':sorted(test),'excluded':['0012']},
            'label_counts':dict(Counter(r['label'] for r in records)),
            'split_counts':dict(Counter(r['split'] for r in records)),
            'scope':'Previously used upstream validation pages. Test groups are unseen only by this junction classifier, not by overall pipeline development.',
            'artifact_class_trainable':False}
    path=ROOT/'data/junction_classifier_v1/reviewed_annotations.json';save(path,result)
    print(result['label_counts'],result['split_counts'])


if __name__=='__main__':main()
