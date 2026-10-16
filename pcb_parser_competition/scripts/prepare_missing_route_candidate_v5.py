"""One train-only partial route candidate, with immutable source labels."""
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from unet_audit_common import ROOT,read,write,sha,assert_protected
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import annotation_targets


def main():
    output=ROOT/'runs/unet_BN_order_diagnostic_v5/blue_label_review/0025_candidate';output.mkdir(exist_ok=True)
    manifest=ROOT/'runs/unet_training_label_review_v4/training_manifest.jsonl'
    rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()];row=next(r for r in rows if r['case_id']=='0025')
    assert row['split']==row['source_split']=='train'
    protected=read(ROOT/'runs/unet_negative_sampling_pilot_v4/protected_inputs.json')|read(ROOT/'runs/unet_negative_sampling_pilot_v4/additional_source_attestation.json')
    assert all(sha(p)==v for p,v in protected.items());assert_protected()
    raw={r['case_id']:r for r in map(json.loads,(ROOT/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    annotation,_=load_source_annotation(raw['0025']['raw_annotation_path'])
    rgb=np.asarray(Image.open(row['image']).convert('RGB'));gt=np.asarray(Image.open(row['wire_mask']))>0
    wire,pin,*_=annotation_targets(annotation,row['width'],row['height'],3,2.)
    assert np.array_equal(np.asarray(wire)>0,gt)
    points=[[14,369],[340,369]];values=rgb[369,14:341].astype(int)
    assert np.all(values[:,2]-values[:,0]>=40) and np.all(values[:,2]-values[:,1]>=40)
    changed=copy.deepcopy(annotation)
    eid='agent_visual_KEY4_partial_v5';assert eid not in changed['edges']
    changed['edges'][eid]=dict(category='wire',state='active',isFlywire=False,keyPointPaths={'wire':[points]})
    newwire,newpin,*_=annotation_targets(changed,row['width'],row['height'],3,2.)
    newgt=np.asarray(newwire)>0;assert np.array_equal(np.asarray(pin),np.asarray(newpin))
    assert np.array_equal(np.asarray(newpin),np.asarray(Image.open(row['pin_heatmap'])))
    add=newgt&~gt;removed=gt&~newgt;assert not removed.any()
    box=[10,365,343,374];x0,y0,x1,y1=box;allowed=np.zeros_like(gt);allowed[y0:y1,x0:x1]=True
    assert add.any() and not (add&~allowed).any()
    newwire.save(output/'wire_mask_candidate.png')
    config=dict(status='local_agent_visual_candidate_not_applied',case_id='0025',split='train',wire_stroke=3,
        new_edge_id=eid,path=points,ROI=box,added_pixels=int(add.sum()),removed_pixels=0,
        base_wire_sha256=sha(row['wire_mask']),candidate_wire_sha256=sha(output/'wire_mask_candidate.png'),
        pin_heatmap_unchanged=True,pin_points_changed=0,source_annotation_path=raw['0025']['raw_annotation_path'],source_annotation_sha256=sha(raw['0025']['raw_annotation_path']),
        evidence='Original KEY4 continuous navy wire and source missing geometry; candidate covers only inspected free span x14..340 at y369.',
        pending='R7 junction, route toward switch tip, other blue routes and full-page labels still need review; no graph/net association assigned.',
        independent_annotation=False,full_page_certified=False,automatic_training_started=False,
        current_training_manifest_changed=False,calibration_or_old26_labels_changed=False,production_adopted=False)
    write(output/'candidate.json',config);write(ROOT/'configs/unet_missing_route_candidate_v5.json',config)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),dpi=170,layout='constrained')
    axes[0].imshow(rgb);axes[0].set_title('0025 训练原图')
    for ax,mask,title in ((axes[1],gt,'现有GT'),(axes[2],newgt,'局部候选：仅补KEY4自由线段')):
        over=rgb.copy();over[mask]=(.2*over[mask]+.8*np.array([20,180,95])).astype(np.uint8)
        if ax is axes[2]:over[add]=(.15*over[add]+.85*np.array([235,115,30])).astype(np.uint8)
        ax.imshow(over);ax.set_title(title)
    for ax in axes:ax.axis('off')
    fig.suptitle(f"原数据保持不变 · 橙=新增{int(add.sum())}个监督像素 · pin监督不变 · 全页未认证")
    fig.savefig(output/'comparison.png',facecolor='white');plt.close(fig)
    assert all(sha(p)==v for p,v in protected.items());assert_protected()
    print(json.dumps(dict(candidate=str(output/'candidate.json'),added_pixels=int(add.sum()),original_files_unchanged=True)))


if __name__=='__main__':main()
