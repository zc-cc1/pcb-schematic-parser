"""Direct BN-buffer intervention on original epoch57; no validation fitting."""
import json
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from unet_audit_common import ROOT,OUT,read,write,sha,rows,image,truth,counts,metrics,aggregate,val_cases,assert_protected
from reestimate_unet_BN_v4 import RGBOnlyCrops,fixed_weights_hash
from pcb_parser.unet_model import WirePinUNet
from pcb_parser.unet_data import predict_tiled,annotation_targets
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_inference import decode_pins
from verify_unet_contrast import path_specs,path_records,pin_metrics


def main():
    run=ROOT/"runs/unet_v1_BN_diagnostic_v4";protocol=read(run/"protocol.json")
    assert sha(__file__)==protocol["script_sha256"]
    assert sha(ROOT/"weights/unet_wire_pin_v1.pt")==protocol["source_checkpoint_sha256"]
    assert_protected();torch.set_num_threads(4);start=time.perf_counter()
    train=[dict(r,source_split="train") for r in rows() if r["split"]=="train"]
    assert len(train)==103 and sorted(r["case_id"] for r in train)==protocol["train_cases"]
    dataset=RGBOnlyCrops(train);assert len(dataset)==986
    source=ROOT/"weights/unet_wire_pin_v1.pt";ckpt=torch.load(source,map_location="cpu",weights_only=True)
    assert ckpt["epoch"]==57
    model=WirePinUNet(ckpt["base_channels"]).to("cuda:0");model.load_state_dict(ckpt["state_dict"],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    bns=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)];assert len(bns)==18
    for bn in bns:bn.reset_running_stats();bn.momentum=None;bn.train()
    loader=DataLoader(dataset,batch_size=protocol["batch_size"],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(protocol["order_seed"]))
    with torch.inference_mode():
        for index,images in enumerate(loader):
            model(images.to("cuda:0"))
            if (index+1)%20==0:print(json.dumps(dict(phase="BN_train_RGB_only",batches=index+1,total=len(loader))),flush=True)
    model.eval();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert fixed_weights_hash(state)==fixed_weights_hash(ckpt["state_dict"])
    changed=[k for k,v in state.items() if not torch.equal(v,ckpt["state_dict"][k])]
    assert len(changed)==54 and all(k.endswith(("running_mean","running_var","num_batches_tracked")) for k in changed)
    checkpoint=run/"diagnostic.pt";torch.save(dict(ckpt,state_dict=state,kind="train_only_BN_buffer_diagnostic",source_sha256=sha(source)),checkpoint)
    weightsha=sha(checkpoint);records=[]
    raw={r["case_id"]:r for r in map(json.loads,(ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    for index,case in enumerate(val_cases()):
        rgb=image(case);folder=run/case;folder.mkdir(exist_ok=True)
        with torch.inference_mode():p=predict_tiled(model,rgb,"cuda:0",512,384)
        np.savez_compressed(folder/"probabilities.npz",wire=p[0],pin=p[1])
        write(folder/"prediction_meta.json",dict(image_only=True,checkpoint_sha256=weightsha,image_sha256=sha(ROOT/f"data/standardized/{case}/{case}.png"),tile_size=512,stride=384,resize=False))
        # Existing ground truth enters after image-only prediction.
        gt=truth(case);annotation,_=load_source_annotation(raw[case]["raw_annotation_path"])
        reference,*_=annotation_targets(annotation,rgb.shape[1],rgb.shape[0],3,2.)
        assert np.array_equal(np.asarray(reference)>0,gt)
        specs=path_specs(annotation,gt.shape);tips=read(ROOT/f"data/segmentation_unet_v1/{case}/pin_points.json")
        with np.load(OUT/f"baseline/{case}/probabilities.npz",allow_pickle=False) as saved:
            original=np.stack([saved["wire"],saved["pin_tip"]])
        assert read(OUT/f"baseline/{case}/prediction_meta.json")["checkpoint_sha256"]==sha(source)
        for name,values in (("v1_original",original),("v1_BN_reestimated",p)):
            mask=values[0]>=.5;paths=path_records(mask,specs)
            pins=decode_pins(values[1],threshold=.8,kernel=5,refinement="quadratic",nms_radius=0.,limit=500)
            r=dict(case_id=case,arm=name,wire=metrics(*counts(mask,gt)),pin=pin_metrics(pins,tips),
                connected_paths=sum(r["connected_within_reference_tube"] for r in paths),total_paths=len(paths))
            records.append(r);write(folder/(name+"_metrics.json"),r)
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/(name+"_wire.png"))
        print(json.dumps(dict(phase="frozen_external26",case_id=case,completed=index+1,total=26)),flush=True)
    summary={}
    for name in ("v1_original","v1_BN_reestimated"):
        selected=[r for r in records if r["arm"]==name]
        summary[name]=dict(wire=aggregate(selected),pin=aggregate(selected,"pin"),connected_paths=sum(r["connected_paths"] for r in selected),total_paths=sum(r["total_paths"] for r in selected))
    assert abs(summary["v1_original"]["wire"]["dice"]-.7946126866)<1e-9
    ids=sorted(val_cases());draw=np.random.default_rng(20261014).integers(0,26,(10000,26));sampled={}
    for name in summary:
        pair={r["case_id"]:r for r in records if r["arm"]==name}
        matrix=np.array([[pair[c]["wire"][k] for k in ("tp","fp","fn")] for c in ids]);total=matrix[draw].sum(axis=1)
        sampled[name]=2*total[:,0]/np.maximum(1,2*total[:,0]+total[:,1]+total[:,2])
    a=summary["v1_original"];b=summary["v1_BN_reestimated"]
    delta=dict(Dice_pp=100*(b["wire"]["dice"]-a["wire"]["dice"]),P_pp=100*(b["wire"]["precision"]-a["wire"]["precision"]),
        R_pp=100*(b["wire"]["recall"]-a["wire"]["recall"]),connected_paths=b["connected_paths"]-a["connected_paths"],
        paired_page_bootstrap95_Dice_pp=np.quantile(100*(sampled["v1_BN_reestimated"]-sampled["v1_original"]),[.025,.975]).tolist())
    probes=[]
    for name in summary:
        target=OUT/"baseline/0036/probabilities.npz" if name=="v1_original" else run/"0036/probabilities.npz"
        with np.load(target,allow_pickle=False) as saved:values=saved["wire"]
        for y,x0,x1,label in ((80,620,681,"X1_fixed61"),(181,605,629,"DOUT_fixed24")):
            v=values[y,x0:x1];probes.append(dict(arm=name,probe=label,detected_pixels=int((v>=.5).sum()),pixels=len(v),median_probability=float(np.median(v))))
    result=dict(status="original_v1_BN_diagnostic_complete",summary=summary,delta=delta,records=records,fixed_probes=probes,
        learned_weights_unchanged=True,learned_weights_sha256=fixed_weights_hash(state),changed_buffers=changed,gradient_updates=0,
        original_checkpoint_unchanged=True,source_checkpoint_sha256=sha(source),diagnostic_checkpoint_sha256=weightsha,
        estimation_images="original103 train only",validation_images_used_for_estimation=0,parameter_selection_on_external26=False,
        independent_test=False,production_adopted=False,elapsed_seconds=time.perf_counter()-start,protected_original_files=assert_protected())
    assert sha(__file__)==protocol["script_sha256"]
    write(run/"evaluation.json",result)
    lines=["# 原v1的BN运行统计诊断", "", "相同epoch57可学习权重；只使用原103训练页的RGB重估BN，一遍986裁块、batch8、seed20261014累计平均。没有读取26验证图来估计统计量或选参数，无梯度更新。旧26页用于固定诊断，不能称独立test。", "",
        "| checkpoint | Dice@0.5 | P | R | 连通路径 | pin F1@0.8 |", "|---|---:|---:|---:|---|---:|"]
    for name,r in summary.items():lines.append(f"| {name} | {r['wire']['dice']:.4%} | {r['wire']['precision']:.4%} | {r['wire']['recall']:.4%} | {r['connected_paths']}/{r['total_paths']} | {r['pin']['dice']:.4%} |")
    lines += ["",f"ΔDice={delta['Dice_pp']:+.4f}个百分点，ΔRecall={delta['R_pp']:+.4f}点，Δ连通={delta['connected_paths']:+d}条；配对页bootstrap95% ΔDice={delta['paired_page_bootstrap95_Dice_pp']}。未知工程内依赖未校正。", "",
        "这一干预直接测量BN运行缓冲量对原v1的影响。若Dice/Recall/连通不改善，则该固定重估方案不能解决原瓶颈；即使改善也不证明全部根因是BN、不证明GroupNorm重训有效。默认权重保持原样。", "",
        "固定先前审计的0036线段覆盖（像素覆盖不是整网连通）：", "", "| 线段 | checkpoint | ≥0.5像素/长度 | p中位数 |", "|---|---|---|---:|"]
    for r in probes:lines.append(f"| {r['probe']} | {r['arm']} | {r['detected_pixels']}/{r['pixels']} | {r['median_probability']:.6f} |")
    lines += ["", "协议与数值：protocol.json / evaluation.json。原679项保护文件hash通过，全部可学习权重hash完全一致。", "", "```powershell", "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\diagnose_original_unet_BN_v4.py", "```"]
    (run/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(summary=summary,delta=delta,elapsed_seconds=result["elapsed_seconds"])))


if __name__=="__main__":main()
