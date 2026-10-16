"""Validate audit invariants, bootstrap page differences, and render evidence."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from unet_audit_common import ROOT, OUT, read, write, sha, image, truth, probability, metrics, aggregate, val_cases, assert_protected


def main():
    b=read(OUT/"baseline_reproduction.json");e=read(OUT/"error_decomposition.json");a=read(OUT/"inference_audit.json");h=read(OUT/"experiment_history.json")
    reviewed=read(OUT/"manual_review30.json")
    assert len(reviewed["records"])==30 and len(set(r["case_id"] for r in reviewed["records"]))==30
    assert sum(e["partitions"].values())==b["micro"]["fp"]+b["micro"]["fn"]
    baseline=next(r for r in a["summary"] if r["variant"]=="native_uniform512")
    assert baseline["micro"]==b["micro"] and baseline["connected_paths"]==1150 and baseline["total_paths"]==1548
    for r in a["summary"]:
        assert r["pages"]==26 and r["micro"]["tp"]+r["micro"]["fn"]==254422 and r["total_paths"]==1548 and not r["adopted"]
    for name in ("control","contrast","rotation"):
        old=next(r for r in h["experiments"] if r["id"]==name)
        fresh=next(r for r in a["summary"] if r["variant"]==name)
        assert old["wire_micro"]==fresh["micro"] and old["connected_paths"]==fresh["connected_paths"]
    # Every D4 transform restores exactly, independent of the model.
    grid=np.arange(6*9*3).reshape(6,9,3)
    for flipped in (False,True):
        for k in range(4):
            transformed=np.rot90(np.fliplr(grid) if flipped else grid,k)
            restored=np.rot90(transformed,-k)
            if flipped:restored=np.fliplr(restored)
            assert np.array_equal(grid,restored)
    rng=np.random.default_rng(20261014);draw=rng.integers(0,26,size=(10000,26));mult=np.stack([(draw==i).sum(axis=1) for i in range(26)],axis=1)
    byname={name:{r["case_id"]:r for r in a["records"] if r["variant"]==name} for name in (r["variant"] for r in a["summary"])}
    def boot_dice(name):
        matrix=np.array([[byname[name][c]["wire"][k] for k in ("tp","fp","fn")] for c in val_cases()])
        total=mult@matrix
        return 2*total[:,0]/(2*total[:,0]+total[:,1]+total[:,2])
    base=boot_dice("native_uniform512");bootstrap=[]
    for r in a["summary"]:
        delta=100*(boot_dice(r["variant"])-base)
        bootstrap.append(dict(variant=r["variant"],delta_pp=r["delta_dice_pp"],paired_page_bootstrap95_pp=np.quantile(delta,[.025,.975]).tolist()))
    tp,fp,fn=(b["micro"][k] for k in ("tp","fp","fn"))
    besttile=max(h["original_history"],key=lambda r:r["val_wire_pixels"]["wire_pixel_dice"])
    result=dict(status="audit_invariants_passed",protected_files=assert_protected(),bootstrap=bootstrap,
        bootstrap_scope="10000 paired resamples of 26 pages; seed20261014; dependence by unknown engineering source is NOT corrected; diagnostic intervals not independent test confidence",
        train_fullpage=aggregate([r for r in b["records"] if r["split"]=="train"]),
        train_val_gap_pp=100*(aggregate([r for r in b["records"] if r["split"]=="train"])["dice"]-b["micro"]["dice"]),
        oracle_remove_all_farFP=metrics(tp,fp-e["partitions"]["FP_far_GT2"],fn),oracle_remove_all_0170_FP=metrics(tp,fp-next(r["wire"]["fp"] for r in b["records"] if r["case_id"]=="0170"),fn),
        oracle_is_not_expected_gain=True,maximum_historical_v1_tile_Dice_epoch=besttile["epoch"],
        maximum_historical_v1_tile_Dice=besttile["val_wire_pixels"]["wire_pixel_dice"],
        min_loss_selected_v1_tile_Dice=h["original_best_record"]["val_wire_pixels"]["wire_pixel_dice"],training_started=False)
    write(OUT/"final_checks.json",result)
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    fig,axes=plt.subplots(2,2,figsize=(13,9),dpi=160,layout="constrained")
    labels=["FP 距GT>2px","FP 距GT≤2px","FN 距预测>2px","FN 距预测≤2px"]
    keys=["FP_far_GT2","FP_near_GT2","FN_far_prediction2","FN_near_prediction2"]
    values=[100*e["partition_error_fractions"][k] for k in keys]
    ax=axes[0,0];ax.barh(labels,values,color=["#d64545","#eaa1a1","#3f74b8","#91b5de"]);ax.invert_yaxis();ax.set(xlabel="占全部FP+FN（%）",title="主导误差：远离GT的FP")
    for i,v in enumerate(values):ax.text(v+.4,i,f"{v:.2f}% / {e['partitions'][keys[i]]}",va="center")
    ax.set_xlim(0,80)
    curves=h["original_history"];ax=axes[0,1]
    ax.plot([r["epoch"] for r in curves],[100*r["train_wire_pixels"]["wire_pixel_dice"] for r in curves],label="训练 tile")
    ax.plot([r["epoch"] for r in curves],[100*r["val_wire_pixels"]["wire_pixel_dice"] for r in curves],label="验证 tile（28页）")
    ax.axvline(57,ls="--",color="#888");ax.legend();ax.set(title="原模型训练史：有泛化差距",xlabel="epoch",ylabel="Dice@0.5（%）")
    ax=axes[1,0];selected=["scale_mean","d4_mean","d4_multiscale_mean","gaussian512","uniform1024","gaussian1024","ensemble_all4","closing3","remove_area16"]
    shown=[next(r for r in bootstrap if r["variant"]==n) for n in selected]
    ax.barh(selected,[r["delta_pp"] for r in shown],color=["#389176" if r["delta_pp"]>0 else "#bb6868" for r in shown]);ax.invert_yaxis();ax.axvline(0,color="#555");ax.set(title="推理对照：没有大幅、同时保持拓扑的收益",xlabel="相对79.4613%的ΔDice（百分点）")
    ax=axes[1,1]
    for group,label in (("train103","训练103页"),("val26_unique","验证26页")):
        sizes=np.array(b["stats"][group]["original_sizes"]);ax.scatter(sizes[:,0],sizes[:,1],s=15,alpha=.6,label=label)
    ax.legend();ax.set(title="实际原图尺寸：输入采用原尺度512裁块",xlabel="宽（px）",ylabel="高（px）")
    fig.savefig(OUT/"audit_summary.png",facecolor="white");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),dpi=160,layout="constrained")
    for ax,name in zip(axes,("control","contrast","rotation")):
        path=ROOT/("runs/unet_retrain_rotation_v3/rotation/history.json" if name=="rotation" else f"runs/unet_retrain_v2/{name}/history.json")
        history=read(path)
        ax.plot([r["epoch"] for r in history],[100*r["train_wire_pixels"]["wire_pixel_dice"] for r in history],label="train tile")
        ax.plot([r["epoch"] for r in history],[100*r["val_wire_pixels"]["wire_pixel_dice"] for r in history],label="val native tile")
        ax.set(title=name,xlabel="epoch",ylabel="Dice@0.5（%）");ax.legend();ax.grid(alpha=.2)
    fig.suptitle("历史微调曲线：rotation训练指标包含旋转样本，不能直接比较原始train分布")
    fig.savefig(OUT/"warmstart_curves.png",facecolor="white");plt.close(fig)
    # Compact gallery links to the legible, separately saved four-panel pages.
    samples=read(OUT/"review30/manifest.json")["samples"][:20]
    canvas=Image.new("RGB",(1800,5*405),(245,245,245))
    for i,r in enumerate(samples):
        tile=Image.open(r["overview"]).convert("RGB");tile.thumbnail((450,400));canvas.paste(tile,((i%4)*450,(i//4)*405))
    canvas.save(OUT/"worst20_contact_sheet.png")
    artifacts={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted(OUT.rglob("*.json")) if p.name!="artifact_manifest.json"}
    scripts={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted((ROOT/"scripts").glob("*unet*audit*.py"))}
    scripts.update({str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted((ROOT/"scripts").glob("audit_unet*.py"))})
    write(OUT/"artifact_manifest.json",dict(audit_artifacts_sha256=artifacts,audit_scripts_sha256=scripts,protected_source_files=assert_protected(),default_checkpoint_unchanged=True))
    print(json.dumps(dict(checks=result,output=str(OUT)),ensure_ascii=False))


if __name__=="__main__":main()
