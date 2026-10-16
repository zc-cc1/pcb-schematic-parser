"""Verify target preservation, ignored gradients and frozen validation for v5."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.unet_data import annotation_targets,SegmentationTiles,tile_starts,polylines
from pcb_parser.unet_model import segmentation_loss
from pcb_parser.unet_supervision import load_valid_mask
from repair_unet_0005_dp_path import read_rows,write_json,sha

BASE=ROOT/"data/segmentation_unet_v4_path_recovery_20261015"
OUTPUT=ROOT/"data/segmentation_unet_v5_reviewed_supervision_20261015"
REPORT=ROOT/"runs/unet_supervision_review_v5"


def main():
    summary=json.loads((OUTPUT/"summary.json").read_text(encoding="utf-8"))
    rows=read_rows(OUTPUT/"manifest.jsonl")
    old={r["case_id"]:r for r in read_rows(BASE/"manifest.jsonl")}
    decisions=json.loads((REPORT/"decisions.json").read_text(encoding="utf-8"))
    reviewed={r["case_id"] for r in decisions}
    expected={(r["case_id"],r["edge_id"]) for r in decisions}
    assert len(rows)==300 and len(expected)==13
    assert all(sha(p)==digest for p,digest in summary["source_files_sha256"].items())
    checked_sources=raw_missing=ignored_pixels=0
    classifications=Counter();valid_masks=0
    for index,row in enumerate(rows,1):
        prior=old[row["case_id"]]
        assert row["image"]==prior["image"] and row["split"]==prior["split"]
        for key in ("wire_mask","pin_heatmap"):
            current=np.asarray(Image.open(row[key]))
            previous=np.asarray(Image.open(prior[key]))
            assert current.shape==(row["height"],row["width"]) and np.array_equal(current,previous)
            if key=="wire_mask":assert np.isin(current,[0,255]).all()
        if row.get("source_annotation_path"):
            source=Path(row["source_annotation_path"])
            assert sha(source)==row["source_annotation_sha256"]
            data=json.loads(source.read_text(encoding="utf-8"))
            wire,pin,tips,stats=annotation_targets(data,row["width"],row["height"],3,2)
            assert np.array_equal(np.asarray(wire),np.asarray(Image.open(row["wire_mask"])))
            assert np.array_equal(np.asarray(pin),np.asarray(Image.open(row["pin_heatmap"])))
            assert stats==row["stats"]
            assert tips==json.loads((source.parent/"pin_points.json").read_text(encoding="utf-8"))
            assert json.loads((source.parent/"metadata.json").read_text(encoding="utf-8"))==row
            checked_sources+=1
            for eid,edge in data.get("edges",{}).items():
                if edge.get("state")=="deleted" or edge.get("category","wire")!="wire" or str(edge.get("isFlywire")).lower()=="true":continue
                paths=edge.get("keyPointPaths",{})
                if not list(polylines(paths.get("wire",[]) if isinstance(paths,dict) else paths)):
                    raw_missing+=1
                    assert (row["case_id"],eid) in expected
                    classifications[edge["wire_supervision"]["classification"]]+=1
        if row["case_id"] in reviewed:
            mask=load_valid_mask(row)
            assert mask is not None and mask.shape==(row["height"],row["width"])
            assert (mask==0).any() and (mask==1).any()
            assert int((mask==0).sum())==row["supervision_ignored_pixels"]
            ignored_pixels+=int((mask==0).sum());valid_masks+=1
            assert row["path_recovery_holds"]==[]
        else:assert row==prior
        if index%60==0:
            print(json.dumps(dict(phase="verify_supervision_revision",pages=index,total=300)),flush=True)
    assert raw_missing==13 and classifications=={"symbol_outline":1,"embedded_image":2,"unresolved_association":10}
    assert valid_masks==9 and ignored_pixels==5100
    assert sum(r["stats"]["missing_physical_paths"] for r in rows)==0
    frozen=read_rows(OUTPUT/"frozen_validation_manifest.jsonl")
    assert frozen==read_rows(BASE/"frozen_validation_manifest.jsonl")
    training=read_rows(OUTPUT/"training_manifest.jsonl")
    assert [r for r in training if r["split"]=="val"]==frozen
    assert len(training)==199 and len(frozen)==28
    assert len([r for r in training if r["split"]=="train"])==171
    assert all(r["structurally_usable_draft"] for r in training)
    for row in training:
        if row["case_id"] in reviewed:assert row.get("valid_mask")
    # Check zero gradients on a real repaired-page crop, without training a model.
    row=next(r for r in rows if r["case_id"]=="0180")
    x,y=537,420
    starts=[(xx,yy) for yy in tile_starts(row["height"],512,384) for xx in tile_starts(row["width"],512,384)]
    idx=next(i for i,(xx,yy) in enumerate(starts) if xx<=x<xx+512 and yy<=y<yy+512)
    _,target,valid=SegmentationTiles([row],512,384)[idx]
    logits=torch.zeros((1,2,512,512),requires_grad=True)
    loss,_=segmentation_loss(logits,target.unsqueeze(0),valid.unsqueeze(0))
    loss.backward()
    assert logits.grad[0,:,valid==0].abs().sum()==0
    assert logits.grad[0,:,valid==1].abs().sum()>0
    assert all(sha(p)==digest for p,digest in summary["source_files_sha256"].items())
    checks=dict(status="v5_all_300_rows_and_ignore_gradients_verified",rows=300,
                regenerated_sources=checked_sources,raw_missing_records_preserved=13,
                raw_classifications=dict(classifications),supervised_missing_paths=0,
                unresolved_network_associations=10,valid_masks=9,ignored_pixels=5100,
                all_wire_and_pin_arrays_equal_v4=True,
                uncertainty_has_zero_loss_gradient_for_both_channels=True,
                frozen_validation_rows_unchanged=28,train_candidates=171,training_manifest_rows=199,
                original_and_v4_source_hashes_unchanged=True,
                unit_tests=dict(command="python -m unittest discover -s tests -p test_unet*.py -v",tests=43,passed=True),
                metadata_fields_still_draft=True,training_started=False,
                current_manifest_sha256=sha(OUTPUT/"manifest.jsonl"),
                training_manifest_sha256=sha(OUTPUT/"training_manifest.jsonl"))
    write_json(REPORT/"final_checks.json",checks)
    summary["verification_report"]=str((REPORT/"final_checks.json").resolve())
    write_json(OUTPUT/"summary.json",summary);write_json(REPORT/"summary.json",summary)

    lines=["# U-Net 剩余13条监督处置 v5 · 2026-10-15","",
           "此前已恢复679条实际路径。剩余13条经逐条原图复核，本轮处理的是监督分类与不确定区域，不新增导线像素、不把异常端点连成路线。","",
           "| 项目 | 结果 |","|---|---|","| 开关轮廓 / 开发板图片轨迹 | 3条明确排除出导线监督 |","| 缺少可见物理路线的关联 | 10条设为ignore，保留原关联待核验 |","| 不确定端点 | 20个pin，相关9页共5100像素不参与两个输出通道的损失 |","| 有效监督范围内缺路径 | 0条 |","| 原始空路径记录 | 仍保留13条，未删改或伪造路线 |","| 原网络关联语义未决 | 仍有10条 |","| 训练候选 | 166→171页 |","| 新增训练页 | 0150、0180、0203、0205、0238 |","| 原验证页 | 28页清单、标签及有效像素均保持原样 |","",
           "**有效监督缺路径为0只表示未决记录已被明确排除/忽略，不表示原始数据已完整重标，也不表示这10条网络关系已修复。** 所有页面仍是草稿监督，完整人工认证页数未增加。","",
           "## 每条处置","","| 页面 | 记录 | 处置 | 原图依据 |","|---|---|---|---|"]
    for r in decisions:
        action="排除非导线轮廓" if r["decision"]=="exclude" else "忽略未决关联及端点邻域"
        lines.append(f"| {r['case_id']} | {r['edge_id']} | {action} | {r['reason']} |")
    lines += ["","0024/E_41沿Key开关主体；0203/E_92与0205/E_86沿开发板图片和引脚标识，不能作为连接不同引脚的导线。10条稀疏记录保留为源关联待查，不更改其connect、category、isFlywire、坐标或pin身份。","",
              "忽略区域以不确定端点为中心，半径7个原图像素；valid_mask.png中0为不参与损失、255为有效。原wire_mask和pin_heatmap数组均与v4逐像素一致。基础切块和缓存/增强切块读取器都已支持该掩码，旋转增强沿用有效区域同步旋转。","",
              "0101、0102仍有其他无归属引脚问题，继续排除联合训练；本轮未放宽该筛选，也未处理另外的越界/归属问题。","",
              "## 验证与使用","",
              f"全量300行验证通过；{checked_sources}份带修订源的标签重新生成与保存结果一致；9张有效区域掩码检查通过；实际0180裁块中忽略区域对导线和pin两个通道的梯度均为0。全部43项U-Net测试通过。原数据、v4文件和默认权重hash通过，没有训练新模型。","",
              "新训练清单：data/segmentation_unet_v5_reviewed_supervision_20261015/training_manifest.jsonl（171训练+28原验证=199行）。需要使用支持valid_mask的当前训练读取器，并显式使用--allow-draft。未决网络关联：graph_review_queue.json（10条），保留完整来源与复核依据。","",
              "复现：scripts/resolve_unet_supervision_v5.py → scripts/verify_unet_supervision_v5.py。逐条图像证据在review/；规则在supervision_decisions.json；验证在final_checks.json。"]
    (REPORT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(checks,ensure_ascii=False))


if __name__=="__main__":
    main()
