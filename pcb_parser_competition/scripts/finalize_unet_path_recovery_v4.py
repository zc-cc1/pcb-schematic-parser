"""Independently verify all final targets, preserve original validation, and report."""
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcb_parser.unet_data import annotation_targets
from repair_unet_0005_dp_path import sha,read_rows,write_json

OUTPUT=ROOT/"data/segmentation_unet_v4_path_recovery_20261015"
REPORT=ROOT/"runs/unet_path_recovery_v4"


def main():
    summary=json.loads((OUTPUT/"summary.json").read_text(encoding="utf-8"))
    assert summary["manual_sparse_repairs_applied"]
    final_rows=read_rows(OUTPUT/"manifest.jsonl")
    base_rows=read_rows(ROOT/"data/segmentation_unet_v1/manifest.jsonl")
    base={r["case_id"]:r for r in base_rows}
    rows={r["case_id"]:r for r in final_rows}
    holds=json.loads((REPORT/"holds.json").read_text(encoding="utf-8"))
    repairs=json.loads((REPORT/"repairs.json").read_text(encoding="utf-8"))
    protected=dict(summary["source_hashes"])
    protected.update(json.loads((ROOT/"data/segmentation_unet_v3_local_path_repairs/repair_summary.json").read_text(encoding="utf-8"))["original_files_sha256"])
    assert len(final_rows)==len(base_rows)==len(rows)==300 and len(holds)==13 and len(repairs)==678
    added_total=removed_total=0
    changed_cases=[];pin_changed=[];verified_sources=0
    for index,row in enumerate(final_rows,1):
        old=base[row["case_id"]]
        assert row["split"]==old["split"] and row["image"]==old["image"]
        wire=np.asarray(Image.open(row["wire_mask"]))
        pin=np.asarray(Image.open(row["pin_heatmap"]))
        assert wire.shape==pin.shape==(row["height"],row["width"])
        assert set(np.unique(wire).tolist()) <= {0,255}
        assert int(np.count_nonzero(wire))==row["stats"]["wire_positive_pixels"]
        old_wire=np.asarray(Image.open(old["wire_mask"]))
        old_pin=np.asarray(Image.open(old["pin_heatmap"]))
        added=int(((wire>0)&(old_wire==0)).sum())
        removed=int(((wire==0)&(old_wire>0)).sum())
        added_total+=added;removed_total+=removed
        if added or removed:
            changed_cases.append(row["case_id"])
        if not np.array_equal(pin,old_pin):
            pin_changed.append(row["case_id"])
        if row.get("source_annotation_path"):
            source=Path(row["source_annotation_path"])
            assert sha(source)==row["source_annotation_sha256"]
            annotation=json.loads(source.read_text(encoding="utf-8"))
            regen_wire,regen_pin,tips,stats=annotation_targets(annotation,row["width"],row["height"],3,2)
            assert np.array_equal(np.asarray(regen_wire),wire)
            assert np.array_equal(np.asarray(regen_pin),pin)
            assert stats==row["stats"]
            saved_tips=json.loads((source.parent/"pin_points.json").read_text(encoding="utf-8"))
            assert tips==saved_tips
            assert json.loads((source.parent/"metadata.json").read_text(encoding="utf-8"))==row
            verified_sources+=1
        if old["structurally_usable_draft"]:
            assert row==old and sha(row["wire_mask"])==sha(old["wire_mask"]) and sha(row["pin_heatmap"])==sha(old["pin_heatmap"])
        if index%60==0:
            print(json.dumps(dict(phase="independent_final_verification",pages=index,total=300)),flush=True)
    assert removed_total==0 and pin_changed==["0102","0180"]
    assert len(changed_cases)==verified_sources==120
    assert added_total==summary["added_pixels"]+96
    assert sum(r["stats"]["missing_physical_paths"] for r in final_rows)==13
    assert sum(r["stats"]["missing_physical_paths"] for r in base_rows)==692
    held_cases={h["case_id"] for h in holds}
    assert all(not rows[case]["structurally_usable_draft"] for case in held_cases)
    training=read_rows(OUTPUT/"training_manifest.jsonl")
    frozen=read_rows(OUTPUT/"frozen_validation_manifest.jsonl")
    original_validation=[r for r in base_rows if r["split"]=="val" and r["structurally_usable_draft"]]
    assert frozen==original_validation and [r for r in training if r["split"]=="val"]==original_validation
    assert len([r for r in training if r["split"]=="train"])==166
    assert all(r["structurally_usable_draft"] and r["case_id"] not in held_cases for r in training)
    assert all(sha(path)==digest for path,digest in protected.items())
    checks=dict(status="all_300_final_rows_verified",source_annotations_regenerated=verified_sources,
                binary_masks_and_shapes_verified=300,original_source_and_selected_labels_unchanged=True,
                original_validation_rows_unchanged=28,added_pixels_from_v1=added_total,removed_pixels=0,
                changed_wire_cases=changed_cases,pin_changed_cases=pin_changed,pin_points_changed=3,
                unresolved_records_excluded_from_training=True,training_cases=166,training_manifest_rows=194,
                all_existing_structurally_usable_132_rows_unchanged=True,
                training_manifest_sha256=sha(OUTPUT/"training_manifest.jsonl"),
                final_manifest_sha256=sha(OUTPUT/"manifest.jsonl"),
                label_outputs_sha256={r["case_id"]:{key:sha(r[key]) for key in ("wire_mask","pin_heatmap")} for r in final_rows},
                training_started=False,full_page_certified=False)
    write_json(REPORT/"final_checks.json",checks)
    summary.update(added_pixels_from_v1=added_total,wire_changed_cases_from_v1=120,
                   final_checks=str((REPORT/"final_checks.json").resolve()))
    write_json(OUTPUT/"summary.json",summary);write_json(REPORT/"summary.json",summary)

    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    samples=[("0005",[145,94,248,143]),("0001",[258,638,365,674]),("0102",[325,625,380,675]),("0180",[518,438,563,476])]
    fig,axes=plt.subplots(len(samples),3,figsize=(12,12),dpi=145,layout="constrained")
    for axs,(case,roi) in zip(axes,samples):
        x0,y0,x1,y1=roi;extent=[x0-.5,x1-.5,y1-.5,y0-.5]
        rgb=np.asarray(Image.open(rows[case]["image"]).convert("RGB"))
        old=np.asarray(Image.open(base[case]["wire_mask"]))
        new=np.asarray(Image.open(rows[case]["wire_mask"]))
        axs[0].imshow(rgb[y0:y1,x0:x1],extent=extent,interpolation="nearest")
        axs[0].set_title(f"{case} 原图")
        axs[1].imshow(old[y0:y1,x0:x1],extent=extent,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
        axs[1].set_title("原 v1 导线标签")
        axs[2].imshow(new[y0:y1,x0:x1],extent=extent,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
        axs[2].set_title("修订后的导线标签")
        for ax in axs:
            ax.set_xlabel("原图 x");ax.set_ylabel("原图 y")
    fig.suptitle("缺路径修复实例 · 白色为导线监督 · 0005 / USB D− / EN / GND")
    fig.savefig(REPORT/"before_after.png",facecolor="white");plt.close(fig)

    lines=["# U-Net 缺路径修复 v4 · 2026-10-15","",
           "原始缺路径计数692条（122页）。0005已修复1条，本轮连续坐标恢复676条，原图逐条核查后另修复2条短路径；累计修复679条，剩余13条位于9页。所有结果保存在独立数据版本。","",
           "| 项目 | 数值 |","|---|---:|","| 连续源坐标恢复 | 676条 |","| 原图复核修复稀疏路线 | 2条 |","| 先前0005修复 | 1条 |","| 剩余异常 | 13条 / 9页 |",f"| 相对v1新增导线像素 | {added_total} |","| 删除导线像素 | 0 |","| 新增训练侧结构候选 | 63页（103→166） |","| 原验证清单与标签 | 28页保持原样 |","| 手工修正关联tip | 3个 |","",
           "676条仅恢复已有连续像素坐标，独立栅格检查保证中心线像素集合完全一致；保持固定3px草稿描边。所有坐标原图墨迹验证通过，未依据模型预测补标。","",
           "0102/E_222根据EN到+5V符号的可见短线重建L形路径，并修正P_30的横向偏移；0180/E_30恢复MCU GND到接地符号的竖线，修正P_95和P_67端点以及该接地符号的局部realBBox。它们是图像复核后的局部修订，未认证整页。","",
           "## 剩余异常及隔离","","| 页面 | 记录 | 原因 |","|---|---|---|"]
    notes={
        ("0101","E_157"):"两个点分别在3V3文字附近和C3另一端，引脚/电源归属冲突；不能跨电容直连。",
        ("0102","E_223"):"电容端点错位，另一点在GPIO引线，缺少对应可见物理路线。",
        ("0102","E_224"):"电容端点与QSPI_SS被关联，若直连会跨芯片主体；需核对原器件及网络语义。",
        ("0124","E_103"):"点落在C8旁空白及未连接PB9附近，没有对应可见路线。",
        ("0150","E_206"):"USB引脚与下方空白点被关联，未见物理导线。",
        ("0150","E_207"):"USB CC1引线与接地符号被跨区域关联，未见物理导线。",
        ("0180","E_31"):"RD引脚被关联到芯片下方空白，未见该路线。",
        ("0180","E_32"):"WR引脚被关联到接地符号附近，未见该路线。",
        ("0180","E_33"):"VCC符号与芯片另一侧P1.0关联，无法由两个点建立物理路线。",
        ("0238","E_43"):"C3接地符号与DREG引脚被跨区域关联，原图中无直接路线。",
        ("0024","E_41"):"连续轨迹沿Key开关主体边框，不能直接当导线恢复。",
        ("0203","E_92"):"轨迹沿开发板图片/多行引脚标识，导线语义不可靠。",
        ("0205","E_86"):"轨迹沿开发板图片/多行引脚标识，导线语义不可靠。",
    }
    for hold in holds:
        note=notes[(hold["case_id"],hold["edge_id"])]
        lines.append(f"| {hold['case_id']} | {hold['edge_id']} | {note} |")
        hold["review_reason_zh"]=note
    write_json(REPORT/"holds.json",holds)
    lines += ["","剩余13条没有虚构路径或删除源网络关系；相关9页保持不参与联合训练，并在holds.json登记。它们需要核对源对象归属/物理与逻辑网络规则。","",
              "## 使用与验证","",
              "训练候选166页来自原训练侧；原28页验证行保持原样。training_manifest.jsonl共194行，可由现有train_unet.py显式读取并使用--allow-draft。所有页仍为草稿监督；结构通过并不等于像素完整认证。原数据、默认模型和默认训练入口均未改动，未开始训练。","",
              "复现顺序：repair_unet_missing_paths_v4.py → apply_unet_sparse_manual_v4.py → finalize_unet_path_recovery_v4.py。恢复模块的5项安全测试及原U-Net单元检查用于验证拒绝端点硬连、保留转折和边界处理。","",
              "全量独立验证：300行尺寸/二值mask、120份修订源重新生成wire/pin与保存结果完全一致、原132个通过结构检查的行完全一致、原28验证行完全一致、原文件hash通过；仅0102和0180的三个tip发生已登记变化。完整数据：summary.json、repairs.json、holds.json、pages.json、final_checks.json。"]
    (REPORT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in checks.items() if k not in {"label_outputs_sha256","changed_wire_cases"}},ensure_ascii=False))


if __name__=="__main__":
    main()
