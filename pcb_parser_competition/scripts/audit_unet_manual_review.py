"""Record agent visual inspection without modifying or approving labels."""
from collections import Counter
import json
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
from unet_audit_common import ROOT, OUT, read, write, image, truth, probability, counts, metrics, aggregate, assert_protected

# Findings are scoped to the supplied ROI pictures, not full-page human certification.
NOTES = {
 "0208": ("confirmed_local_label_geometry", "两处电源到连接器的GT走出原图没有的倒U形；GT斜短线与原图水平引线不符。模型全页输出空，既有真实漏检，也有错误监督。", [498,539,618,644]),
 "0170": ("model_frame_FP", "图框/分区表格边被整段预测为导线；logo区GT为空。R18/C4与R7/C3横段有小幅红蓝并行，不能把全部FP都认作标注错误。", [306,136,683,293]),
 "0286": ("model_missed_straight_wires", "LM4674A两条长输入横线几乎全漏，存在真实线形证据；输出侧引脚/导线分界截短会引起部分FP，需语义规范确认。", [0,247,267,347]),
 "0007": ("wire_pin_semantics_unresolved", "IC多根引脚与蓝色导線连续但GT止点/缺段不统一；右侧长竖线没有GT。可能是元件引脚定义或漏标，不能单凭外观判错。", [379,65,534,165]),
 "0012": ("local_label_width_alignment_suspect", "彩色高亮电源线预测宽于3pxGT；GT存在台阶和密集引脚段偏离，TP仍连续。需核对高亮截图与标签坐标规范。", [578,6,978,106]),
 "0213": ("confirmed_local_label_geometry", "VCC旁横向GT在原图空白处，电容间隙却被标成短竖导线；右侧反馈长竖线真实存在而漏检。", [315,3,435,103]),
 "0033": ("wire_pin_semantics_unresolved", "上/下外环被预测但GT缺失，可能是器件内部图形或真实导线；EN竖段及WDO横段部分漏检。不能直接补GT。", [88,30,208,132]),
 "0029": ("model_component_border_FP", "TDM Master方框上/左边与大IC上边被预测；两个电容支路长竖段漏检。", [0,347,176,476]),
 "0267": ("model_component_border_FP", "模拟信号箭头线大多识别，部分方框角被误检；线条上下红蓝并行属于小位置/宽度差异。", [229,116,349,216]),
 "0148": ("confirmed_local_label_geometry", "D5到R9的横向GT与原图绿线明显分离；R13/R32上下横线也偏移，不能用单一全图平移修复。", [468,320,588,420]),
 "0105": ("model_text_FP", "绿色中文提示文字被预测为导线；电容支路末端有标签与像素止点差异。", [445,490,565,590]),
 "0036": ("model_missed_straight_wires", "X1两侧真实细线漏检，左侧无连接引脚叉附近存在短GT延伸；L2接地竖线局部位置误差，中文标题少量FP。", [589,30,715,130]),
 "0002": ("model_component_border_FP", "BCM2835/IC2矩形边识别为导线；真正横线有细条FN，属于宽度或1px偏移差异。", [415,52,695,152]),
 "0042": ("model_missed_straight_wires", "多根GND短横段、竖母线存在缺口；电阻引线的wire/pin分界不明确。交叉处识别强，不能据栅格交叉判电气连接。", [416,125,536,239]),
 "0236": ("model_component_border_FP", "TMP112-Q1及左侧器件矩形上边FP；带junction dot的真实横线大体存在，红蓝并行为形状差异。", [0,74,204,174]),
 "0018": ("wire_pin_semantics_unresolved", "PB3/SWDIO长横线可见且被识别，GT却只留短点；可能未连接引脚/标注过滤，语义未认证；邻线有宽度偏差。", [574,147,720,247]),
 "0049": ("model_component_border_FP", "VIN/PVIN与PWM2/AGND IC外框上/下边FP，真实连线整体连通但局部边界有红蓝差异。", [533,58,757,158]),
 "0227": ("wire_pin_semantics_unresolved", "空心端子符号的长杆没有GT但模型识别，需确认杆属于pin还是wire；另一横线有细边FN。", [203,133,323,233]),
 "0149": ("confirmed_local_label_geometry", "C15/C16上方竖段GT明显移到原图绿线右侧；模型沿原图线，造成成对FP/FN。", [745,106,865,206]),
 "0287": ("local_label_width_alignment_suspect", "粗黑电阻引线/接点被3pxGT表达，短线止点及拐角出现少量FP/FN，未见大段真实导线缺失。", [290,0,410,100]),
 "0129": ("confirmed_local_label_geometry", "BUZZER供电支路GT经原图空白和文字，原图下方环线缺标；不能把大片红色全归因模型。", [785,207,905,307]),
 "0073": ("mixed_missing_label_and_model_FN", "CC2实际横线无GT（与既有训练修正记录一致），CC1/VOUT真实细横线漏检；IC边误检另存。", [409,49,529,149]),
 "0265": ("confirmed_local_label_geometry", "0.1uF上方多出的GT横支路不在原图；粗线连接点附近真实漏检和wire/pin止点差异同时存在。", [320,27,449,127]),
 "0281": ("model_component_border_FP", "大矩形上/下边误检；真实连线覆盖较完整，junction/箭头只缺少量像素。", [0,3,296,103]),
 "0088": ("confirmed_local_label_geometry", "C1/C2左右横支路GT系统高于原图线约11–12px，与既有训练修正证据一致。模型多数沿原线。", [374,0,518,100]),
 "0215": ("confirmed_local_label_geometry", "C3/L2/R3/C5区域GT多出穿越元件的横线、下方真实母线缺标；另有小局部偏移。", [197,153,527,309]),
 "0091": ("model_missed_straight_wires", "+5V/+3V3电源短竖线与文字相邻，基本未检；短引脚与GT止点存在少量差异。", [454,192,574,292]),
 "0079": ("wire_pin_semantics_unresolved", "SENSE_EN/SEL长箭头线无GT，外环长段也缺，需区分符号与导线；接地GT局部覆盖了方框底边。", [50,163,184,263]),
 "0253": ("wire_pin_semantics_unresolved", "长外环/矩形边无GT但预测，整页语义需确认；GPIO/VBAT线大体检出，短端点漏检。", [598,127,760,243]),
 "0254": ("confirmed_local_label_geometry", "C1两条水平支路GT均高于原图真实线，红蓝平行分离明显；模型沿实际线。", [261,1,396,101]),
}


def main():
    assert_protected();manifest=read(OUT/"review30/manifest.json");records=[]
    for sample in manifest["samples"]:
        c=sample["case_id"];kind,note,box=NOTES[c];x0,y0,x1,y1=box
        rgb=image(c);gt=truth(c);p=probability(c);mask=p>=.5
        roi=np.zeros_like(gt);roi[y0:y1,x0:x1]=True
        sk=skeletonize(gt)&roi;gray=np.asarray(Image.fromarray(rgb).convert("L"));ink=gray<220
        distance=ndimage.distance_transform_edt(~ink) if ink.any() else np.full(gt.shape,np.inf)
        values=distance[sk]
        record=dict(case_id=c,split=sample["split"],classification=kind,observation=note,ROI=box,
            ROI_wire_metrics=metrics(*counts(mask,gt,roi)),ROI_GT_skeleton_points=int(sk.sum()),
            ROI_GT_skeleton_distance_to_any_ink_median=float(np.median(values)) if len(values) else None,
            ROI_GT_skeleton_distance_to_any_ink_gt2_fraction=float((values>2).mean()) if len(values) else None,
            source_picture=sample["roi_picture"],visual_inspection="agent inspected original/GT/error ROI panels; select5 also whole-page overview",
            independent_human_annotation=False,full_page_certified=False,labels_modified=False)
        records.append(record)
        sample["visual_review_status"]="agent_ROI_inspection_complete_not_human_certification"
    kinds=Counter(r["classification"] for r in records)
    baseline=read(OUT/"baseline_reproduction.json");val=[r for r in baseline["records"] if r["included_in_26"]]
    influence=[]
    for r in val:
        left=aggregate([x for x in val if x["case_id"]!=r["case_id"]])
        influence.append(dict(case_id=r["case_id"],fp_share=r["wire"]["fp"]/baseline["micro"]["fp"],
            error_share=(r["wire"]["fp"]+r["wire"]["fn"])/(baseline["micro"]["fp"]+baseline["micro"]["fn"]),
            leave_one_out_diagnostic_dice=left["dice"],not_removed_from_evaluation=True))
    result=dict(status="30_agent_ROI_reviews_complete",records=records,classifications=dict(kinds),
        confirmed_geometry_cases=[r["case_id"] for r in records if r["classification"]=="confirmed_local_label_geometry"],
        biased_sample=True,population_label_error_rate=None,inter_annotator_Dice=None,annotation_ceiling=None,
        rationale="Worst20 val plus worst10 train; visible local path/image disagreement does not estimate independent annotation consistency or population ceiling. Unmarked pin/symbol lines remain unresolved.",
        influence=influence,protected_files=assert_protected())
    write(OUT/"manual_review30.json",result);write(OUT/"review30/manifest.json",manifest)
    errors=read(OUT/"error_decomposition.json");errors["status"]="statistics_and_30_agent_ROI_reviews_complete";write(OUT/"error_decomposition.json",errors)
    text=["# 30张最差样本的局部核对", "", "这是代理对图像的逐张检查，不是独立人工重标、全页认证或总体噪声率估计。原始GT、验证集、生产模型均未修改。", "", "| 图号 | split | Dice@0.5 | 局部观察 | ROI | 图像 |", "|---|---|---:|---|---|---|"]
    for r in records:
        sample=next(s for s in manifest["samples"] if s["case_id"]==r["case_id"])
        text.append(f"| {r['case_id']} | {r['split']} | {sample['wire']['dice']:.2%} | {r['observation']} | {r['ROI']} | [局部对照]({PathFix(r['source_picture'])}) · [整页]({PathFix(sample['overview'])}) |")
    (OUT/"manual_review30.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps(dict(classifications=dict(kinds),confirmed_geometry=result["confirmed_geometry_cases"],influence_0170=next(r for r in influence if r["case_id"]=="0170")),ensure_ascii=False))


def PathFix(s):return s.replace("\\","/")


if __name__=="__main__":main()
