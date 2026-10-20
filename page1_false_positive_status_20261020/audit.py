"""Diagnose the six IC-scope FP records; never alter production predictions."""
import copy
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(WORK/'pcb_parser_competition/scripts'))
import platform_metrics as metrics
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
source=WORK/'pin_numeric_connector_implementation_20261020/deployed/1_Pred.json'
pred=read(source)
truth=read(next((WORK/'赛题六公开数据集/赛题六公开数据集/open_GT/open_GT/1').glob('*.json')))
pin=metrics.score_pins(pred,truth,20)[0]
assert pin['tp']==40 and pin['fp']==6 and pin['fn']==0
rows=[]
for owner,comp in pred['components'].items():
    if not metrics.is_ic(comp['type']):continue
    for key,p in pred['pins'].get(owner,{}).items():
        ref=truth['pins'].get(owner,{}).get(key)
        is_scored=metrics.is_ic(truth['components'].get(owner,{}).get('type'))
        point=[p['point'][axis] for axis in ['x','y']]
        dist=lambda r:math.dist(point,[r['point'][axis] for axis in ['x','y']])
        if is_scored and ref and p['pinname']==ref['pinname'] and dist(ref)<=20:continue
        nearest=min(truth['pins'].get(owner,{}).items(),key=lambda kv:dist(kv[1]))
        rows.append({'owner':owner,'key':key,'name':p['pinname'],'point':point,
                     'predicted_type':comp['type'],'reference_type':truth['components'][owner]['type'],
                     'same_number_error_px':dist(ref) if ref else None,
                     'nearest_reference_key':nearest[0],'nearest_reference_error_px':dist(nearest[1]),
                     'network_references':[nid for nid,n in pred['nets'].items() if owner+'.'+key[4:] in n['hyperGraph'].strip('()').split(',')]})
assert len(rows)==6 and {r['owner'] for r in rows}=={'SW2','H1','H2'}
assert {(r['owner'],r['key']) for r in rows}=={('SW2','pin_2'),('SW2','pin_6'),('SW2','pin_13'),('SW2','pin_14'),('H1','pin_1'),('H2','pin_1')}
switch=[]
for key,ref in truth['pins']['SW2'].items():
    got=pred['pins']['SW2'].get(key)
    error=math.dist([got['point'][a] for a in ['x','y']],[ref['point'][a] for a in ['x','y']]) if got else None
    switch.append({'key':key,'exported':got is not None,'error_px':error,'correct_20px':bool(got and got['pinname']==ref['pinname'] and error<=20)})
assert sum(r['correct_20px'] for r in switch)==3
# Diagnostic ablation using reference classes only. This is not a deployable
# inference rule and the modified prediction is deliberately not saved.
diagnostic=copy.deepcopy(pred)
for owner in ['SW2','H1','H2']:diagnostic['components'][owner]['type']=truth['components'][owner]['type']
type_only_pin=metrics.score_pins(diagnostic,truth,20)[0]
assert type_only_pin['tp']==40 and type_only_pin['fp']==type_only_pin['fn']==0
inventory=next(r for r in read(WORK/'pin_detection_audit_20261020/audit.json')['per_case'] if r['case']=='1')
records=read(Path(inventory['model_cache'])/'p/export/completed_component_records.json')['records']
classification=[{k:r.get(k) for k in ['component_id','type','detection_confidence','bbox_top_left']} for r in records if r['component_id'] in ['SW2','H1','H2']]
report={'case':'1','prediction':str(source),'IC_pin_score':pin,'scored_fp_records':rows,
        'source_classifications':classification,'switch_all_pin_diagnostic':switch,
        'switch_summary':{'reference_pin_count':16,'exported_pin_count':4,'correct_20px':3,'wrong_records':1,'absent_keys':12,'not_correctly_matched':13},
        'classification_only_diagnostic_IC_pin_score':type_only_pin,
        'classification_diagnostic_uses_reference_types_and_is_not_an_inference_rule':True,
        'new_model_calls':0,'new_ocr_calls':0,'production_code_modified':False}
(ROOT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');image=Image.open('D:/pcb_open_gt_in/1.png').convert('RGB');height=image.height
fig=plt.figure(figsize=(12,7.5),dpi=160);grid=fig.add_gridspec(2,2,width_ratios=[1.35,1],hspace=.4,wspace=.25)
axes=[fig.add_subplot(grid[:,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,1])]
panels=[('SW2',(1130,795,1410,1045),'SW2：开关被识别为 IC；4 项计入 FP'),('H1',(1295,250,1410,330),'H1：BAT+ 单针端子被识别为 IC'),('H2',(1445,425,1560,510),'H2：BAT- 单针端子被识别为 IC')]
for ax,(owner,box,title) in zip(axes,panels):
    ax.imshow(image.crop(box),extent=(box[0],box[2],box[3],box[1]),interpolation='nearest')
    for p in truth['pins'][owner].values():
        ax.scatter([p['point']['x']],[height-p['point']['y']],s=38,facecolors='none',edgecolors='#6c7f89',linewidths=1.1)
    for key,p in pred['pins'][owner].items():
        wrong=owner=='SW2' and key=='pin_2'
        color='#d06b00' if wrong or owner!='SW2' else '#007eb4'
        ax.scatter([p['point']['x']],[height-p['point']['y']],s=95,facecolors='none',edgecolors=color,linewidths=2)
    ax.axis('off');ax.set_title(title,fontproperties=font,fontsize=12)
    if owner=='SW2':
        ax.text(.5,-.045,'橙圈：当前标为 2，实际是 11 号位置\n另外 6、13、14 号为真实且位置正确的接点\n灰圈：参考中的全部 16 个接点',transform=ax.transAxes,ha='center',va='top',fontproperties=font,fontsize=10,linespacing=1.6)
    else:
        error=next(r['same_number_error_px'] for r in rows if r['owner']==owner)
        ax.text(.5,-.04,f'1 号真实存在；橙圈端点向内缩约 {error:.2f} 像素',transform=ax.transAxes,ha='center',va='top',fontproperties=font,fontsize=10)
fig.suptitle('第 1 页 6 项计分误检：根因是元件分类，不能直接删掉真实接点',fontproperties=font,fontsize=15,y=.98)
fig.subplots_adjust(top=.90,bottom=.16)
fig.savefig(ROOT/'六项误检来源.png',bbox_inches='tight',facecolor='white');plt.close(fig)

table=[]
for r in sorted(rows,key=lambda r:(r['owner'],int(r['key'][4:]))):
    kind='开关' if r['owner']=='SW2' else '单针端子'
    if r['owner']=='SW2' and r['key']=='pin_2':
        reason=f"编号错配：当前点距真实 11 号仅 {r['nearest_reference_error_px']:.2f} 像素；距真正 2 号 {r['same_number_error_px']:.2f} 像素"
    elif r['owner']=='SW2':reason=f"真实接点，编号正确，误差 {r['same_number_error_px']:.2f} 像素"
    else:reason=f"真实的 1 号接点；端点向框内偏约 {r['same_number_error_px']:.2f} 像素"
    table.append(f"| {r['owner']}.{r['key'][4:]} | {kind} → 错标 box/IC | {reason} |")
text='''# 第 1 页六项 IC 引脚计分误检：排查结果

需要更正此前“6 个多余引脚”的表述：这 6 条记录主要是**元件类别错误导致被纳入 IC 引脚评分**，并非 6 个凭空出现的接点。SW2、H1、H2 从原始模型结果起就被识别为 ic，导出时映射为 box，此次之前的引脚修复没有改变它们。

| 计分 FP 记录 | 元件类型问题 | 原图与位置核对 |
|---|---|---|
'''+ '\n'.join(table)+'''

## 为什么恰好是六个

当前 IC 引脚评分在真值侧只统计 box/ic 元件，在预测侧则统计所有预测为 box/ic 的元件引脚。U1、U2、U3、USB1、USB2 的 40 个评分引脚已全部匹配；SW2 错当 IC 后额外进入 4 条记录，H1/H2 各多进入 1 条，合计 TP/FP/FN = 40/6/0。

仅用于归因的模拟：在副本中依据参考类型把 SW2 改为 switch、H1/H2 改为 pin，其他字段、引脚和网络完全不改，IC 引脚分即变为 40/0/0。该模拟使用参考类型，不能充当通用推理规则，也没有写入正式代码或保存为部署预测。评分器未改动。

## 真实问题还包括哪些

- SW2 是原图中可见内部切换触点的 16 引脚开关，当前仅导出 4 条；其中 6、13、14 号正确，2 号错误地放到上方 11 号接点。真正的 2 号在下方。按完整开关引脚检查，只有 3/16 正确，还有 12 个编号记录缺失，加上 1 个错误记录，共 13 个未正确匹配。
- H1 是 BAT+ 单针端子，H2 是 BAT- 单针端子，各自的 1 号接点真实存在。当前点都向元件框内缩了约 12 像素；它们在此前的 20 像素容差内，但端点仍值得校正。
- 因此，正确方向是验证元件类型纠正、SW2 引脚身份及完整性、H1/H2 端点位置；直接删除这六条记录会损失真实接点信息。不能仅靠修改类型后的 IC 分数宣称整个原理图的引脚完整。

此前“13 页漏检清零”仅适用于已使用的 IC 引脚评分范围，并没有验证开关、电阻、电容和单针端子等全部元件的引脚完整性。本次明确将这两种范围分开。

## 当前范围

此次只读取既有预测、原图、原模型缓存、参考标注和本地评分代码。正式代码和正式预测均未修改；模型及 OCR 调用为 0。具体诊断证据在 status.json；三个原图裁剪为 SW2.png、H1.png、H2.png。

'''+f"![六项误检来源]({(ROOT/'六项误检来源.png').as_posix()})\n"
(ROOT/'六项误检分析.md').write_text(text,encoding='utf-8')
print(json.dumps({'fp_records':rows,'switch':report['switch_summary'],'classification_only_pin_metric':type_only_pin,'new_inference_calls':0},ensure_ascii=True))
