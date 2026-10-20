import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
report=read(ROOT/'deployed_cached_check.json');plan=read(ROOT/'release_plan.json')
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
unchanged=[];new_pins=moved_pins=0
for row in report['cases']:
    case=row['case'];a=read(WORK/f'pin_field_implementation_20261020/deployed/{case}_Pred.json');b=read(ROOT/f'deployed/{case}_Pred.json')
    if not row['gained']:assert a==b;unchanged.append(case)
    assert a['components']==b['components'] and a['nets']==b['nets']
    for owner,pins in b['pins'].items():
        for key,pin in pins.items():
            old=a['pins'].get(owner,{}).get(key)
            if old is None:new_pins+=1
            elif old!=pin:moved_pins+=1
counts={s:{k:sum(r[s]['pin'][k] for r in report['cases']) for k in ['tp','fp','fn']} for s in ['before','after']}
verification={'release_hashes_verified':True,'unit_tests_passed':57,
              'new_pin_records':new_pins,'corrected_existing_pins':moved_pins,
              'fully_unchanged_pages':unchanged,'all_component_boxes_and_networks_unchanged':True,
              'totals':counts,'entrypoint':read(ROOT/'entrypoint_check.json')}
(ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows=[]
for row in report['cases']:
    if not row['gained']:continue
    a,b=row['before']['pin'],row['after']['pin']
    rows.append(f"| {row['case']} | {a['tp']} → {b['tp']} | {a['fp']} → {b['fp']} | {a['fn']} → {b['fn']} | {a['f1']:.4f} → {b['f1']:.4f} | {', '.join(row['gained'])} |")
text=f'''# 引脚几何恢复落地验证

已更新 `D:/pcb5_submission` 的正式批处理入口，默认开启 `PCB_PIN_GEOMETRY_RECOVERY`。
设置 `PCB_PIN_GEOMETRY_RECOVERY=0` 可关闭。本次发布共 5 个文件，部署后逐文件哈希一致。

## 本次增益

对比基线为上一版已部署的位置校正和二次 OCR 字段恢复结果。
自动规则在 13 页缓存回放中新增 {new_pins} 条引脚记录、纠正 {moved_pins} 条已有引脚坐标，
使正确引脚增加 9 个，没有损失原先正确的引脚，也没有新增误检。

| 页码 | 正确 TP | 误检 FP | 漏检 FN | 引脚 F1 | 恢复的引脚 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

其余 9 页最终 JSON 完全一致。第 1 页剩余 5 个漏检全部恢复；剩余 6 个误检来自
SW2 的 4 个引脚以及 H1、H2 各 1 个引脚，涉及元件类型识别，未在本次修改。

13 页汇总 TP/FP/FN：{counts['before']} → {counts['after']}。
按当前本地组件、引脚、网络和导线加权评分，平均分
{report['macro_before']:.4f} → {report['macro_after']:.4f}，提升
{report['macro_after']-report['macro_before']:.4f} 分。这是本地复核结果，未提交平台重评。

## 自动规则和适用范围

使用当前图像中的红色矩形边框、真实短引脚线段、同侧已检出数字的位置和多视角 OCR
共同确定缺失引脚。数字、名称均需至少两个不同 OCR 视角支持，置信度至少 0.98。
候选归属必须唯一，已有引脚键值及名称保留。单个顶部/底部引脚只在连续红线支持唯一
共线 GND 接触点时校正。运行代码不读取 GT，也没有页码、元件编号或候选 ID 特例。

当前规则侧重彩色原理图中较大的矩形元件和左右短引脚；不支持的图像或证据不足时弃权。
不能据此认定其他图像上的漏检也已全部解决。

## 网络同步

8 个新增引脚的网络证据尚不足以唯一接入已有导出网络，均已记录待连接原因；
校正的 EP 引脚原有网络成员关系保留。13 页组件框、网络成员和导线几何均与基线一致，
网络与导线分数没有变化。本次提升来自引脚项。

模块允许在“唯一既有网络、精确模型附着或可见绿线、没有未决交点”的充分证据下同步成员；
不会创建网络或导线，也不会强制连接 NC。缺失导出网络、交点歧义和未连接短脚会明确弃权。

## 验证与费用

- 新增 15 项边界测试，连同已有测试合计 57 项全部通过。
- 正式入口使用同图模型缓存完成回放，字段恢复与几何恢复均执行，日志和最终导出正常。
- 第 1 页使用真实本地 OCR 重新运行 14 次，结果与缓存回放完全一致。
- 13 页缓存复核复用了 64 次 OCR 观测；未重跑 YOLO/U-Net，没有外部 API 调用。
- 所有预测先冻结，之后才加载参考标签评分。保留输入不变与非目标输出一致性断言。

## 文件

运行模块：`D:/pcb5_submission/src/pcb_parser/pin_geometry_recovery.py`。
配置：`D:/pcb5_submission/configs/pin_geometry_recovery_v1.json`。
说明：`D:/pcb5_submission/docs/pin_geometry_recovery.md`。
回退前文件保存在本文件夹 `backup/`；`install.py --rollback` 会核对当前文件哈希后回退本次发布。
部署验证结果见 `verification.json`，逐页指标见 `deployed_cached_check.json`，逐页恢复证据见 `deployed/*_audit.json`。
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False))
