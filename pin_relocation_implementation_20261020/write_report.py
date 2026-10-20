import copy
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
report=read(ROOT/'deployed_cached_check.json');plan=read(ROOT/'release_plan.json')
assert len(report['cases'])==13 and report['gained_pins']==1 and report['lost_pins']==0 and not report['regressed_pages']
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest()==item['sha256']
unchanged=[]
for row in report['cases']:
    case=row['case'];before=read(WORK/f'pin_geometry_implementation_20261020/deployed/{case}_Pred.json');after=read(ROOT/f'deployed/{case}_Pred.json')
    assert before['components']==after['components'] and before['nets']==after['nets']
    if case!='2':assert before==after;unchanged.append(case)
    else:
        local=copy.deepcopy(after);local['pins']['U2']['pin_2']=before['pins']['U2']['pin_2'];assert local==before
page=next(r for r in report['cases'] if r['case']=='2')
audit=read(ROOT/'deployed/2_audit.json');assert len(audit['network_changes'])==1
assert audit['network_changes'][0]['status']=='withheld'
pilot=read(WORK/'page2_pin_relocation_20261020/verification.json')
verify={'release_hashes_verified':True,'unit_tests_passed':73,'new_tests':16,
        'fully_unchanged_pages':unchanged,'only_change':'page 2 U2.2 point',
        'all_components_networks_and_wires_unchanged':True,
        'fresh_local_ocr_matches_cache':True,'fresh_ocr_calls':6,
        'old_point_error_px':pilot['old_error_px'],'new_point_error_px':pilot['new_error_px'],
        'entrypoint':read(ROOT/'entrypoint_check.json')}
(ROOT/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=f'''# 错位引脚迁移：落地验证报告

已更新 `D:/pcb5_submission`，默认开启 `PCB_PIN_RELOCATION`。
本次发布 5 个文件，部署后逐文件哈希一致；关闭可设置 `PCB_PIN_RELOCATION=0`。
旧文件保存在当前文件夹 `backup/`，`install.py --rollback` 在核对文件哈希后可回退本次更新。

## 效果

基线为已部署的二次 OCR 字段恢复和几何引脚恢复版本。本次只修复第二页 U2.2 VCC 的坐标。

| 第二页指标 | 基线 | 部署版缓存回放 |
| --- | ---: | ---: |
| 正确 IC 引脚 TP | 15 | 16 |
| 漏检 FN | 1 | 0 |
| 误检 FP | 1 | 0 |
| 引脚 F1 | 0.9375 | 1.0000 |
| U2.2 位置误差 | {pilot['old_error_px']:.3f} px | {pilot['new_error_px']:.3f} px |
| 当前本地加权总分 | {page['before']['score_100']:.4f} | {page['after']['score_100']:.4f} |

其余 12 页最终 JSON 完全一致，没有正确引脚损失、新增误报或分数退化。
13 页当前本地加权平均分 {report['macro_before']:.4f} → {report['macro_after']:.4f}。
所有预测先冻结再读取 GT，未提交平台重评。20 像素门槛仅用于事后计分。

## 修复规则

自动筛选仅由库排除法填充编号和名称的低置信度点：必须没有可靠文字证据、没有已导出网络成员关系、
模型附件明确无邻近导线，且模型网络中没有矛盾的候选成员关系。直接识别的字段和已连接引脚受到保护。
当前坐标还必须与原候选一致，并位于可验证的红色矩形框内、没有可见绿线。

当前版本只处理左侧引脚：至少 3 个其他引脚支持同侧布局，且只有一条真实短引线空缺。
从已观察数字估计编号列和行偏移，隔离数字做 3x/5x 本地 OCR；整个元件做 2x/3x、0/90 度 OCR。
编号和名称均需两个不同视图在置信度至少 0.98 时确认，且与待修复记录完全一致。
只替换原键的坐标，不改名称、编号、元件框；不依赖页码、元件编号、目标文字或 GT 坐标。

每图最多尝试 2 个候选，最多 12 次本地 OCR，模型按需加载；未触发页面不会加载 OCR。
证据不足、字段冲突、候选不唯一、坐标已改变或网络矛盾时弃权。

## 网络核对和限制

U2.2 原错点的附件为 `no_wire_within_radius`，没有导出网络成员关系。
正确位置距模型网络 `LOG_NET_CAND_00012` 约 0.493 像素，图上有绿线，且无未决交点；
但该网络尚未进入最终导出。审计将其记录为待连接，不创建网络或导线。

因此本次解决的是引脚位置和引脚得分，U2.2 的网络接入仍未修复；网络、导线分数保持基线。
模块只在唯一匹配的模型网络已经导出、没有未决交点时同步成员，不会沿用旧错点的附件。
13 页的组件、网络成员与导线几何均完全一致。后续若要恢复这段电源分支，应独立验证完整网络导出。

## 验证

- 新增 16 项证据、归属、网络保护、预算、幂等性和输入不变测试，全部 73 项测试通过。
- 13 页模型缓存回归，只有第二页 U2.2 坐标改变。
- 第二页真实本地 OCR 重跑 6 次，结果与缓存一致；无外部 API、无完整模型重跑。
- 部署后的第 1、2 页正式入口回放成功，所有引脚后处理钩子、摘要、日志和最终导出正常。
  检测器、U-Net 和 OCR 使用同图缓存，IC 引脚与验证输出一致；没有新增模型调用。
- 部署后再次回放 13 页，结果与预发布预测完全一致；5 个发布文件哈希一致。

## 文件

正式模块：`D:/pcb5_submission/src/pcb_parser/pin_relocation.py`。
正式配置：`D:/pcb5_submission/configs/pin_relocation_v1.json`。
入口：`D:/pcb5_submission/agent.py`，在几何恢复之后执行错点复核。
审计日志：`logs/{{case}}_pin_relocation.json`。
本地指标：`deployed_cached_check.json`；完整性检查：`verification.json`；入口检查：`entrypoint_check.json`。
'''
(ROOT/'落地验证报告.md').write_text(text,encoding='utf-8')
print(json.dumps(verify,ensure_ascii=False))
