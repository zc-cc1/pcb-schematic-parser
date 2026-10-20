"""Verify release artifacts and describe page 11 OCR/library reconciliation."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

report = read(ROOT/'deployed_cached_check.json')
plan = read(ROOT/'release_plan.json')
assert len(report['cases']) == 13
assert report['gained_pins'] == 9 and report['lost_pins'] == 0 and not report['regressed_pages']
for item in plan['files']:
    assert hashlib.sha256((Path(plan['target'])/item['path']).read_bytes()).hexdigest() == item['sha256']
cfg = read(ROOT/'runtime/configs/pin_row_library_v1.json')
library_path = Path(plan['target'])/cfg['library_path']
library_sha = hashlib.sha256(library_path.read_bytes()).hexdigest()
assert library_sha == cfg['library_sha256']
library = read(library_path)
audit = read(ROOT/'deployed/11_audit.json')
fresh = read(ROOT/'deployed_fresh_ocr_audit.json')
assert audit['library_sha256'] == library_sha
for key in ['changes', 'component_changes', 'network_changes', 'network_withheld']:
    assert fresh[key] == audit[key]
assert fresh['ocr_calls'] == audit['ocr_calls'] == 22
entry = read(ROOT/'entrypoint_deployed_check.json')
assert entry['runtime'] == 'deployed' and len(entry['cases']) == 8
assert all(r['success'] for r in entry['cases'])

unchanged = []
for row in report['cases']:
    case = row['case']
    old = read(WORK/f'pin_vertical_repair_implementation_20261020/deployed/{case}_Pred.json')
    new = read(ROOT/f'deployed/{case}_Pred.json')
    assert new == read(ROOT/f'staged/{case}_Pred.json')
    if case != '11':
        assert old == new
        unchanged.append(case)
before = read(WORK/'pin_vertical_repair_implementation_20261020/deployed/11_Pred.json')
after = read(ROOT/'deployed/11_Pred.json')
assert before.keys() == after.keys()
for key in before.keys()-{'components', 'pins', 'nets'}:
    assert before[key] == after[key]
assert before['components'].keys() == after['components'].keys()
expected_models = {'U1': 'TP4333', 'USB1': 'TYPE-C 6P(073)', 'USB2': 'AF 90° WJDG'}
for owner, comp in before['components'].items():
    expected = dict(comp)
    if owner in expected_models:
        expected['Name'] = expected_models[owner]
    assert expected == after['components'][owner]
assert before['pins']['U1']['pin_9'] == after['pins']['U1']['pin_6']
assert 'pin_9' not in after['pins']['U1']
added = []
for owner, pins in before['pins'].items():
    for key, pin in pins.items():
        if (owner, key) != ('U1', 'pin_9'):
            assert after['pins'][owner][key] == pin
for owner, pins in after['pins'].items():
    for key, pin in pins.items():
        if key not in before['pins'].get(owner, {}) and (owner, key) != ('U1', 'pin_6'):
            added.append({'owner': owner, 'key': key, 'pin': pin})
assert {(r['owner'], r['key']) for r in added} == {
    ('USB1', 'pin_7'), ('USB1', 'pin_A5'), ('USB1', 'pin_B5'),
    *{('USB2', f'pin_{n}') for n in range(1, 6)}}
members = lambda net: sorted(x.strip() for x in net['hyperGraph'].strip('()').split(',') if x.strip())
net_changes = {r['network_id']: r for r in audit['network_changes']}
assert set(net_changes) == {'LOG_NET_CAND_00002', 'LOG_NET_CAND_00003'}
assert before['nets'].keys() == after['nets'].keys()
for nid, old in before['nets'].items():
    new = after['nets'][nid]
    assert old.keys() == new.keys()
    for key in old.keys()-{'hyperGraph'}:
        assert old[key] == new[key]
    if nid in net_changes:
        assert members(old) == net_changes[nid]['before']
        assert members(new) == net_changes[nid]['after']
    else:
        assert old == new
all_members = {m for net in after['nets'].values() for m in members(net)}
assert not all_members & {'U1.9', 'U1.6', 'USB1.A5', 'USB1.B5', 'USB2.2', 'USB2.3', 'USB2.5'}
assert {'member': 'U1.6', 'reason': 'source_model_route_not_exported'} in audit['network_withheld']

evidence = {'library_path': str(library_path), 'library_sha256': library_sha, 'models': {}}
for group in audit['groups']:
    model = group['model']
    raw = library[model]['pins']
    unique = sorted({(str(p['number']), p['name']) for p in raw})
    evidence['models'][model] = {
        'owner': group['owner'], 'raw_library_pins': raw,
        'unique_number_name_pairs': [{'number': n, 'name': name} for n, name in unique],
        'model_ocr_readings': group['model_readings'], 'observed_rows': group['rows']}
usb2_pairs = evidence['models']['AF 90° WJDG']['unique_number_name_pairs']
assert usb2_pairs == [{'number': str(n), 'name': name} for n, name in enumerate(['VCC','D-','D+','GND','SHLD'], 1)]
write(ROOT/'library_evidence.json', evidence)
counts = {s: {k: sum(r[s]['pin'][k] for r in report['cases']) for k in ['tp','fp','fn']} for s in ['before','after']}
assert counts == {'before': {'tp':181,'fp':7,'fn':10}, 'after': {'tp':190,'fp':6,'fn':1}}
verification = {
    'release_hashes_verified': True, 'library_hash_verified': True,
    'unit_tests_passed': 218, 'new_tests': 33,
    'fully_unchanged_pages': unchanged, 'fresh_local_ocr_matches_cache': True,
    'page11_local_ocr_calls': 22, 'all_pages_cached_ocr_views': report['reused_views'],
    'external_api_calls': 0, 'full_model_runs': 0,
    'existing_pin_records_preserved_except_U1_number_rekey': True,
    'new_pin_records': added, 'component_changes': audit['component_changes'],
    'existing_wire_segments_preserved': True, 'network_changes': audit['network_changes'],
    'network_withheld': audit['network_withheld'], 'aggregate_pin_metrics': counts,
    'predictions_frozen_before_reference_reads': report['predictions_frozen_before_reference_reads'],
    'entrypoint': entry}
write(ROOT/'verification.json', verification)

font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
image = Image.open('D:/pcb_open_gt_in/11.png').convert('RGB')
fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), dpi=160)
panels = [
    ('U1', (900,330,1145,495), 'U1：VDD 编号 9 → 6', 'pin_6'),
    ('USB1', (70,240,330,455), 'USB1：补齐 7、A5、B5', None),
    ('USB2', (550,320,765,490), 'USB2：按精确型号补齐 5 脚', None)]
for ax, (owner, box, title, only) in zip(axes, panels):
    ax.imshow(image.crop(box), extent=(box[0],box[2],box[3],box[1]), interpolation='nearest')
    changes = [r for r in audit['changes'] if r['owner'] == owner]
    for r in changes:
        x = r['after']['point']['x']; y = image.height-r['after']['point']['y']
        color = '#874dc3' if r['method'] == 'unique_remaining_library_contact' else '#1676bd'
        ax.scatter([x],[y],s=110,facecolors='none',edgecolors=color,linewidths=2)
    ax.set_title(title, fontproperties=font, fontsize=13)
    ax.axis('off')
    if owner == 'U1':
        desc = '原图数字 6 + VDD，局部 OCR 确认\n保留原坐标，修正错误编号\n型号恢复为 TP4333'
    elif owner == 'USB1':
        desc = '7 / EH、A5 / CC1、B5 / CC2\n均由原图局部 OCR 确认\nCC1、CC2 的不连接叉号保留'
    else:
        desc = '库型号：AF 90° WJDG\n1 VCC · 2 D- · 3 D+ · 4 GND\n紫圈：唯一剩余库引脚 5 / SHLD'
    ax.text(.5,-.07,desc,transform=ax.transAxes,ha='center',va='top',fontproperties=font,fontsize=11,linespacing=1.7)
fig.suptitle('第 11 页修复验证：蓝圈为原图文字支持，紫圈为元件库唯一剩余接点',fontproperties=font,fontsize=15)
fig.subplots_adjust(top=.84,bottom=.26,wspace=.18)
fig.savefig(ROOT/'第11页修复结果.png', facecolor='white', bbox_inches='tight')
plt.close(fig)

page = next(r for r in report['cases'] if r['case']=='11')
text = f'''# 第 11 页 OCR 与元件库补全：落地验证报告

已部署到 `D:\\pcb5_submission`。前四项通过原图局部 OCR 修复；USB2 缺少印刷编号，使用项目自带元件库中的精确型号补齐五个引脚。

| 本地指标 | 修复前 | 修复后 |
|---|---:|---:|
| 正确引脚 TP | 11 | 20 |
| 误检 FP | 1 | 0 |
| 漏检 FN | 9 | 0 |
| 引脚 F1 | 68.75% | 100% |
| 连接关系指标 | {page['before']['connectivity']:.6%} | {page['after']['connectivity']:.6%} |
| 综合评分（100 分制） | {page['before']['score_100']:.5f} | {page['after']['score_100']:.5f} |

这是本地复现评测，非官方提交分数；引脚匹配沿用 20 像素容差。导线仍为 TP/FP/FN = 18/11/6，原有线段完全不变。100% 仅指引脚 F1。

## 前四项：原图 OCR 验证

- U1 的 VDD 原来误编为 9，原图紧裁剪两次读到 6（0.99958、0.99956），修正为 6/VDD，位置和名称原样保留。型号 TP4333 的库表也支持 6/VDD。
- USB1 补齐 7/EH、A5/CC1、B5/CC2。7 的紧裁剪识别置信度为 0.99995、0.99996；CC1/A5、CC2/B5 均有多视图高置信文字证据。
- 恢复三处原本仅有位号的型号字段：U1 → TP4333、USB1 → TYPE-C 6P(073)、USB2 → AF 90° WJDG。其他元件字段不变。

## 后五项：精确型号与库表

库来源：`{library_path.as_posix()}`；SHA-256 为 `{library_sha}`。未修改库文件，运行时检查该哈希。

USB2 型号 AF 90° WJDG 在四个局部视图中一致，置信度分别为 0.98102、0.98401、0.99679、0.99170。原图可读四个名称，但没有编号；型号和名称同时匹配后，从库取出对应编号。

| 引脚 | 名称 | 补全依据 |
|---|---|---|
| 1 | VCC | 原图名称，两次局部确认 0.98198、0.98026；库表唯一映射 |
| 2 | D- | 原图名称，两视图均超过 0.998；库表唯一映射 |
| 3 | D+ | 原图名称，两视图均超过 0.990；库表唯一映射 |
| 4 | GND | 原图名称，四视图均超过 0.999；库表唯一映射 |
| 5 | SHLD | 原图存在一个未印字的左侧物理接点；另外四脚已唯一对应，库中仅剩此项 |

SHLD 是库补全结果，没有宣称 OCR 读出了原图不存在的文字。引脚位置均来自实际红色引脚和原模型候选，不采用库内布局或参考标注坐标。

库中的重复条目会合并，但冲突名称全部保留。例如 TYPE-C 6P(073) 的 7 号存在 EH/GND 别名冲突，本次采用原图明确读出的 EH；规则不会任取第一条库记录。存在型号冲突、多个未确认接点、引脚数量不符或高置信文字矛盾时，整组不提交。

## 网络与未解决边界

- 原网络 LOG_NET_CAND_00002 补入 USB1.7、USB2.4；LOG_NET_CAND_00003 补入 USB2.1。两处均有原图绿色接触和原模型既有连接支持。
- USB1 的 A5/CC1、B5/CC2，以及 USB2 的 2/D-、3/D+ 存在不连接叉号，保持不接网。USB2 的 5/SHLD 为悬空接点，也保持不接网。
- U1.6 的原模型支路此前未被导出，本次保留这一限制；引脚编号修正成功不代表该支路已修复。未猜测其连接关系。
- 没有新增元件、网络或导线。原有网络编号、全部线段及无关成员不变。

## 回归与成本

- 新增 33 项边界测试，部署后共 218 项测试全部通过。
- 部署后第 11 页进行了 22 次真实本地局部 OCR，最终预测与缓存验证完全相同。
- 13 页使用同图模型及 OCR 缓存回归，其余 12 页完整 JSON 均未变化；先冻结预测，再加载参考标注评分。
- 第 1、2、4、5、6、8、10、11 页实际导出入口验证通过。全部既有页面修复保留。
- 13 页引脚 TP/FP/FN 从 181/7/10 变为 190/6/1；综合分逐页均值从 {report['macro_before']:.5f} 变为 {report['macro_after']:.5f}。
- 新规则完整运行 13 页需要 42 次局部 OCR：第 11 页 22 次，第 1/4/13 页分别尝试 12/4/4 次但不改结果；回归复用缓存。外部 API 调用及完整模型重跑均为 0。

部署文件为 `agent.py`、`src/pcb_parser/pin_row_library.py`、配置、测试及说明，共五个文件，部署哈希已逐一核验。默认启用，设置 `PCB_PIN_ROW_LIBRARY=0` 可关闭。备份及安装清单保留于本目录；重复执行不再修改已修正元件。

证据：`verification.json`、`library_evidence.json`、`deployed_cached_check.json`、`deployed_fresh_ocr_audit.json`、`entrypoint_deployed_check.json`。13 页最新预测位于 `deployed/`，实际入口导出位于 `cached_entrypoint/`。

![第 11 页修复结果]({(ROOT/'第11页修复结果.png').as_posix()})
'''
(ROOT/'落地验证报告.md').write_text(text, encoding='utf-8')
print(json.dumps({'report': str(ROOT/'落地验证报告.md'), 'release_verified': True,
                  'unchanged_pages': len(unchanged), 'counts': counts}, ensure_ascii=False))
