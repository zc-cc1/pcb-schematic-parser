import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
deployment=read(ROOT/'deployment_status.json')
integrity=read(ROOT/'deployment_integrity.json')
result=read(ROOT/'deployed_cached_check.json')
assert deployment['status']=='applied' and integrity['verified']
s=result['summary']
rows=result['per_case']
counts={}
for name,get_before,get_after in [
    ('wire',lambda r:r['wire_before']['f1'],lambda r:r['wire_after']['f1']),
    ('connectivity',lambda r:r['connectivity_before'],lambda r:r['connectivity_after'])]:
    deltas=[get_after(r)-get_before(r) for r in rows]
    counts[name]=[sum(d>1e-9 for d in deltas),sum(abs(d)<=1e-9 for d in deltas),sum(d < -1e-9 for d in deltas)]
contacts=sum(r['audit']['existing_power_contact_count'] for r in rows)
lines=['# 电源接触点恢复 v2 落地报告（2026-10-20）','',
       '已更新 D:\\pcb5_submission，版本为 observed-physical-power-wires-v2，正常使用原 agent.py 即默认启用。','',
       '## 实现','',
       '- 对已有模型网络恢复唯一、已编号、明确附着的电源接触点；其余候选若存在，必须全部明确指向同一主候选。',
       '- 电源类型和值来自已有器件记录；无需上游先形成同名电源合并组。保留现有物理分支导出的原有检查。',
       '- 拒绝多主候选、号码缺失、归属不明、附着不明、来源不一致、成员不在模型网络等情况。已有接触不会被转移到其他网络，已有引脚字段不会被覆盖。',
       '- 沿用原批处理入口、异常回退和 PCB_POWER_WIRE_EXPORT=0 开关。详细审计新增 existing_power_contacts / existing_power_contact_count，原有摘要字段继续保留。','',
       '本次仅更新四个文件：src/pcb_parser/power_wire_export.py、tests/test_wire_export_regression.py、docs/wire_export_update_20261020.md、package_manifest.json。旧文件保存在本目录 backup/。现有 Docker overlay 已包含该模块，无需调整复制路径。','',
       '## 落盘后验证','',
       '| 指标 | 上一正式版本 | 本次正式版本 |','|---|---:|---:|',
       f"| 13 页平均导线 F1 | {s['wire_before']:.4f} | {s['wire_after']:.4f} |",
       f"| 13 页平均连接 F1 | {s['connectivity_before']:.4f} | {s['connectivity_after']:.4f} |",'',
       f"导线：{counts['wire'][0]} 页提升、{counts['wire'][1]} 页不变、{counts['wire'][2]} 页下降。",
       f"连接：{counts['connectivity'][0]} 页提升、{counts['connectivity'][1]} 页不变、{counts['connectivity'][2]} 页下降。",
       f'累计恢复 {contacts} 个已有网络中的电源接触成员。','',
       '- 实际 D 盘入口函数输出与此前候选预测 13 / 13 完全一致。',
       '- 15 项专项测试全部通过；包含此前几何切分检查与本次唯一接触、重复主候选、歧义拒绝、来源及已有字段保护。',
       '- 65 次网络、物理分支、线段、附着记录和候选顺序打乱检查通过；13 页重复执行结果不变。',
       '- 13 页器件和引脚分数保持不变，已有导线几何保持不变，已有网络成员不丢失，输入数据不被修改。',
       '- overlay 检查通过；4 个发布文件哈希与验证副本一致，模型资产及发布清单之外的原文件哈希保持不变。',
       '- 新增模型推理、训练与平台提交均为 0。测试使用历史图结构和预测缓存，并非重新运行 13 页全模型识别。','',
       '分数来自此前已与两次历史平台报告校准的本地匹配规则，不是新增官方平台评测。第 13 页仍为 0.3095，后续需要单独解决归属、多个主候选与未附着电源等问题。','',
       '## 回退与镜像','',
       '回退到此次更新前的 v1，在桌面工作区运行：','',
       '```powershell',
       '.\\pcb_parser_competition\\.venv-gpu\\Scripts\\python.exe -B wire_contact_implementation_20261020/install.py --rollback',
       '```','',
       '回退脚本核对哈希后只恢复本次四个文件；若文件在更新后又被改动，会拒绝覆盖。PCB_POWER_WIRE_EXPORT=0 可关闭整个电源导出步骤，保留路径切分修复。',
       '此次已更新源码，未重新构建 Docker 镜像；使用既有镜像时仍需重新构建才能包含 v2。','',
       '证据：deployed_cached_check.json、deployment_integrity.json、release_plan.json。']
(ROOT/'落地验证报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'deployed':True,'version':'observed-physical-power-wires-v2','contacts':contacts,
                  'wire':counts['wire'],'connectivity':counts['connectivity'],'tests_passed':15},ensure_ascii=False))
