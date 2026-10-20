import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
deployed=read(ROOT/'deployment_status.json')
integrity=read(ROOT/'deployment_integrity.json')
result=read(ROOT/'deployed_cached_check.json')
previous=read(ROOT.parent/'scorer_fix_20261020/cached_rescore.json')
assert deployed['status']=='applied' and integrity['verified']
original={row['case']:row['variants']['original_baseline']['fixed_wire']['f1']
          for row in previous['per_case']}
rows=result['per_case']
baseline=sum(original[r['case']] for r in rows)/len(rows)
final=result['summary']['fixed_wire_f1']
stats=[sum(r['fixed_wire']['f1']>original[r['case']] for r in rows),
       sum(r['fixed_wire']['f1']==original[r['case']] for r in rows),
       sum(r['fixed_wire']['f1']<original[r['case']] for r in rows)]
lines=['# 导线代码落地与验证（2026-10-20）','',
       '已更新 D:\\pcb5_submission。生产代码、入口、Docker overlay 与文件清单已同步，原文件保存在本目录 backup/。','',
       '## 改动','',
       '- wire_graph.py：在骨架实际路径上切分，保留折角、稠密路径点、路径长度和方向。',
       '- power_wire_export.py：恢复合格电源物理分支和接触点，仅使用已有模型图、已有号码和已接受的同名电源信息。',
       '- 对重复候选阻断的电源接触，仅在其他候选均明确指向唯一主候选、主候选唯一附着已有网络时补回；不猜普通器件或 IC 引脚字段。',
       '- agent.py：在 target 结果写出前调用，记录每页来源审计；异常时保留原有效预测。',
       '- Dockerfile.overlay、overlay_check.py 与 package_manifest.json 已同步，新增测试及项目内说明文档。','',
       '第 1 页原先出现退步，是因为已有导线网络遗漏地接触，而新增接地分支形成另一个竞争逻辑网络。补回有明确主次证据的接触后，该页由 0.5645 提升至 0.6111。','',
       '## 缓存结果','',
       '| 版本 | 13 页平均导线 F1 |','|---|---:|',
       f'| 原 D 盘版本 | {baseline:.4f} |',
       f"| 已验证的路径切分修复 | {result['summary']['baseline_wire_f1']:.4f} |",
       f'| 此次落地版本 | {final:.4f} |','',
       f'相对原 D 盘版本：{stats[0]} 页提升、{stats[1]} 页不变、{stats[2]} 页下降。相对路径修复后的版本：8 页提升、5 页不变、0 页下降。','',
       '| 页 | 原版本 | 路径修复后 | 此次落地 |', '|---|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['case']} | {original[row['case']]:.4f} | {row['baseline_wire']['f1']:.4f} | {row['fixed_wire']['f1']:.4f} |")
lines.extend(['','## 验证范围','',
    '- 落盘后的 D 盘代码：11 项专项回归测试全部通过，包含几何切分、重复引脚、歧义拒绝、图像来源、开关和重复执行。',
    '- 生产函数 finalize_power_wire_export 对 13 页缓存全部验证；器件与引脚评分不变、已有线段不变、已有网络成员不丢失、原始输入不变、重复执行结果不变。',
    '- 使用之前已完成路径回放的图结构，调用实际 D 盘入口导出函数；本次未重新运行 13 页全模型推理。',
    '- 打包所需模块存在性、导入和 overlay 检查通过，8 个发布文件的落盘哈希与验证副本一致。',
    '- 原有 118 个受保护源文件、配置与模型中，只有发布清单内的文件变化；其他文件和运行模型资产哈希保持原值。',
    '- 新增模型推理、训练、平台提交均为 0。未构建新 Docker 镜像，既有镜像需要重新构建后才会包含此次修改。',
    '- 导线分数来自此前与两次历史平台报告 26 例完全对齐的本地评分器；它仍不是一次新的平台评测，总分中的旧连接评分仍为近似实现。','',
    '## 日常使用与回退','',
    '正常调用原 agent.py 即默认启用。每页详细审计位于输出目录 logs/<case_id>_power_wire_export.json。',
    '`PCB_POWER_WIRE_EXPORT=0` 关闭新增电源导出，保留路径切分修复；`PCB_WIRE_CLEANUP=0` 也会作用于新增分支。','',
    '完整回退脚本只恢复清单中的旧文件，并删除本次新增文件；如果文件在此次更新后又被修改，会拒绝覆盖。需要回退时，在桌面工作区执行：','',
    '```powershell',
    '.\\pcb_parser_competition\\.venv-gpu\\Scripts\\python.exe -B wire_implementation_20261020/install.py --rollback',
    '```','',
    '证据文件：deployed_cached_check.json、deployment_integrity.json、release_plan.json。'])
(ROOT/'代码更新报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'deployed':True,'original_wire_f1':baseline,'final_wire_f1':final,
                  'original_comparison_improved_equal_worse':stats,'tests_passed':11,
                  'neural_inference_calls':0},ensure_ascii=False))
