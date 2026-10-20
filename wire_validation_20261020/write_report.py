"""Generate a concise Chinese report from completed paired experiments."""
from validate import HERE, read


def main():
    data=read(HERE/'comparison.json')
    integrity=read(HERE/'deployment_integrity.json')
    protection=read(HERE/'current_protection.json')
    combined=read(HERE/'combined_protection.json')
    rows=data['per_case_comparison']
    baseline,candidate=data['baseline'],data['candidate_path']
    gain=candidate['score_100']-baseline['score_100']
    checks=data['independent_checks']
    text=['# 导线优化隔离验证报告','',
        '日期：2026-10-20。正式项目：`D:/pcb5_submission`。本次只修改独立测试副本，未修改正式源码、配置或权重。','',
        '## 验证结论','',
        f'路径修正小幅有效：本地导线 F1 {baseline["macro"]["wire_f1"]:.4f} → {candidate["macro"]["wire_f1"]:.4f}；总分 {baseline["score_100"]:.2f} → {candidate["score_100"]:.2f}（{gain:+.2f}）。元件、引脚和连通三项不变。',
        '13 页中 4 页导线提升、3 页略降、6 页不变；所有流程 13/13 成功。独立匹配复核同样改善，支持后续只落地路径修正。切点保护单独或组合使用均未取得增益，暂不采纳。','',
        '## 测试方式','',
        '- 13 页 open_GT；先用当前部署代码的原样副本完成 PNG 全流程基线。',
        '- 候选与原算法重放对照均复用同一批检测、OCR、引脚和 U-Net 概率缓存；冻结无关的 LED 模型框。',
        '- 候选仅在测试副本中沿真实骨架路径切分，保留折点、完整路径和像素支持；参考 JSON 仅由独立评分程序读取。',
        '- 主指标：`platform_metrics_v3`，20px 容差，逐例宏平均；另用一对一几何匹配及网络/线段双层一对一匹配排查匹配顺序假象。',
        '- 引脚切点保护另做 1px、4px 对照，不与路径修正混在一起判断。','',
        '## 完整流程结果','',
        '| 指标 | 原算法 | 路径修正 | 变化 |','|---|---:|---:|---:|']
    for key,label in [('component_f1','元件 F1'),('pin_f1','引脚 F1'),('connectivity_f1','连通 F1'),('wire_f1','导线 F1')]:
        b,c=baseline['macro'][key],candidate['macro'][key]
        text.append(f'| {label} | {b:.6f} | {c:.6f} | {c-b:+.6f} |')
    text.extend([f"| 本地总分 / 100 | {baseline['score_100']:.4f} | {candidate['score_100']:.4f} | {gain:+.4f} |",'',
        f"一对一几何 F1（忽略分网）：{checks['baseline_geometry']:.6f} → {checks['candidate_geometry']:.6f}。",
        f"双层一对一导线代理 F1：{checks['baseline_robust']:.6f} → {checks['candidate_robust']:.6f}。",'',
        '## 逐例导线结果','', '| 案例 | 原算法 | 路径修正 | 变化 |','|---|---:|---:|---:|'])
    for row in rows:
        text.append(f"| {row['case']} | {row['baseline_wire']:.4f} | {row['candidate_wire']:.4f} | {row['wire_delta']:+.4f} |")
    text.extend(['','## 单独保护切点','', '| 方案 | 导线宏平均 F1 | 本地总分 |','|---|---:|---:|'])
    for key,label in [('existing_output','现有清理'),('protect_1px','保护半径 1px'),('protect_4px','保护半径 4px')]:
        m=protection['macro'][key]
        text.append(f"| {label} | {m['wire_f1']:.6f} | {m['score_100']:.4f} |")
    text.extend(['','路径修正之后再保护切点：','', '| 方案 | 导线宏平均 F1 | 本地总分 |','|---|---:|---:|'])
    for key,label in [('existing_output','仅路径修正'),('protect_1px','路径修正 + 保护 1px'),('protect_4px','路径修正 + 保护 4px')]:
        m=combined['macro'][key]
        text.append(f"| {label} | {m['wire_f1']:.6f} | {m['score_100']:.4f} |")
    text.extend(['','## 真实图复核','',
        '第 11 页 USB 接口附近：原算法把折线换成斜穿器件的直线，路径修正保留原图走线。图片只叠加模型预测，没有使用参考答案。','',
        f'![真实导线对照]({(HERE/"case11_wire_comparison.png").as_posix()})',''])
    control=read(HERE/'control_integrity.json')
    text.extend(['','## 验证边界与文件完整性','',
        f"- 原算法缓存重放与 PNG 基线完全一致：{control['identical_cases']}/{control['cases']} 例。",
        f"- 三组检测/OCR/引脚/U-Net 缓存逐字节相同：{control['frozen_neural_inputs_identical']}。",
        f"- 正式部署检查：{integrity['files']} 个源码/配置/模型文件，变化数 {len(integrity['changed'])}。",
        '- 这些都是本地开发集结果，不能写成平台实测成绩；未重新训练、构建镜像或提交平台。',
        '- 构造几何回归检查 6 项通过，详见 `synthetic_checks.json`。',
        '- 逐例四项成绩和独立匹配结果：`comparison.json`；切点保护：`current_protection.json`；原始日志在各实验输出目录。','',
        '## 复现入口','',
        '- `prepare_candidate.py`：从原样快照构造一次性实验实现。',
        '- `path_preserving_split.patch`：供后续正式落地审查的单文件补丁。',
        '- `run_candidate.py`：相同模型缓存下重跑全部下游流程。',
        '- `validate.py`：部署文件哈希核验、评分和独立切点保护重放。',
        '- `compare.py`：四项宏平均、逐例差异和一对一匹配复核。',''])
    (HERE/'验证报告.md').write_text('\n'.join(text),encoding='utf-8')
    print(HERE/'验证报告.md')


if __name__=='__main__':main()
