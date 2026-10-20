"""Summarize the frozen ten-page extension without running inference or tuning."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VARIANT='restore_physical_contacts'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summary(rows):
    result={}
    for name in ['baseline',VARIANT]:
        result[name]={'metrics':{k:sum(r['variants'][name]['metrics'][k] for r in rows)/len(rows)
                                 for k in rows[0]['variants'][name]['metrics']},
                      'wire_proxy':sum(r['variants'][name]['one_to_one_network_wire_f1'] for r in rows)/len(rows),
                      'geometry_f1':sum(r['variants'][name]['geometry_f1'] for r in rows)/len(rows)}
    for key,extract in [('wire_proxy',lambda v:v['one_to_one_network_wire_f1']),
                        ('connectivity',lambda v:v['metrics']['connectivity_f1'])]:
        deltas=[extract(r['variants'][VARIANT])-extract(r['variants']['baseline']) for r in rows]
        result[key+'_counts']={'improved':sum(d>1e-9 for d in deltas),
                              'unchanged':sum(abs(d)<=1e-9 for d in deltas),'regressed':sum(d< -1e-9 for d in deltas)}
    return result


def main():
    initial=read(ROOT/'small_set_results.json')
    extension=read(ROOT/'remaining_ten_results.json')
    assert set(initial['cases']).isdisjoint(extension['cases'])
    rows=sorted(initial['per_case']+extension['per_case'],key=lambda r:int(r['case']))
    assert {r['case'] for r in rows}=={str(i) for i in range(1,14)}
    for row in rows:
        for field in ['component_f1','pin_f1']:
            assert row['variants']['baseline']['metrics'][field]==row['variants'][VARIANT]['metrics'][field]
    result={'variant_fixed_before_extension':VARIANT,'initial_cases':initial['cases'],
            'extension_cases':extension['cases'],'extension':summary(extension['per_case']),
            'all_thirteen':summary(rows),'per_case':rows,'new_neural_inference_calls':0,
            'new_training':False,'deployment_changed':False,
            'caveat':'Wire proxy is diagnostic and is not an official platform score. Legacy wire scorer remains order-dependent.'}
    (ROOT/'expansion_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 电源导线恢复：其余十页扩展验证','',
        '2026-10-20。新增测试第 1、2、3、4、7、8、9、10、11、12 页。规则沿用前三页实验，不调参；全部复用缓存，没有重新推理或训练。正式部署文件核验无变化。','',
        '固定评估方案：恢复有现成符号/OCR/接点证据的电源物理子网及电源接点。基线是上一轮已验证路径修正的测试副本，尚未部署到 D 盘。','',
        '## 新增十页','', '| 指标 | 基线 | 恢复方案 |','|---|---:|---:|']
    for key,title in [('connectivity_f1','本地连通 F1'),('component_f1','元件 F1'),('pin_f1','引脚 F1')]:
        lines.append(f"| {title} | {result['extension']['baseline']['metrics'][key]:.4f} | {result['extension'][VARIANT]['metrics'][key]:.4f} |")
    lines.extend([f"| 一对一导线代理 | {result['extension']['baseline']['wire_proxy']:.4f} | {result['extension'][VARIANT]['wire_proxy']:.4f} |",
        f"| 一对一几何 F1，忽略分网 | {result['extension']['baseline']['geometry_f1']:.4f} | {result['extension'][VARIANT]['geometry_f1']:.4f} |",'',
        '导线代理 5 页提升、1 页略降、4 页不变；连通 6 页提升、4 页不变，无退步。','',
        '## 加上前三页后的十三页汇总','', '| 指标 | 基线 | 恢复方案 |','|---|---:|---:|'])
    total=result['all_thirteen']
    lines.extend([f"| 本地连通 F1 | {total['baseline']['metrics']['connectivity_f1']:.4f} | {total[VARIANT]['metrics']['connectivity_f1']:.4f} |",
        f"| 一对一导线代理 | {total['baseline']['wire_proxy']:.4f} | {total[VARIANT]['wire_proxy']:.4f} |",
        f"| 一对一几何 F1，忽略分网 | {total['baseline']['geometry_f1']:.4f} | {total[VARIANT]['geometry_f1']:.4f} |",'',
        '元件和引脚评分均不变。导线代理累计 7 页提升、1 页略降、5 页不变。','',
        '## 新增页逐例','', '| 案例 | 连通：基线 → 恢复 | 导线代理：基线 → 恢复 | 恢复网络数 |','|---|---:|---:|---:|'])
    for row in extension['per_case']:
        b,c=row['variants']['baseline'],row['variants'][VARIANT]
        lines.append(f"| {row['case']} | {b['metrics']['connectivity_f1']:.4f} → {c['metrics']['connectivity_f1']:.4f} | {b['one_to_one_network_wire_f1']:.4f} → {c['one_to_one_network_wire_f1']:.4f} | {len(c['restored'])} |")
    lines.extend(['','## 判断与限制','',
        '改善延续到了前三页以外的样本，说明效果具有一定覆盖面，尤其是连通性。第 1 页导线代理 0.6694 → 0.6667 略降，应保留为后续回归样本。',
        '导线代理采用网络与线段双层一对一匹配，不能当作平台 NetLineF1。旧本地导线评分仍有网络顺序依赖；新增十页该旧分为 0.5313 → 0.5272，不能据此直接断言平台升降。',
        '正式代码、权重、配置未修改。待导线评分口径进一步核实后，再决定是否落地。','',
        '原始扩展结果：`remaining_ten_results.json`；合并汇总：`expansion_summary.json`。',''])
    (ROOT/'十页扩展验证报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['extension','all_thirteen']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
