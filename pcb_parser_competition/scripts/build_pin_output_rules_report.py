"""Coverage report, explicitly not a recognition accuracy evaluation."""
from run_pin_output_rules import ROOT, read, save, sha


def main():
    stage=ROOT/'runs/s12_pin_output_rules';batch=read(stage/'predictions/batch_summary.json')
    if not batch.get('completed'):raise ValueError('Batch incomplete')
    a=batch['aggregate'];rows=batch['cases']
    power={k:sum(r['power_guard'][k] for r in rows) for k in ['power_components','power_without_owned_tip','networks_with_possible_omitted_power_member']}
    example=stage/'predictions/0002'
    geometry=read(example/'candidate_edge_geometry.json')
    text=f'''# 引脚输出约定与小网络验证

本阶段保持检测器、U-Net、OCR文字、引脚编号、1477个原tip及网络成员不变。补充名称字段并区分保守出口与候选出口，尚未完成整图赛事提交。

## 同26页开发数据的出口覆盖

| 出口 | 元件 | 字段齐全pin | net |
|---|---:|---:|---:|
| S11原出口 | {a['before']['exported_components']} | {a['before']['exported_pins']} | {a['before']['exported_nets']} |
| 新保守出口 | {a['conservative']['exported_components']} | {a['conservative']['exported_pins']} | {a['conservative']['exported_nets']} |
| 被动器件名称候选出口 | {a['candidate']['exported_components']} | {a['candidate']['exported_pins']} | {a['candidate']['exported_nets']} |

保守出口补{a['label_names_added']}个已接受唯一普通标签的pinname。候选出口另补{a['candidate_passive_names_added']}个被动器件名称，要求已有合法编号、至少2个同位置OCR视图、置信度≥0.98及编号分配间隔≥15；没有按左右/上下猜编号。候选名称来自“编号作名称”约定，不是新增OCR识别能力。这是出口覆盖计数，不是准确率、PinF1或NetF1。

赛题PDF第25页的RUN_1例子使用标签文字作pinname，R1例子使用“1”作pinname。本实现保留这些例子的证据来源；将编号名称约定扩展到电容、电感和晶振只放入候选出口，官方Golden未验证。原始草稿标注常有空名称，也不能证明官方接受该扩展。空名称仍保留未决；已有编号/名称不覆盖。

## 小网络与电源遗漏检查

0002候选JSON的网络为 `(C86.1,IC2.R8,X2.3)`。已通过原图视觉复核：XTALN/R8线路经XN节点分支到晶振X2的3脚和C86的1脚，局部可见三端关系一致；未核验官方pinname及5px位置评分。该网包括全部3个模型成员，没有删掉未知pin来凑完整网络。见 `predictions/0002/candidate_net_01.png` 与旁路审查记录。

该网络原导线图出口已有{geometry['input_pixel_segments']}段，候选JSON仍为{geometry['output_segments']}段，本例没有进一步减少段数。新增有界压缩接口保持每条路径的端点固定、最大偏离≤1个原图像素，导线图与pin坐标没有改变，也未跨断线补线。正式NetLineF1仍未知。

0002另一条+1V8网络因检测到电源符号但没有所属/接入tip被扣留；本轮全量有{power['power_components']}个电源类检测，其中{power['power_without_owned_tip']}个没有已分配tip，{power['networks_with_possible_omitted_power_member']}个网络触发可能遗漏电源成员提示。后者是6px邻近检查的候选数量，需要复核，不能当错误率。候选出口实际扣留{a['candidate_nets_excluded_by_power_guard']}个原本可能导出的网。没有生成电源身份后缀或同名电源逻辑合并。

## 运行与产物

在项目目录执行：

```powershell
.\\scripts\\run_pin_output_rules.ps1 -Image data/standardized/0002/0002.png -Device cuda:0
```

默认输出 `runs/pin_output_predictions/0002`。`partial_submission.json` 为保守部分输出；`submission_candidate.json` 为含名称约定及几何压缩的开发候选，当前两者均非整图完整赛事提交。对应coverage标明profile、缺口与官方未验证状态。`pin_field_changes.json`、`power_review_queue.json` 和 `candidate_edge_geometry.json` 保存每次补全/扣留/压缩证据。

PNG入口不读取该图片的CAD、源标注或参考JSON；缓存入口会校验原PNG、pin/OCR字节、模型及配置来源。`data/reviews/pin_output_rules_v1.json`只存助手离线视觉核验，未用于训练或提升分数。

16项新测试与13项原字段出口回归测试已在主工作区通过。无新增训练或依赖安装。下一阶段重点是无文字的被动器件编号学习、电源tip漏检与电源身份/作用域规则；当前被动器件编号未决仍是主要缺口。
'''
    verify=stage/'runtime_verification.json'
    if verify.is_file():
        v=read(verify)
        text+=f"\n## 桌面项目验证\n\n桌面项目16项新测试、13项字段回归通过；26页缓存回放计数及JSON与主工作区一致。{v['protected_files_unchanged']}个受保护上游文件SHA未变。0002完整PNG入口不使用旧检测、U-Net、OCR或ownership输入缓存，保留{v['fresh_png_pin_count']}个原tip；保守/候选出口分别为{v['fresh_export']['conservative']['exported_pins']}/{v['fresh_export']['candidate']['exported_pins']}个pin、{v['fresh_export']['conservative']['exported_nets']}/{v['fresh_export']['candidate']['exported_nets']}个net。\n"
    (stage/'report.md').write_text(text,encoding='utf-8')
    save(stage/'audit.json',{'aggregate':a,'power_review':power,'old_development_pages':len(rows),
                           'original_pin_points_preserved':1477,'recognition_accuracy_evaluated':False,
                           'official_PinF1':None,'official_NetF1':None,'complete_competition_JSON':False})


if __name__=='__main__':main()
