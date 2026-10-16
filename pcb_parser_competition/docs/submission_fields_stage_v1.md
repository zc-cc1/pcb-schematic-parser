# 标签身份与输出字段阶段 v1

本阶段在普通标签/逻辑网络之后运行。输入为PNG及经过hash核对的网络、pin和OCR预测；输出为部分赛事JSON、字段证据、身份候选与逐网络缺口复核图。没有重新训练YOLO或U-Net。

核对赛题0803Rev1第14–18页后采用以下策略：

- 唯一普通标签：至少3个已有OCR视图支持、有检测符号和归属pin证据时，补入component身份和Name。这仍是模型预测，没有取得官方准确率。
- 重复标签：保存原始标签名及视觉顺序的后缀候选。赛题后缀依赖原始component ID自然排序，PNG中无法恢复，所以候选不进入严格部分JSON。空格/下划线近似的不同拼写也阻止“唯一身份”判定，不改字串或合并网络。
- 单引脚标签：没有可靠编号文字或冲突编号候选时，可按PDF第18页(c)生成pin_1。证据注明为格式规则，不能算识别出编号字符。Net_short、IC和二端元件不会按空间次序编号。
- pinname：保留OCR识别和未知状态，不用网名、空字符串或预设模板补齐。未知名称仍阻止引脚和网络进入完整字段出口。
- 局部OCR：已有标签裁剪按2/3/4倍识别，保留原文、分数和建议。小裁剪依然可能丢下划线，目前仅作复核证据，不自动替换。

## 当前开发结果

26页新增5个标签身份预测，单引脚格式编号10个；部分JSON元件194→199，字段齐全pin32→32，net0→0。保留1477个原引脚点及原网络成员关系，没有训练集修改。上述是输出覆盖统计，不是准确率。

399个多引脚网络候选中，333个未记录交叉/断线阻断但仍缺字段；未记录阻断不证明电气拓扑正确。每个网络的成员、字段缺口、元件框和tip见复核队列。28个标签局部复读得到2个建议，均未应用。

主要文件：

| 文件 | 用途 |
|---|---|
| completed_pin_text_result.json | 新增字段的pin副本，保留原坐标及证据 |
| completed_component_records.json | 元件身份更新副本 |
| identity_proposals.json | 原始网名、后缀候选、接受/阻断原因 |
| label_roi_readings.json | 逐尺度OCR原始复读证据与待复核建议 |
| network_field_review_queue.json | 按缺字段数排序的网络与逐pin缺口 |
| network_fields_*.png | 对应原图裁剪及tip序号 |
| partial_submission.json | 经本地结构/引用/坐标校验的部分输出 |
| export_coverage.json | 不能导出的pin/net及原因 |

## 在桌面项目运行

```powershell
.\scripts\run_submission_fields.ps1 -Image data/standardized/0033/0033.png -Device cuda:0
```

这是完整PNG入口，会运行检测、U-Net、OCR、物理网络、普通标签及本轮字段模块。`-SkipLabelRoi`只跳过最后的证据复读。

冻结旧26页调试：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/batch_submission_fields.py
.\.venv-gpu\Scripts\python.exe scripts/audit_submission_fields.py
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p test_submission_fields.py
```

配置为configs/submission_fields_v1.json。OCR支持视图数、置信度、复核图数量及局部尺度可调整；名称补空串和拼写替换尚未实现，改动相关策略会报错，避免配置看似生效。

运行不读CAD或配套真值JSON。批量audit只审核预测的结构、覆盖与来源；它不计算独立识别准确率，也不是官方评测器。原有run_network_labels和训练入口继续可用。

下一项优先补IC/连接器身份与引脚编号/名称的可核验参考，并解决重复标签规范编号。完整赛事JSON和官方NetF1仍未完成。
