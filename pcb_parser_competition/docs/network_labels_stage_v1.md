# 网络标签与逻辑飞线：阶段输入输出和运行

本阶段使用现有 YOLO 标签符号框、RapidOCR 文字、U-Net 导线图及引脚点进行网络标签关联。不新增训练，不修改三类模型权重，不使用传统原图分割。几何关联与 NetworkX 分组在已有模型输出上进行。

## 本轮行为

标签符号明确、文字拼写有证据、接入网络不歧义，且同一PNG内有至少两个不同位置的同名普通标签时，才构建逻辑网络候选。名称按实际OCR拼写匹配，保留原文及备选；不把空格换成下划线、不改大小写、不进行模糊匹配。重复裁剪或重复检测框不算第二个标签位置。

如果网名拼写发生冲突，只有至少3种OCR视图支持一个完整拼写、且领先其他拼写至少2种视图时才使用该已观察到的拼写。不能把同一OCR模型的多个视图当作统计独立样本。其余冲突保留待复核。

仅靠文字附近有线端点不能证明它是网名。例如0026里的说明文字“Differential”、0012的BGA编号都可能通过距离关联，所以归为 `endpoint_text_role_unverified`，不会触发逻辑合并。实体元件的位号/型号/参数、已绑定引脚文字和元件内部文字会被排除。VCC/GND等电源名以及模型识别为电源/地的符号单独保存，电源语义尚未实现。

`net_short`检测类在这批图里常包含菱形标签符号，U-Net会在其内部形成左右两个导线片段。只有两个相对的模型引脚簇、附近实体元件能确定外部文字侧时，才使用该符号的两个接触网络。此处局部语义仍是开发假设，不适用于所有符号规范。单个标签位置不会凭此形成飞线。

逻辑合并后仍保留原物理网络、所有引脚候选坐标及物理线段，另存物理网络→逻辑网络映射和每次合并原因。没有凭空生成两处标签之间的长导线。严格出口只有完整成员、标签组件和引脚信息都具备时才产生飞线的零长度edges；不能通过丢弃未知引脚让hyperGraph看似完整。

## 阶段接口和检查指标

| 步骤 | 输入 | 输出 | 检查项 |
|---|---|---|---|
| 输入冻结 | PNG、graph_proposals、pin_text_result、texts、component_records | 校验记录 | 图像SHA、大小、坐标原点、引脚输入SHA、U-Net及图配置符合采用版本 |
| 文字角色与拼写 | 原始OCR/视图、元件字段、引脚字段 | label_bindings.sites | 拒绝/待决原因、原字串、视图支持数；无网名静默改写 |
| 接入物理网络 | 标签符号框、原tip、骨架端点 | 锚点、physical network IDs | 符号/端点证据、候选距离与歧义；原tip不移动 |
| 逻辑分组 | 同名不同位置的普通标签 | logical_graph_proposals | 单例不形成飞线；电源/跨图/拼写近似不合并；交点、断线和冲突有门槛 |
| 部分出口 | 完整组件/引脚字段、逻辑组 | partial_submission、export_coverage | 完整pin集合和引用、左下坐标、两点线段；未知标签pin阻止严格出口 |
| 离线比较 | 同一组图预测＋逻辑参考快照 | evaluation、report | 条件与全量引脚对代理同时报告；原始/视觉补充参考均保留；official_NetF1为空 |

当前结果见 [阶段报告](../runs/s9_network_labels_v1/report.md)。26张旧开发页保留1477个引脚，应用0170中ADC_IN12的一组逻辑候选，将3个物理片段归为一个逻辑组，未新增物理导线。严格部分出口仍为194个元件、32个字段齐全引脚、0个net。

## 关于参考数据和指标

源0170的C_114关联T_42含ADC_IN12文字，但没有`netLabel`角色，所以原转换器漏掉该角色。助手查看原PNG及源字段后，将这一处角色复核单独保存到 `data/reviews/network_labels_logical_v1.json`；原源数据、训练标注和tip坐标没有修改。该文件只由离线评估读取，不输入预测。

在这份普通标签参考与单例视觉补充下：条件引脚对F1从74.74%到75.87%，全量参考引脚对代理F1从38.04%到38.79%；新增14个正确引脚对，FP计数不变。这不是官方NetF1，也不是独立泛化评测。新增参考增加了应有的逻辑连接分母，不能直接与上轮物理参考的75.43%/38.22%混比。

未加入视觉角色补充时的结果同时保存在 `raw_source_reference_aggregate`：原草稿会把这14个新增关系计为FP。复核只处理这一例，不能据此宣称全部标签角色准确。复核裁剪会显示说明文字、BGA编号、OCR粘连等残余候选；只有带符号证据且通过门槛的组可能应用。

## 桌面项目运行

项目：`C:/Users/Pc/Desktop/新建文件夹/pcb_parser_competition`。使用已有 `.venv-gpu`，本阶段没有新增依赖。

```powershell
.\scripts\run_network_labels.ps1 -Image data/standardized/0170/0170.png -Device cuda:0 -Output runs/network_label_predictions/0170
```

该入口从PNG依次运行检测、U-Net、OCR、物理导线图和标签关联，不读取CAD/源标注。默认使用保守v1导线图。加 `-LearnedJunctionPilot` 可以显式试验上一阶段交点CNN，不替换旧默认配置。

```powershell
.\.venv-gpu\Scripts\python.exe scripts/batch_network_labels.py
.\.venv-gpu\Scripts\python.exe scripts/evaluate_network_labels.py --reuse-references
.\.venv-gpu\Scripts\python.exe -m unittest discover -s tests -p 'test_network_labels.py'
```

冻结模型缓存调试时必须同时提供 `--physical-graph`、`--pin-text`、`--texts`，三者及PNG的SHA和坐标契约都要匹配。输出目录可独立指定。离线参考快照已转移，不必重新解析原Dataset300文件即可复算；生产入口不会读取该快照。

默认配置为 `configs/network_labels_v1.json`。可调整符号文字距离、端点距离、匹配歧义门槛、拼写视图支持数，或把 `enable_ordinary_label_merges` 关闭以只观察绑定候选。修改时使用新的输出目录，并核对错误图及原始/补充参考两套分数。

## 下一阶段缺口

优先解决网名的完整OCR和角色判断，尤其ADC_IN11/ADC IN11、MCU_RX/MCU RX等格式差异；需要图像证据而不是直接替换字符。随后补标签组件身份、编号及pinname，扩大真标签/说明文字/引脚文字的复核集。电源逻辑、断线恢复和完整赛事JSON仍待完成。
