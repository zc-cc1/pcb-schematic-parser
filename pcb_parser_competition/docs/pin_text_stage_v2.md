# IC/连接器身份与引脚文字 v2

本阶段新增独立入口run_pin_semantics.ps1，串接v2文字、既有v1物理图、普通标签、字段补全和部分JSON出口。原v1入口、检测器、U-Net、训练权重与标注均保留。

## 实现

- 位号补全：只处理身份未知的IC/框式器件/连接器/运放；以模型预测框和侧面tip范围确定上下标题区域，读取现有OCR证据，支持U1H、H1、CN1、LD1等可见格式。至少两个同位置OCR视图支持，至少一个分数≥0.95。大IC左上角位号不再受框中心距离惩罚。侧边BGA球位不能作为标题；重复位号且原始ID顺序未知时保留未决。
- 型号补全：只选择标题区中实际观察到的型号文字，保留大小写及完整证据，不依赖芯片库或猜型号。
- 引脚文字试验：密集排使用更窄的沿排匹配范围，同一文字区域分组阻止大框跨行；可保留整行多词名称。对于同侧同行、沿法向分开的模型tip假设，只给代表点分配新的OCR证据，所有原坐标与假设均保留。
- 采用策略：编号改法在草稿参考上变差，因此保留v1全部编号和已有非空pinname。只补入confidence≥0.97、至少两个已有OCR视图支持、未占用旧编号/名称OCR区域的新名称。完整v2改法单列experimental_pin_text_result.json。

没有按编号顺序猜引脚，没有把原tip移到bbox边界，没有按参考改字串，没有把网络名直接当作pinname。多个OCR视图仍来自同一模型，不能当独立验证。

## 已有26页开发结果

| 项目 | v1/上一阶段 | 本轮采用版 |
|---|---:|---:|
| 恢复编号 | 273 | 273 |
| 恢复名称 | 274 | 302 |
| 名称＋归属＋位置代理F1 | 58.46% | 60.69% |
| 条件名称精确率 | 95.00% | 93.47% |
| 条件名称覆盖率 | 77.73% | 84.55% |
| 部分JSON元件 | 199 | 213 |
| 字段齐全的pin | 32 | 88 |
| 严格可导出net | 0 | 0 |

新增可见身份预测14个、型号8个、缺失名称28个。代理F1不是官方PinF1，条件指标只计算归属与坐标匹配且有非空源字符串的范围。数量增加不等于全部新增字段正确。26页都参与开发，没有独立工程测试。

源草稿有空名称，OCR与草稿也存在字符串不一致；这些差异需要逐项看图判断，不能一概归因于参考错误。本轮没有修改参考。条件精确率下降需保留在报告中，不能只报告覆盖增长。原1477个tip、归属索引和物理线段/接入关系保持原样；继承检测类别未修改，框式连接器仍可能被检测成ic。

## 运行

在桌面“新建文件夹”项目根目录运行：

```powershell
.\scripts\run_pin_semantics.ps1 -Image data/standardized/0018/0018.png -Device cuda:0
```

完整图片入口默认输出到runs/pin_semantics_predictions/<图片名>。pin_text、physical、network、submission分别对应文字、物理图、标签和部分JSON。正式推理不读取配套真值JSON、CAD或评估快照。

只调试文字：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_pin_text_v2.py --image data/standardized/0018/0018.png --output runs/pin_text_v2_debug/0018 --device cuda:0
```

重放冻结开发数据并评估：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/batch_pin_text_v2.py
.\.venv-gpu\Scripts\python.exe scripts/evaluate_pin_text_v2.py
.\.venv-gpu\Scripts\python.exe scripts/batch_pin_semantics.py
.\.venv-gpu\Scripts\python.exe scripts/build_pin_text_v2_report.py
```

最后两条重用已校验的U-Net概率，逐页检查物理几何/网络接入关系一致。evaluate脚本只用于离线草稿评估，不被PNG推理入口导入。

## 产物

| 文件 | 用途 |
|---|---|
| pin_text_result.json | 采用版引脚字段及证据 |
| experimental_pin_text_result.json | 完整v2试验，编号策略未采用 |
| identity_recovery_debug.json | 标题候选、接受身份、重复/歧义阻断 |
| field_adoption_debug.json | 新名称的采用或阻断原因 |
| field_review_queue.json、field_review_*.png | 新增身份和名称的42项待复核记录 |
| downstream/<case>/submission/partial_submission.json | 部分赛事结构JSON |
| downstream/<case>/submission/network_field_review_queue.json | 网络仍缺少的字段 |

复核队列初始reviewed=false，不会自动写入训练标注。下一步应集中解决被动器件/电源符号的空名称和自动编号规则，并核验一个小电路的完整网络出口；正式Golden/评测器仍待取得。
