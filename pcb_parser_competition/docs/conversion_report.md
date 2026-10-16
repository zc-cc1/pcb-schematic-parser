# Dataset300 转换交付报告

日期：2026-10-11（Asia/Shanghai）。转换器版本：0.3。

已处理 300 例，成功生成 300 份参考JSON和同尺寸PNG，失败 0 例。原始图片及标注哈希在转换前后核验一致。

输出顶层为 `components`、`pins`、`nets`；元件含 `type/Name/value/bbox`，引脚含 `pinname/point`，网络含 `hyperGraph/edges`。每条edge恰有两个点。

## 文件位置

- 标准化数据：`data/standardized/<case_id>/<case_id>.json` 及同目录PNG。
- 全量清单：`data/standardized/manifest.jsonl`。
- 逐例来源、映射、警告：`data/conversion_reports/<case_id>.json`。
- 人工复核队列：`data/conversion_reports/review_queue.jsonl`。
- 全量统计：`data/conversion_reports/summary_all.json`。
- 结构和引用复查：`data/conversion_reports/verification.json`。

## 实际结果

| 项目 | 数量 |
|---|---:|
| 元件 | 12789 |
| 输出引脚 | 29652 |
| 网络 | 9899 |
| 两点线段 | 42230 |
| 保持解码像素的重新编码图片 | 92 |
| 修复相同重复键的案例 | 10 |
| 相同重复键项 | 26 |
| 位号未确定的元件 | 4546 |
| 含未确定元件的案例 | 297 |

## 已做的处理

1. 保留所有原件；92张非PNG内容解码后另存PNG，其余PNG复制。300例的尺寸和RGBA解码像素均保持一致。
2. 忽略原始deleted对象；元件位号采用声明文字或有记录的候选，重复位号按原始component ID自然排序编号。
3. 解析文字引用、类别、型号和值；对图像已人工确认的0001/0002字段使用独立overrides文件修正。
4. 引脚采用tip，拆分多编号；无有效编号时按显式本地规则生成编号，并逐项标记。
5. 根据原始连接、精确重合端点及声明标签合并网络；保留源路径分段并去掉完全相同的两点段。
6. 严格重复键检查发现10例共26项重复连接键；内容均相同，已去重并记录。冲突重复键仍会阻止转换。
7. 来源映射与问题记录放在侧文件，正式结构JSON不混入置信度或调试字段。

## 需要优先复核

下列数量是待核验项数量，不是错误率或F1。

| 问题 | 数量 | 处理建议 |
|---|---:|---|
| 位号缺失/无法唯一确定 | 4530 | 输出UNRESOLVED_*并保存候选；按原图修正 |
| IC型号缺失 | 235 | 核对原图/允许的元件库 |
| R/C取值缺失 | 1499 | 核对用途及单位，保持原文 |
| 本地自动编号 | 22409 | 与官方自动编号顺序核验；不等于已恢复物理编号 |
| 引脚编号冲突 | 447 | 已分配临时编号避免覆盖；核对源编号 |
| 根级引脚缺少可解析所属元件 | 208 | 原记录保留，输出未虚构所属器件 |
| 网络引脚引用仍无法恢复 | 16 | 对照原图及原标注；已记录未导出引用 |
| 未获官方确认的类别映射 | 2031 | 补充官方type全集并修改config |
| 源导线缺少可用路径 | 692 | 核对几何；未按距离虚构连接 |
| 源坐标超出图像范围 | 11 | 记录原坐标，待核对裁剪/外框定义 |

## 验证范围与限制

已重新打开全部300份输出，检查顶层/字段类型、bbox顺序、两点edges、引脚引用、单网归属和重复键；项目结构校验错误为0。转换器9项回归检查通过。

这些是本项目结构与引用校验，尚未运行官方评测器。输出为本地参考标注，不能直接宣称是完整官方Golden或已有识别F1。

出口当前采用H-y、location框；H-1-y和其他框口径可配置重建，须用正式Golden核验。非通用类别、自动编号顺序、标签合并及源路径分段仍是显式待核验规则。

空pinname并不自动代表漏标：无源器件可能没有图示名称；其评分期望需按官方转换规则核验。未确定编号和型号也没有从模型预测中补成答案。

抽查配套SPICE自称reconstructed_hybrid，并带待确认项，因此未用它自动补齐缺失字段。

## 调试样例与重建

- [0002标准JSON](C:/Users/Pc/Desktop/vision/pcb_parser_competition/data/standardized/0002/0002.json)：IC2型号、8个IC引脚编号、X2及C74编号已结合原图核对。
- [0001标准JSON](C:/Users/Pc/Desktop/vision/pcb_parser_competition/data/standardized/0001/0001.json)：修正U1的RP2350B型号及U2位号，其余未确定项见报告。

```powershell
$pcbPython = 'C:\Users\Pc\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $pcbPython scripts/convert_dataset.py
& $pcbPython scripts/build_conversion_report.py
```

修改 `configs/annotation_overrides.json` 可以按原始ID修正已核验字段；修改 `configs/conversion.json` 可以调整规则。单例试验使用 `--case-id 0002`，不会改写全量manifest/summary；全量重建后再生成报告。

原始标注只用于离线准备、训练和评测；实际推理入口不能读这些答案文件。
