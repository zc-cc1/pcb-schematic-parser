# OCR与元件文字绑定优化实测

已采用 semantic_v2，保持v3检测器、SAHI、融合和逐类阈值不变。原两例从PNG重新OCR与局部复读，复用冻结检测预测；额外4页也使用真实图片预测。两例位号覆盖率达到原内部90%目标，但6页总体仍未达到该目标，不能宣称整个OCR阶段已经验收完成。

## 原两例：同一101字段参考前后比较

0001/0002的54个位号、41个数值、6个型号参考与优化前逐字一致，文件哈希校验通过。优化前采用归档的 semantic_v1 实际输出，不与更早的整图最近邻方案混用。

| 本地指标 | 优化前 semantic_v1 | 优化后 semantic_v2 |
|---|---:|---:|
| 位号绑定精确率 | 42/44，95.45% | 49/51，96.08% |
| 位号真实覆盖率 | 42/54，77.78% | 49/54，90.74% |
| 数值严格匹配率 | 37/41，90.24% | 40/41，97.56% |
| 型号严格匹配率 | 5/6，83.33% | 5/6，83.33% |

0002的16个适用字段继续全部正确。原两例已正确检测并匹配的49个位号现在49/49恢复；总体仍以54为分母。两条已绑定位号的微小TP检测框没有达到严格归属框IoU门槛，因此位号精确率仍计49/51。型号5/6受Q1检测类别错误限制。

## 额外4页与6页总结果

新增0007（蓝色密集开关）、0042（灰度密集电阻文字）、0131（蓝色电源图）、0229（无数字位号的芯片及电容）共54个人工复核字段。4页也参与了这次通用规则开发，属于开发回归集，不能当作未参与调参的独立测试。

| 本地指标 | 优化前 semantic_v1 | 优化后 semantic_v2 |
|---|---:|---:|
| 位号绑定精确率 | 9/9，100.00% | 20/20，100.00% |
| 位号真实覆盖率 | 9/30，30.00% | 20/30，66.67% |
| 数值严格匹配率 | 8/21，38.10% | 18/21，85.71% |
| 型号严格匹配率 | 1/3，33.33% | 3/3，100.00% |

六页合计155个字段，覆盖84个位号、62个数值、9个型号：

| 本地指标 | 优化前 semantic_v1 | 优化后 semantic_v2 |
|---|---:|---:|
| 位号绑定精确率 | 51/53，96.23% | 69/71，97.18% |
| 位号真实覆盖率 | 51/84，60.71% | 69/84，82.14% |
| 数值严格匹配率 | 45/62，72.58% | 58/62，93.55% |
| 型号严格匹配率 | 6/9，66.67% | 8/9，88.89% |

额外页面的旧结果由归档 semantic_v1 代码和原配置重新生成，输入为相同冻结检测与此次保存的基础OCR（未加局部复读）；原两页旧结果来自真实归档。比较的是整套OCR后处理与绑定改动，未逐项做消融。不同页面字段数量不同，总体按字段计数汇总，不按页面平均。

## 改动及原因

1. 全局匹配以替代整套分配的代价差判断歧义，解决局部候选接近却全局唯一的R13/R14绑定。加入对齐项，允许留空，减少远处文字误绑。
2. 重合的同类检测只选一个文字归属代表，解决重复L2框争抢文字。检测JSON和检测指标本身保持原样；调试文件记录重复框对应关系。
3. 对低置信度、大小写异常和可能粘连导线符号的文字裁剪放大，再用RGB、灰度及图中主要文字色复读。只比较真实OCR得到的字符串：U2和3.3uH来自原图复读，没有直接大小写转换或0/O替换。
4. 同位置的数值替代观测按识别置信度及几何证据选择；数值与位号重叠、引脚球号截断的候选被拒绝，避免部分数字抢走电阻数值。
5. 支持S/SW、LED、JP等位号，以及带标点或同前缀粘连的位号、重复完整单位数值。保留原文字大小写、空格与单位。
6. 无位号器件可保留可靠字段，例如芯片内部TMP118及电容10nF；位号仍为空。无位号型号限定为芯片内部、足够长且具有字母数字证据的候选，排除GPIO等引脚标签，避免误取外部网络文字。
7. 修复RapidOCR局部复读改变运行开关后影响下一张图的问题；每次基础OCR明确开启检测、方向和识别。缓存核验图像哈希、基础OCR配置与局部复读配置。

模型仍为已安装的RapidOCR识别模型，没有新增训练或下载另一组OCR权重。图片预测入口只读取PNG、检测预测和配置；复核JSON仅供独立评价与审查图生成，不作为推理输入。

## 评价口径与未解决问题

归属使用类别一致、框IoU≥0.5的最大匹配一对一对应，再严格比较字符串。位号覆盖率=正确位号/全部参考位号；位号绑定精确率=归属与位号均正确/全部已绑定位号。数值与型号报告参考字段上的严格匹配率，尚未统计完整图上所有字段的误报精确率。字段归属与位号分别评分，本表不是整个元件所有字段同时正确的比例，也不是mAP或赛事ComponentF1。

参考是本地人工复核，范围有限，尚无正式Golden/评测器。新增参考中图上L1为2.2uH、两个开关均印成S10，按可见原字符记录；型号互相冲突的0007芯片未列入已核验字段，并在参考文件中注明原因。这些离线参考修改没有改动原始Dataset300或预测字符串。

剩余重点是0007蓝色文字与线路重叠、两个S10造成身份冲突，以及L2单位读成4700hm的0/O歧义。后者记录疑点并留空，未猜成470Ohm。Q1类别、R28漏检与微小TP框归属问题属于上游检测或匹配限制。重复位号的赛事JSON身份如何表达仍待正式规则，不凭空重编号。

全部失败字段保留在分母：

| 页面 | 元件 | 字段 | 应为 | 预测 | 失败类型 |
|---|---|---|---|---|---|
| 0001 | L2 | value | 470Ohm | 空 | empty_field |
| 0001 | Q1 | designator | Q1 | 空 | detector_missed_or_wrong_class |
| 0001 | Q1 | Name | RE1C001ZPTL | 空 | detector_missed_or_wrong_class |
| 0001 | TP1 | designator | TP1 | 空 | detector_missed_or_wrong_class |
| 0001 | TP2 | designator | TP2 | 空 | detector_missed_or_wrong_class |
| 0001 | TP3 | designator | TP3 | 空 | detector_missed_or_wrong_class |
| 0001 | TP7 | designator | TP7 | 空 | detector_missed_or_wrong_class |
| 0007 | S7 | designator | S7 | 空 | empty_field |
| 0007 | S3 | designator | S3 | 空 | empty_field |
| 0007 | S5 | designator | S5 | 空 | empty_field |
| 0007 | S10 | designator | S10 | 空 | empty_field |
| 0007 | S2 | designator | S2 | 空 | empty_field |
| 0007 | S10 | designator | S10 | 空 | empty_field |
| 0007 | S0 | designator | S0 | 空 | empty_field |
| 0007 | UNNUMBERED_C_10 | value | 30p | 空 | empty_field |
| 0007 | UNNUMBERED_C_15 | value | 10 K | 10K | wrong_text_or_owner |
| 0042 | R28 | designator | R28 | 空 | detector_missed_or_wrong_class |
| 0042 | R28 | value | 0 | 空 | detector_missed_or_wrong_class |
| 0131 | L1 | designator | L1 | 空 | empty_field |
| 0131 | TP3 | designator | TP3 | 空 | detector_missed_or_wrong_class |

## 验证、耗时与产物

47项解析/融合/文字回归通过，另在文档Python环境通过3项CAD清点测试，共50项。已检查两个默认配置与semantic_v2配置一致，权重哈希、原两例检测预测哈希及参考哈希不变；当前源码哈希与全部6页运行摘要对应。

| 页面 | 冻结检测数 | 已绑定位号 | 已绑定数值 | 已绑定型号 | 基础OCR调用 | 局部复读墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| 0001 | 96 | 43 | 33 | 4 | 38.61s | 2.65s |
| 0002 | 13 | 8 | 7 | 1 | 4.55s | 0.20s |
| 0007 | 29 | 5 | 4 | 0 | 11.82s | 2.55s |
| 0042 | 10 | 5 | 5 | 0 | 2.78s | 0.16s |
| 0131 | 16 | 10 | 9 | 2 | 9.95s | 0.02s |
| 0229 | 8 | 0 | 1 | 1 | 2.35s | 0.00s |

已绑定数不代表全部正确。基础OCR时间仅为引擎调用累计；局部复读时间包括局部预处理和引擎调用，两项不是相同口径，均不包括完整进程冷启动和输出写盘，不能直接推算50例总耗时。原两例新OCR未复用文字缓存。

主要文件：

- 配置：`configs/baseline_v3_adopted.json`，版本快照`configs/component_text_v2.json`。
- 实现：`src/pcb_parser/component_text.py`、`src/pcb_parser/text_recovery.py`。
- 原两例：`runs/s3_component_text_quick_dev/`；额外4页：`runs/s3_component_text_crosspage_v2/`。
- 每页：`base_texts.json`、`texts.json`、`recovery_debug.json`、`component_records.json`、`binding_debug.json`、`components_partial.json`及`overlay.png`。token来源和局部复读原字符均可追溯。
- 原版归档：`runs/s3_component_text_v1_archive/`；额外页原版重建：`runs/s3_component_text_crosspage_v1/`。
- 6页汇总：`runs/s3_component_text_quick_dev/optimization_aggregate.json`；参考：`data/reviews/component_text_quick_dev.json`与`data/reviews/component_text_crosspage.json`。

在项目根目录运行：

```powershell
.\scripts\run_ocr_binding.ps1 -Device cpu
# 只重跑文字流程，冻结检测：
.\scripts\run_ocr_binding.ps1 -Device cpu -ReuseDetections
# 只调绑定，文字配置未变时复用经过校验的缓存：
.\scripts\run_ocr_binding.ps1 -Device cpu -ReuseDetections -ReuseTexts
& .\.venv-gpu\Scripts\python.exe scripts/evaluate_component_text.py --baseline-run runs/s3_component_text_v1_archive/prediction
& .\.venv-gpu\Scripts\python.exe scripts/build_text_optimization_report.py
```

修改基础OCR或局部复读参数后需重新识别，不能复用旧的完整文字缓存；`--base-texts-run`仅用于基础OCR不变时复用基础观测并重跑局部复读。原始图像和标注保持只读。

下一步可接入二端器件引脚tip定位与导线提取，同时用复杂蓝色图继续追踪OCR残余错误。当前pins/nets仍为空，输出是组件字段部分结果；出口H-y约定也仍需正式Golden校准。
