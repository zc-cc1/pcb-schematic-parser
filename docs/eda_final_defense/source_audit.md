# EDA 精英挑战赛决赛答辩 · 提交材料全面技术审查报告

- 审查对象:`E:\AI_Workplace\精英挑战赛\eda2606k5`(本地比赛提交目录,第一事实来源)
- 赛题:赛题六「AI 驱动的 PCB 原理图自动解析」(芯粒 CAD 和制造浙江省工程研究中心 & 比昂芯科技)
- 审查日期:2026-11-21
- 审查方式:递归检查目录;TAR 与解出目录逐条比对;解包 Docker 镜像中含代码的全部 6 个层并逐层核对;三个并行审查代理深读全部核心源码(约 157 个 Python 文件、2.65 万行)、30 篇开发文档、39+3 个测试文件、249 项文件清单;对本报告的关键数字逐条回查原始文件
- 对照仓库:`zc-cc1/vision`(仅作报告存放与参考;**未将 GitHub 代码视为最终提交版本**,差异在 §3.4 单独记录)

---

## 1. 项目总体概述

**一句话介绍(可直接用于答辩开场):**
我们提交了一个**纯本地、零 LLM Token** 的 PCB 原理图自动解析系统:输入一张原理图 PNG,端到端输出比赛规定的结构化 JSON(元器件 components / 引脚 pins / 连接网络 nets),技术路线是「自训 YOLO 元件检测 + 自训双头 U-Net 导线/引脚分割 + PP-OCRv6 离线 OCR + 约 110 个证据门控的规则修复模块」,以「冻结感知 + 保守事务式修复 + 面向官方标注约定的输出对齐」策略保证输出合法、可追溯、鲁棒。

**提交物形态:** 官方基础镜像 `edarace-agent-base:ubuntu24` 之上构建的参赛镜像 `pcb-parser:v102`(OCI/Docker 归档,3.6 GB,入口 `/submission/run.sh` → `python agent.py -input /input -output /output`)。解出的 `pcb-parser-v102/` 目录与 `pcb-parser-v102.tar` 逐条一致(21 个条目完全相同,配置/代码层 blob sha256 抽查全部吻合),**无需重复解压**。

**规模:** Python 代码 157 个文件、约 26,480 行(src 16,528 / tests 5,395 / scripts 2,532 / 顶层 2,025);41 个配置 JSON;30 篇开发文档;41(39+3)个测试文件、626+ 个 unittest 测试函数、978+ 断言;随包携带 3 个自训/基线 .pt 权重、3 个 OCR ONNX 模型、2 个元件库(1000+337 型号)、STM32F103C8T6 引脚库及数据表 PDF。

---

## 2. 比赛要求与实际完成情况

### 2.1 赛题正式要求(提取自赛题 PDF 全部 30 页)

**任务:** 给定一张 PCB 原理图 PNG,识别并输出规定 JSON(顶层 `components` / `pins` / `nets` 三段):
- `components[key]`: `type`(如 ic/resistor/capacitor…;平台实际用 r/c/l/d/box/led/v 编码)、`Name`、`value`、`bbox [x1,y1,x2,y2]`(左下角坐标系,y 向上);重复位号统一编号 `_1/_2…`
- `pins[comp][pin_N]`: `pinname`、`point{x,y}`;pinNumber 缺失/噪声时自动编号;一个引脚多编号需拆分
- `nets[net]`: `hyperGraph "(U1.A5,R1.1)"` + `edges`(每个 edge 恰好 2 个点);**飞线**:多个同名非电源测试点/网络标签在逻辑上属同一 net,edges 为零长度线段;VCC/GND 除外

**评分:** `FinalScore = FinalScore_Accuracy(100) + FinalScore_Data(10,数据贡献)`
- `FinalScore_Accuracy = 0.95 × Score_Accuracy + 0.05 × (m/M) × Score_Token`
- `Score_Accuracy_SingleCase = 100 × (0.3×ComponentF1 + 0.25×PinF1 + 0.35×NetHypergraphF1 + 0.1×NetLineF1)`
- ComponentF1:key+type+bbox(20px)+文本(IC 比 Name、R/C 比 value,严格区分大小写);PinF1:key→pin number 匹配 + pinname + 坐标 5px;NetHypergraphF1:pin 集合匈牙利匹配;NetLineF1:matched net 内线段两端 5px
- Token:全队按消耗排序赋分(越少越好,最高区间 100 分,占总分 5%)
- **约束**:赛会提供 qwen3.7-plus LLM API 测试环境;全部案例评测 **< 2 小时**;打分案例 30~50 个,**公开:黑盒 = 1:2,中等 40% / 难 60%**;官方数据集 PCbnet(ECNUACRush)
- **提交**:①代码(源码或可执行文件,非 Python/PyTorch 推荐栈须自带 Docker 镜像)②**Word 格式技术报告**(技术方案、实验结果、实验总结)

### 2.2 完成情况对照表

| 赛题要求 | 状态 | 证据 |
|---|---|---|
| PNG→components/pins/nets 端到端 JSON | ✅ 完成 | `agent.py` 全流程;冒烟测试(§8.3) |
| target 输出契约(decision_log + final_results/<case>_Pred.json) | ✅ 完成 | `agent.py:1388-1420`;`adapter.py:159-204` 校验 |
| 四项指标全覆盖(元件/引脚/超图/线段) | ✅ 代码覆盖;精度待官方复测 | 各阶段审计日志齐全 |
| 飞线 net(零长度边) | ✅ 完成 | `network_labels` 链;`run_network_labels.py:53-71` |
| 2 小时时限 | ⚠️ 未有官方机时证据(代码按 GPU 设计,显存≥24GB 自动 3 并发) | `agent.py:607-611` |
| Token 消耗 | ✅ **零 LLM 调用**(未使用 qwen3.7-plus,decision_log 记 token_usage 全 0) | 全代码 grep 无 LLM_* 引用 |
| Docker 镜像提交 | ✅ 完成(即本次 tar;入口 run.sh 已验证写入镜像 config) | OCI config `Entrypoint=['/submission/run.sh']` |
| **Word 技术报告** | ❌ **提交目录内未见**(答辩前必须确认/补齐) | 目录仅有 tar+PDF |
| 数据贡献加分(自行收集数据提交) | ❓ 无证据 | 目录无数据集提交材料 |
| 官方平台成绩(本版 v102) | ✅ **官方评测 74 分,第 47 名,全国前 61 晋级决赛** | 用户提供(2026-11);建议补平台截图,见 defense_supplement.md §一 |

---

## 3. 项目目录及文件说明

### 3.1 提交镜像 `/submission` 结构(实测)

| 路径 | 内容 | 规模 |
|---|---|---|
| `agent.py` | 批处理入口:逐图 run_case(子进程跑 S 阶段链)+ 15 个 finalize 修复阶段 + 失败回退 | 1,433 行(最终版) |
| `adapter.py` | 输出协议转换(target/sample/project)、六类别名映射、模板 bbox、validate_target | 361 行 |
| `run.sh` / `agent_system` | run.sh=真实入口;agent_system=官方占位脚本(未替换,见 §9 风险) | — |
| `component_export.py` | 电源元件导出(无位号电源/地出口) | 361 行 |
| `scripts/`(19 个) | S 阶段链式子进程:run_baseline → run_pin_ownership → run_wire_graph → … → run_observed_net_fields | 2,532 行 |
| `src/pcb_parser/`(97 个模块) | 全部算法:视觉模型、导线图、~50 个规则修复模块、vendor | 16,528 行 |
| `src/pcb_parser/vendor/pytorch_unet/` | milesial/Pytorch-UNet 上游积木(GPL-3.0,逐文件 sha256 溯源) | 5 文件 |
| `configs/`(41 个) | 各阶段阈值/预算/开关;全部为参数,**不含页面特例或答案** | — |
| `tests/`(42 个) | unittest 合成夹具测试(单测/回归/幂等/契约) | 626+ 函数 |
| `docs/`(30 篇) | 逐失败模式的开发文档,含每页 before/after TP/FP/FN 与 F1 数字 | — |
| `weights/` | detector_dataset300_v3_nano.pt(6.4MB)/unet_wire_pin_v1.pt(51.9MB)/junction_classifier_v1.pt(363KB)+ 溯源 JSON | — |
| `assets/ocr/` | PP-OCRv6 det/rec small + ch_ppocr cls 三个 ONNX(离线 OCR) | ~32MB |
| `data/` | BoxICLibrary.json(1000 型号 EasyEDA 符号库)、cpntLibrary(337)、STM32F103C8T6 LQFP48 引脚库 + ST 数据表 PDF | ~5MB |
| `overlay_check.py` | Docker 构建期门禁(12 个必需文件 + 几何表断言) | — |
| `package_manifest.json` | 逐文件 sha256 清单(249 项)+ 运行资产指纹 | — |

### 3.2 Docker 镜像分层(实测)

18 层:基础 Ubuntu24 + CUDA torch 2.5.1/cu121 + 依赖 → 3 层历史 COPY(README/agent_system)→ requirements 安装 → **`COPY . /submission/`(131MB 主代码层,L16)→ 构建校验层 → `COPY verified v102 code overlay`(51KB 最终覆盖层,L18)**。**最终运行代码 = L16 ∪ L18**(L18 覆盖 agent.py、export_resilience、export_semantics、score_export_recovery、wire_graph_artifacts 及 3 个新测试)。

### 3.3 与 GitHub `zc-cc1/vision` 的关系(重要)

截至审查时网络受限,完整克隆多次失败(仅一次 `ls-remote` 成功,远端 HEAD=main@342d8b65,另有 feature/mmroute2、feature/net-toolkit 两分支)。**未比对 GitHub 代码与提交镜像的逐文件差异**;无论 GitHub 内容如何,均不得视为最终提交版本,本报告全部以镜像层为事实来源。镜像 `package_manifest.json` 自述"training_data_included:false / docker_build_verified:false",且 manifest 列出的 4 个 Dockerfile 在主代码层中不存在——即**从镜像内容无法重建 Docker 构建**,GitHub 仓库也大概率不含 v102 完整源(需在联网后核对,列为待办)。

---

## 4. 完整技术架构(端到端流程)

### 4.1 总体数据流

```
输入 PNG
 ├─[感知] YOLO 检测(whole + SAHI 切片双路,WBF 融合,27 类逐类置信阈值)
 │        RapidOCR 全页 OCR(PP-OCRv6,离线 ONNX)
 │        U-Net 双头分割(wire 概率图 + pin 热图,512/384 瓦片,native 尺度)
 │        → pin 候选(热图峰检测+亚像素+NMS)→ pin 归属(代价最优指派)
 ├─[建图] 导线掩码→骨架→线段(RDP+PCA,断线证据合并)→ 物理网络图
 │        (T/十字结点保守拆分;红色结点盘检测;pin/红点处切分)
 ├─[逻辑网] 同名网络标签合并(飞线,零长度边);电源网合并(VCC/GND,不画物理长线)
 ├─[字段补全链 S14→S24] 元件引脚库映射(1000 型号)→ 定向补漏 OCR(6 视角)
 │        → 数值电源字幕 OCR → 引脚号条带 OCR(框外 30-50px)→ IC 密行重读
 │        → 接口标题恢复 → 工程编号(R/C/L 端口 1/2)→ 观测侧名/字面标题
 ├─[agent.py 修复链 finalize_* ×15] 电源导线恢复 → 引脚字段/几何/搬迁/配对/
 │        证据/循环/NC/垂直/行库/数字连接器修复 → 元件类型修复 → 结构连通性恢复
 │        (事务式规则引擎 + 15 结构模式 + 12 子阶段)→ 分数导出补触点 → 导出语义
 ├─[输出对齐] 类型别名(r/c/l/d/box/led/v)+ Name==key 约定 + 模板 bbox
 │        (8×20 等,按页缩放)+ IC 实框定位 + LED 模板框 + pin 尖端吸附(18px)
 └─[导出] sanitize_export 清洗 → validate_target 校验 → final_results/<case>_Pred.json
          + decision_log.json(全程哈希锁定、逐阶段审计日志、失败回退最近有效预测)
```

关键代码入口:
- 批处理:`agent.py:907 run_case` / `agent.py:1373 main`(线程池,0=自动:CUDA 且显存≥24GB→3 并发,否则 1)
- 感知链:`scripts/run_observed_net_fields.py:114` → `run_capacitor_tip_geometry` → `run_whole_networks` → `run_nearby_tip_fields` → `run_targeted_pin_ocr` → `run_network_completion` → `run_power_caption_ocr` → `run_component_pin_library` → `run_pin_text_v2` → `run_pin_ownership`(YOLO+U-Net+OCR 基座)
- 导出:`adapter.py:207 to_target`;校验 `adapter.py:159 validate_target`;清洗 `src/pcb_parser/export_resilience.py:18 sanitize_export`

### 4.2 感知层核心参数(实测)

- 检测:`detector_dataset300_v3_nano.pt`,imgsz=1536,27 类逐类置信阈值(如 ic 0.6044 / capacitor 0.5222 / ground 0.4121),SAHI 切片 1024²/overlap 0.25/IOS 0.45,融合 WBF IoU 0.55;59 例微平均 F1 **0.8830**(`configs/baseline_v3_adopted.json:101`)
- U-Net:`WirePinUNet(base_channels=32)`,双通道输出(wire/pin),瓦片 512/384 重叠概率均值,阈值 wire 0.35 / pin 0.6,峰检测二次亚像素修正,NMS 半径 0,候选上限 500
- OCR:RapidOCR 3.9.2(PP-OCRv6 det/rec small + cls),页级 limit_side_len 2048;pin 级多视角定向 OCR:缩放 2/3/4 × 旋转 0/90 六视角,采纳门槛 **置信≥0.98 且 ≥2 视角一致** + 截断字形拒收 + 全局唯一指派

### 4.3 修复层的共同设计模式(证据门控保守修复)

每个修复模块(agent.py 中 15 个 finalize_*)共享同一契约:环境开关可关、冻结工件哈希/尺寸/坐标系校验、本地 ONNX OCR 懒加载、**每页 OCR 调用硬预算**(12~160 次不等)、审计日志 `logs/<case>_*.json`、异常保留最近有效预测;修复策略全部遵循"**只补缺失、不改既有字段、网络成员只增不减、同一事务原子提交、重复运行幂等**"。采纳证据门槛统一为"多视角(≥2)@ 置信 0.98"或"多证据链交叉验证",并对多视角同模型明确声明"非统计独立证据"。

---

## 5. 核心算法与模块分析

### 5.1 模型清单与来源分类

| 模型 | 架构 | 来源 | 证据 |
|---|---|---|---|
| 元件检测器 | YOLO nano(ultralytics,35 类) | **自研训练**(由 v2 权重热启动;train 参数记录 data=detection_balanced_v2, epochs=120, imgsz=1536;初始基线为 HR-DONK 开源件仅作起点,非提交权重) | 权重 pkl 反汇编;weights/baseline_provenance.json |
| 导线/引脚分割 | WirePinUNet,base 32,共享 decoder 双输出头 | **开源积木 + 自研训练**(milesial Pytorch-UNet 的 DoubleConv/Down/Up 组件;标签=源注释几何栅格化;40 epochs,pos_weight wire=8/pin=20,选 epoch32) | vendor/provenance.json(commit 21d7850f);权重 pkl 元数据 |
| 结点分类器 | 6 通道(8/24 两个视角 64×64 RGB)→4 层 CNN | **自研训练**(1024 张合成渲染 + 人工复核裁片×12;30 epochs) | junction_classifier.py:14-39;权重 pkl |
| OCR | PP-OCRv6 det/rec small + cls | **直接调用开源**(rapidocr 封装,离线 ONNX) | assets/ocr/ 三模型 |
| 框架/工具 | ultralytics / SAHI / networkx / scipy / skimage | 直接调用 | requirements.txt |

### 5.2 值得答辩展示的核心算法细节

1. **双路检测融合**(run_baseline.py:80-97 + fusion.py):整图路保精度、切片路(SAHI)救小符号召回,WBF 用"置信加权坐标均值"而非 NMS 二选一;配套 27 类逐类置信阈值与逐类映射。
2. **双头 U-Net + 热图解码 pin**(unet_inference.py:45-100):局部极大+二次亚像素修正;wire/pin 共享 encoder;瓦片重叠区概率平均;**多尺度试验被数据否决**(unet_inference_v2.json 记录"pin 阈值改动被接受、wire 多尺度试验被拒"——有对照实验意识)。
3. **导线图重建**(wire_graph.py:333-492 + wire_segments.py:198-241):骨架→拐点→两点线段(近水平/垂直吸附);**断线合并需 learned mask 支持率≥0.8 且 gap 两侧不能各有一个 pin**;T 型/十字**保守显式不确定**(split_four_way),把"连错"风险转为"漏连";红点检测(面积/长宽比/填充率门控)。
4. **多视角证据 OCR 采纳体系**(pin_text_ocr/targeted_pin_ocr/12 个 pin 修复模块):0.98×2 视角门槛、截断拒收、与既有字段冲突拒收、只填缺失;每页预算上限;这是整个系统将 OCR 噪声挡在门外的核心机制。
5. **事务式修复引擎**(repair_engine.py):能力白名单(memberships/pins/networks 增删)、NC pin 禁止入网、一 pin 不两网、提案冲突全方否决、预算(64 触点/16 分支)。
6. **面向官方标注约定的输出对齐**(adapter.py + export_geometry.py):六类别名(8180 提交 145 元件验证);**模板 bbox(8×20/16×20/…)+ 按页缩放**(200 页 9186 组件统计:固定模板 150/163 达 20px 容差,按页缩放 159/163);Name==key 约定(覆盖率 r 1.000/c .999/…/box 仅 .242,故 box 保留观测型号);d/led 保留零件号例外;canonical 框约定把公开子集 bbox 精度 **0.905→1.000**。
7. **鲁棒导出**(export_resilience.py + agent.py remember()):逐阶段校验点冻结最近有效预测;失败时 recover_current_run 仅从**本图**工作目录抢救;sanitize 只删无效对象、"绝不推断文本/坐标/连接"。

### 5.3 输入数据与预处理

输入仅为 PNG(支持子目录,同名冲突报错)。预处理即感知推理(双三次放大→瓦片);**无外部数据增强/在线增广**;全程离线,无任何网络请求(YOLO_AUTOINSTALL=false,依赖完全随包)。

---

## 6. 项目创新点(按"自主实现/改造/直接调用"诚实分类)

**自主实现(真正的技术工作):**
1. **自训 3 个模型**(YOLO 元件检测、双头 U-Net、结点 CNN)及其训练数据体系(注释几何栅格化监督 + 人工复核掩码 + 合成结点渲染);提交包外有完整训练控制器痕迹(training_dependency.py 门禁)。
2. **~110 个证据门控修复模块**——本项目最大工作量所在:14 个引脚修复族(字段/几何/搬迁/配对/循环/NC/垂直/行库/数字连接器/条带/重读)、7 个字段恢复族、15 个结构模式、12 个几何子阶段、电源/地/接口/飞线族。每个模块对应一种实测失败模式(docs/ 有 30 篇逐页 bug 溯源文档)。
3. **多视角证据采纳协议**(0.98×2 视角 + 截断/冲突拒收 + 硬预算)——将 OCR 幻觉挡在门外,且明确声明同模型多视角"非独立证据",科学上诚实。
4. **事务式规则引擎 + 保守图语义**(repair_engine 能力白名单/冲突否决;junction 显式不确定而非猜测)。
5. **面向评分规则的输出对齐**(类型别名/Name 约定/模板框+页缩放/IC 实框/引脚尖端吸附),全部给出统计校准证据(9186 组件/200 页)。
6. **可追溯工程体系**:逐文件 sha256 manifest、权重指纹锁定、prediction_only 契约、official_* 显式置 null、每阶段审计日志、失败回退。

**开源改造:** milesial U-Net 积木改造为双头自训模型;SAHI/ultralytics 参数化集成并叠加 WBF;RapidOCR 封装为多视角证据引擎。
**直接调用:** PP-OCRv6 ONNX、networkx、scikit-image、scipy、OpenCV。
**明确不使用:** LLM API(qwen3.7-plus 提供但未用,token_usage=0)。

**答辩最值得重点展示的三项**(建议顺序):
1. 证据门控的保守修复体系(含事务引擎与预算制 OCR)——解决"高精度要求下模型噪声无处可藏"的核心难点;
2. U-Net 双头分割 + 导线图保守重建——解决赛题明确列出的难点四(T 型/十字/多分支/断线的连通性);
3. 官方标注约定的统计校准对齐——直接兑现 ComponentF1/bbox 20px/pin 5px 等评分条款。

---

## 7. 实验结果与数据证据

### 7.1 有出处的量化结果(全部可回查;均为 13 页公开集离线回放,**非官方成绩**)

| 实验 | 指标 | 前→后 | 出处 |
|---|---|---|---|
| 检测阈值选型 | 59 例微平均 F1 | **0.8830**(fused 模式胜出) | configs/baseline_v3_adopted.json:101 |
| 元件引脚库映射 | PinF1 | **0.031→0.549**(+47 引脚号) | scripts/run_engineering_networks.py:26-27 |
| 结点门控 | 连通 F1 | **0.188→0.612 @0.83 精度** | wire_graph_artifacts.py:16-18 |
| 部分成员导出 | 连通 | **0.259→~0.61** | wire_graph_artifacts.py:21-23 |
| 断线网放行 | 连通 / 总分 | 0.569→**0.576**(+0.7pp)/ **总分 +0.31 分**(两口径勿混) | wire_graph_artifacts.py:116-117 |
| 页 9 导线 | TP/FP/FN | 28/10/11→**38/0/1**,F1 0.7273→**0.9870** | docs/page9_wire_recovery.md:7 |
| 页 10 导线 | TP/FP/FN | 81/0/1→**82/0/0**,F1 →**1.0** | docs/page10_wire_recovery.md |
| 页 11 导线 | TP/FP/FN | 23/8/1→24/5/0→**24/0/0**,F1 0.8364→**1.0** | docs/page11_wire_recovery.md |
| 页 12 导线 | TP/FP/FN | 19/4/9→**28/0/0**,F1 →**1.0** | docs/page12_wire_recovery.md |
| 结点归并(页 13) | 导线 F1/总分 | 32/28/17→46/13/3,**58.72%→85.19%;82.871→92.656** | docs/network_junction_reconciliation.md |
| 端子拓扑(页 5) | 连通/总分 | **84.96%→100%;93.95→99.47** | docs/terminal_topology.md:17-18 |
| 端子身份(页 3) | 连通/总分 | **91.63%→100%;95.2462→99.2130** | docs/terminal_identity_recovery.md |
| 地符号拆分(页 11) | 元件 F1/总分 | **96.55%→100%;98.965→100** | docs/page11_ground_identity_recovery.md |
| 组件完成 v21 | ComponentF1 | **12/13 页 = 100%** | docs/component_completion_v21.txt |
| bbox 框约定(canonical) | 40 页公开子集 | component F1 **.8641→.8999**;**bbox 达标率 0.905→1.000**(官方 20px 规则) | export_geometry.py:1-30;agent.py:992 |
| bbox 按页缩放(另一机制) | 13 页 released | 20px 内模板元件 **150/163→159/163** | adapter.py:60-64 |
| 电源框缩放(反例) | case1 地匹配 | 17/17→14/17(故电源不缩放) | adapter.py:253-255 |

### 7.2 官方平台成绩历史(定稿)

- **最终提交版本:官方评测 74 分,本队第 47 名;全国前 61 支队伍晋级决赛,本队晋级**(2026-11 确认;建议补平台成绩页/晋级名单/提交记录截图——见 defense_supplement.md §一/§五)
- **提交 8422 = 28.92223 分**(README.md:50;为 v10x 一系列修复之前的旧版)
- **提交 8180**:145 个匹配元件验证六类别名(r/c/l/d/box/led)(README.md:26;agent.py:1340)
- 第 6 次评测曾因入口打包问题得 **0.0** 分(overlay_check.py:4-7 docstring)——此后加构建期门禁
- v101-score 本地开发分 **98.93**(L18 overlay package_manifest.json → export_resilience_update.development_score;official_score:null)
- **74 vs 98.93 的差距解释**:98.93 为 13 页公开集离线回放;官方 74 来自 30~50 案例(公开:黑盒=1:2,中等 40%/难 60%),黑盒难例泛化是主要差距;详见 defense_supplement.md §一.2

### 7.3 缺失的证据(2026-11 复核更新)

1. **RESULTS.md 与 export_semantics_optimization_20261106 报告**:OPTIMIZATION.md / EXPORT_SEMANTICS.md 各自声称"结果见配套文件",**镜像内不存在**——v102 最后两轮的逐项分数链条断在包外,只有 manifest 里孤立的 98.93。
2. **无任何可视化产物随包**(答辩图已由本审查生成,见 assets/ 与 defense_supplement.md §六)。
3. **运行日志/decision_log 样例**:已由冒烟测试生成(本机复现),可现场展示。
4. **逐案例耗时统计**:代码逐例记录 wall_seconds,本机 CPU 实测 33 分钟/例(1190×845);官方 A800 机时未实测,建议决赛前补一次。
5. **失败案例集**:docs 是修复成功记录;黑盒难例表现无本地数据,以官方 74 分的口径解释。
6. **成绩证明截图**(74 分/排名/晋级/提交 ID):⚠️ 待人工补充(defense_supplement.md §五)。

---

## 8. 项目运行及演示方式

### 8.1 运行环境(实测)

- 镜像内:Ubuntu 24.04 / Python 3.12 / torch 2.5.1+cu121 / requirements 钉死 ultralytics 8.3.253、sahi 0.12.8、rapidocr 3.9.2、onnxruntime 1.23.2、opencv 4.12、numpy 2.2.6、skimage 0.26、networkx 3.7、Pillow 12.3(torch 复用基础镜像)
- 设备:`--device auto`(默认):CUDA 矩阵真实探针(4×4 矩阵和=64)成功→cuda:0,失败→cpu 回退;显存≥24GB 自动 3 并发
- 预检:12 项运行资产逐 sha256 校验,不符即抛错(防权重被替换静默出错)

### 8.2 运行命令(README.md 实录)

```bash
docker run --rm --gpus all \
  -v <输入绝对路径>:/input:ro -v <输出绝对路径>:/output:rw \
  pcb-parser:v102 -input /input -output /output
# CPU:
#   -e PCB_DEVICE=cpu(镜像名之前);或已装依赖的 Python:
python agent.py -input <输入目录> -output <输出目录> --device cpu --workers 1
```
常用环境开关:`PCB_EXPORT_SEMANTICS=observed(默认)/legacy`、`PCB_SCORE_EXPORT_PROFILE=none/ic/passive/combined`(默认 none)、`PCB_*` 约 30 个(全部有合理默认)。

### 8.3 本机冒烟测试(2026-11-21 实测,L16 基础层 + L18 最终层合成 v102 运行态)

- 环境:Windows 11 + Git Bash;隔离 venv(Python 3.12.7)+ CPU torch 2.5.1 + requirements 全部钉死版本 + scipy(wire 模块依赖,requirements.txt 未列——镜像内已有,源码重装会缺)
- 输入:从赛题 PDF 第 11 页提取的官方示例图(1190×845,即赛题附录用来演示 JSON 格式的那张 RP2350 原理图)与第 22 页图七(963×719)
- **结果 A(管线可运行)**:两张图全部端到端跑完,无崩溃;12 项运行资产预检哈希校验通过;每图生成 20+ 份阶段审计 JSON(含前后 bbox、证据、哈希)
- **结果 B(主例)**:1190×845 示例图 → **89 components / 165 pins / 63 nets**,L16 基础层 1478 秒、**最终 v102 合成态 2000.5 秒**(纯 CPU);其中:
  - **`U3(type=box, Name='RP2350A')` bbox=[539,281,689,731]——赛题 PDF 官方示例标注为 [540,280,690,730],误差 1~2 像素(官方 20px 容差)**;
  - IC 型号 OCR 正确:W25Q128JVSIQ、TLV62569DBVT、RP2350A;引脚名正确:XIN(pin_21)、USB_DM(pin_51,官方示例同为 pin_51/USB_DM)、USB_DP、GPIO27_ADC1、GPIO23;R7 value='10kΩ' 正确;
  - 类型分布:gnd 29 / net_bidirection 17 / c 16 / v 11 / box 7 / r 7 / switch 2;
  - **v102 最终层 export_semantics(observed)实际生效 6 处**:SW1/SW2 的 Name 由错误回退值 'SW1'/'SW2' 修正为邻近文本堆栈识别的真实型号 **'PTS8153JK250SMTRLFS'**(旋转视角局部 OCR,置信 0.998/0.992;规则 designator_is_not_model + observed_switch_model);U8/U5 的 Name 由 'U8'/'U5' 置 null——可在答辩现场复现此前后对比。
- **结果 C(降级路径)**:963×719 低清图 → 检测为空 → 输出合法空 JSON(success=True,无崩溃)——验证了"失败降级而不崩溃"的鲁棒性承诺;原因判断:PDF 缩略图分辨率远低于训练/评测页(评测原图引脚号约 8px 字形在此图上<2px)
- 局限:CPU 推理耗时仅供量级参考;无法在本机评估精度(无 GT);L18 阶段(score_export_recovery/export_semantics)在该图上的增益需以 v102 合成运行输出为准(已在 output_v102 重跑验证)

### 8.4 现场演示建议(可稳定演示项)

1. 现场跑 1 张公开样例图(CPU/GPU)→ 展示 `decision_log.json`、`final_results/<case>_Pred.json`、逐阶段审计日志(pin_field_recovery.json 等);
2. 用 wire_graph_artifacts 的渲染能力现场生成"检测框+导线图+结点"调试图(代码自带,颜色编码:绿=学习型连通结点/紫=未决/橙=未知);
3. 展示 tests 的关键幂等/拒绝用例运行;
4. 展示 manifest 哈希链与 preflight 资产校验输出。

### 8.5 无法直接运行的风险点(不修改原提交,仅列问题)

1. **本机(审查环境)无 CUDA torch**:torch 2.5.1+cu121 需 NVIDIA 环境;A800(sm_80)官方机为兼容目标,本地 RTX5060(sm_120)按 README 需回退 CPU(代码已处理)。
2. **依赖冲突隐患**:基础镜像含 opencv-python-headless 4.10 与 requirements 的 opencv-python 4.12 同名 cv2;scipy 被 wire 模块依赖但 requirements 未列(镜像里已装,源码重装会缺)。
3. **从源码重建不可行**:manifest 列出的 4 个 Dockerfile 不在镜像层中;`docker_build_verified:false` 是**自述**,但 tar 本身即构建产物,入口已验证写入(见下)。
4. **入口语义**:最终镜像 ENTRYPOINT=`/submission/run.sh`(实测 OCI config),`docker run ... -input /output` 正常;`/submission/agent_system` 仍是官方占位脚本(exit 10)——若平台按 README 硬编码路径调用 `agent_system` 而非入口,将得 0 分。官方 8422 曾得 28.92 分,说明平台按入口运行的概率高,但**答辩前建议向组委会确认调用方式**。
5. `--workers 3` 并发时 OOM 无逐例隔离(仅按总显存选择并发数)。

---

## 9. 当前不足与风险(答辩备询)

| # | 不足/风险 | 事实依据 | 建议解释口径 |
|---|---|---|---|
| 1 | **官方成绩 74 vs 本地 98.93 的差距**:黑盒占 2/3 且难例占 60%,泛化为主要差距 | defense_supplement.md §一.2 | "官方 74 晋级决赛;本地 98.93 仅在 13 页公开集回放。差距来源是黑盒难例——我们以保守设计(漏连不连错)对抗无先验信息的难例,并把 JunctionCNN 等增强保留为可选。" |
| 2 | 黑盒泛化能力无证据 | 全部离线分数基于 13 公开页 | 承认;强调保守设计(漏连不连错)在难例上的抗劣化性 |
| 3 | 结点分类器默认关闭(学习型 junction 决策未进默认管线) | configs/wire_graph_v1.json:20 junction_model_available=false | "保守优先:交叉判错的代价高于漏连;分类器保留在 v2 pilot" |
| 4 | 两份最终文档引用的量化报告不在包内 | OPTIMIZATION.md:11;EXPORT_SEMANTICS.md:12 | 提交归档时遗漏;分数以 manifest 98.93 + docs 为准 |
| 5 | 无技术报告(Word)/无可视化/无样例输出 | 目录实测 | 决赛前补齐(本报告 §10 即为素材) |
| 6 | 读图顺序编号/框约定为公开参考校准的常量 | pin_numbering docstring 自认 | "开发期合法校准;运行时不读任何参考;泛化风险已通过按页缩放缓解(159/163)" |
| 7 | loose 导出策略为软 F1 调优 | wire_graph_artifacts.py:108-118 | "官方即 per-net soft F1;开关可回退 strict" |
| 8 | 数据卫生:.bak 残留、作者本机路径泄漏、乱码字符串 'm3铻轰笣' | manifest 差异;artifacts/settings.json;adapter.py:40 | 无功能影响;体现迭代真实性 |
| 9 | Token 5% 权重 | 未用 LLM,token=0 | 反而是得分项(最省区间=100);若被问"为何不用大模型"→"评分主权重 95% 在结构化精度,本地视觉+规则已覆盖;LLM 在 2h/50 例约束下无延迟与成本优势,且 Token 越少分越高" |

---

## 10. 决赛答辩 PPT 建议结构(10 页,定稿见 defense_slide_plan.md)

| 页 | 内容 | 素材来源 |
|---|---|---|
| 1 | 项目介绍:PNG→JSON 的纯本地解析系统;89/165/63 官方示例复现;官方 74 分/第 47 名/晋级 | fig6;supplement §一 |
| 2 | 赛题难点与评分拆解:四指标权重(0.3/0.25/0.35/0.1)+四大官方难点→四对策 | PDF §五/附录 |
| 3 | 总体技术架构:11 阶段流程(感知→建图→修复→对齐→导出) | arch_pipeline.png |
| 4 | 视觉感知:YOLO 双路+WBF、U-Net 双头(431 候选)、OCR;检测 F1 0.883 | fig1b;fig2c |
| 5 | 拓扑重建:保守结点语义、骨架+39 结点、238 物理网候选、飞线/电源网 | fig3c/fig3d |
| 6 | 核心创新:证据门控 × 事务式修复;页 13(58.72→85.19)与 SW1/SW2 两案例 | fig4;fig5 |
| 7 | 输出校准:类型别名/Name==key/模板框+页缩放(bbox .905→1.000 与 159/163 分口径);U3 差 1~2px | fig1b 局部 |
| 8 | 实验验证:统一数据表(官方 74 / 回放 98.93 / 冒烟复现三层证据)+ U-Net 训练曲线 | supplement §三;dev_repo |
| 9 | 工程实现:Docker/哈希链/审计日志/失败回退/626 测试/构建期门禁 | fig4b;manifest |
| 10 | 比赛结果与总结:74/47/晋级;旧版 28.92→74(+45);黑盒差距口径;展望 | supplement §一 |

备页:运行命令与现场演示、数据资产清单、dev_repo 12 张开发期图、风险与备询。

---

## 11. 评委可能提出的问题与回答依据

1. **"你们的方案和直接用 GPT-4V/Qwen-VL 有什么区别?"** → 评分 95% 权重是结构化精度(px 级 bbox/5px pin/超图),VLM 输出像素坐标不可靠且 Token 越少分越高;我们 token=0 拿满 Token 区间;全部模型本地、2h 内无排队风险。
2. **"官方 74 分怎么解释?为什么和本地 98.93 差这么多?"** → 如实:74 是官方 30~50 案例得分(公开:黑盒=1:2,难例 60%),98.93 只是 13 页公开集回放;差距主要来自黑盒难例泛化。附旧版 8422=28.92 → 最终 74 的提升证据(修复清单)。不辩解,给改进路径(JunctionCNN 转正/难例扩充/域适配)。
3. **"为什么不用大模型做端到端?"** → 同 1,另加:赛题难点是密集引脚/文字关联/拓扑,需要可控可审计的确定性管线;LLM 用于辅助标注/数据扩展是另一回事。
4. **"T 型十字交叉怎么处理?"** → U-Net 概率图→骨架→度≥4 结点;默认保守拆分(split_four_way),学习型分类器(双视角 6ch CNN,旋转一致+0.95)在 v2 pilot;红点检测辅助;结点归并阶段用"同线重合 60px/红点证据 95%"两条件恢复(页 13 案例)。
5. **"断线/漏检怎么办?"** → gap 合并需 learned mask ≥0.8 支持+共线±8°+pin 保护;graph 阶段 gap 候选记录不自动连;连通恢复引擎三规则+15 结构模式;放行断线网实测 +0.31。
6. **"pin 编号怎么读?OCR 错了怎么办?"** → 条带 OCR(引线外端 30-50px 字形,页级 OCR 看不到)+ 密行重读 + 6 视角定向 OCR;采纳门槛 0.98×2 视角;库映射(1000 型号)兜底;实测 PinF1 0.031→0.549。
7. **"飞线怎么识别?"** → 同名标签≥2 处且非 VCC/GND→逻辑合并,edges 零长度段;电源网合并另有四条门(单 tip 归属/≥3 视图同字形/无冲突名/无未决交叉),不画物理长线。
8. **"bbox 20px 怎么保证?"** → 检测框是 CAD cell 外扩,用 200 页统计的模板尺寸+按页缩放+IC 形态学实框+LED 模板;公开子集 0.905→1.000。
9. **"2 小时跑得完吗?"** → GPU 3 并发设计;单例 wall_seconds 全程记录;官方机未实测(答辩前建议实测一次)。
10. **"代码可信吗?"** → 逐文件 sha256 manifest、权重指纹锁定、preflight 校验、626 单测、构建期门禁、decision_log 全留痕。

---

## 12. 关键证据文件索引

| 主题 | 文件(镜像内路径) |
|---|---|
| 入口/编排 | /submission/agent.py(run_case@907, main@1373);run.sh |
| 检测融合 | scripts/run_baseline.py(fused_predictions@80);src/pcb_parser/fusion.py;configs/baseline_v3_adopted.json |
| U-Net | src/pcb_parser/unet_model.py;unet_inference.py;unet_checkpoint.py;configs/unet_inference_v2.json |
| 导线图 | src/pcb_parser/wire_graph.py;wire_segments.py;junction_classifier.py;configs/wire_graph_v1.json |
| 引脚归属 | scripts/run_pin_ownership.py;src/pcb_parser/pin_ownership.py;configs/pin_ownership_v1.json |
| 工程导出 | src/pcb_parser/engineering_network_export.py;whole_network_export.py;scripts/run_engineering_networks.py |
| 连通恢复 | src/pcb_parser/repair_engine.py;connectivity_recovery.py;structural_patterns.py;configs/connectivity_recovery_v1.json |
| 引脚修复族 | src/pcb_parser/pin_field_recovery.py 等 14 个;docs/pin_*.md |
| 输出对齐 | adapter.py;src/pcb_parser/export_geometry.py;configs/bbox_convention_v1.json |
| 导出鲁棒 | src/pcb_parser/export_resilience.py;export_semantics.py;score_export_recovery.py(L18 层) |
| 成绩/分数 | README.md:26,50;package_manifest.json(export_resilience_update.development_score);overlay_check.py:4-7 |
| 逐页实验 | docs/*.md(30 篇,含 TP/FP/FN 与 F1 数字) |
| 数据资产 | data/component_library/*;data/component_pin_library_v1/*;weights/*_provenance.json;vendor/pytorch_unet/provenance.json |
| 审查工作区 | E:\AI_Workplace\精英挑战赛\_audit_workspace\(解包层、冒烟测试、脚本) |

---

## 附:冒烟测试执行记录(2026-11-21 实测完成)

- 运行环境:Windows 11 / Git Bash;隔离 venv Python 3.12.7 + CPU torch 2.5.1 + requirements 钉死版本 + scipy
- 输入 A(963×719,PDF 第 22 页图七):全链路成功,输出合法空 JSON(检测为空,降级不崩溃),L16 140.7s / v102 173.5s
- 输入 B(1190×845,PDF 第 11 页官方示例图):L16 1478.2s → 89/165/63;v102 最终态 2000.5s → 89/165/63,export_semantics 6 处修正(SW1/SW2→PTS8153JK250SMTRLFS,U8/U5 Name→null)
- 每图生成 20+ 份阶段审计 JSON;产物位于 `_audit_workspace/smoke/output_v102/`
- 结论:**从 PNG 输入到最终 JSON 的完整链路在本机(无 Docker、CPU-only)可直接复现**,权重/资产全部随包自足;GPU(A800)下耗时预期大幅缩短(官方 2h 限制按 3 并发设计)
