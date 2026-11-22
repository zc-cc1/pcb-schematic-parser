# EDA 决赛答辩 · 评委问答手册(官方模板版,12 问)

> 配套《EDA精英挑战赛_决赛答辩_官方模板版.pptx》。回答时先给一句话结论,再按需展开;所有数字口径必须带数据范围(官方 30~50 例 / 13 页公开集 / 40 页公开子集 / 200 页训练页 / 官方示例页)。

---

## Q1 为什么不用赛会提供的 qwen3.7-plus 大模型端到端解析?

**一句话**:95% 的权重在像素级结构精度上,大模型输出像素坐标不可靠且不可审计;Token 越少分越高,我们零调用拿最优赋分区。
展开:评分主体是 bbox 20px / pin 5px / 超图 soft F1 这类硬指标;全本地推理在 2 小时约束下无排队风险;decision_log 记录 token_usage=0。
依据:赛题评分细则;`agent.py`;source_audit.md §2.2。

## Q2 官方 74 分与本地 98.93 差距怎么解释?

**一句话**:口径不同——74 来自含 2/3 黑盒(难例 60%)的官方评测;98.93 只是 13 页公开集的本地回放(开发评估分)。
展开:黑盒页绘制风格/分辨率/密度无先验信息,泛化是主要差距;对策是保守设计(漏连不连错)+ 可选增强;旧版 8422=28.92 → 74 的提升有逐项修复清单。
依据:defense_supplement.md §一.2;README.md:50;L18 manifest。

## Q3 T 型、十字交叉节点怎么处理?为什么不用你们训练的分类器?

**一句话**:默认保守拆分、显式不确定(把"连错"转为"漏连"),配合红点检测;JunctionCNN 已训练但在默认管线关闭,作为可选增强。
展开:学习型决策仅对度=4 结点、要求 4 旋转一致@0.95 才接受;默认选择保守语义是面向 soft per-net F1 评分的取舍。
依据:configs/wire_graph_v1.json:20 / wire_graph_v2.json;junction_classifier.py;defense_supplement.md §四.1。

## Q4 断线、漏检怎么恢复?会不会把不该连的连上?

**一句话**:图阶段只记录 gap 不自动连接;修复阶段凭硬证据恢复——60px 同线重合(±3°、垂距≤2px)或红点+95% 墨迹支持;实测总分 +0.31 分。
展开:gap 合并还要求 learned mask 支持率≥0.8、gap 两侧不能各有一个 pin;放行断线网本身另计(连通 0.569→0.576)。
依据:wire_segments.py:102-174;wire_graph_artifacts.py:116-117;docs/network_junction_reconciliation.md。

## Q5 pin 编号太小、页级 OCR 读不到怎么办?

**一句话**:三重机制——条带 OCR(引线外端 30-50px 的 8px 字形,3/4/6× 双向)、密行重读(只重读该字形自身框)、6 视角定向 OCR(2/3/4× × 0/90°),加 1000 型号引脚库兜底;PinF1 0.031→0.549。
展开:采纳门槛 ≥2 视角@0.98;截断字形/冲突拼写拒收;每页硬预算。
依据:run_engineering_networks.py:26-27;pin_number_strip/pin_number_reread docstrings;targeted_pin_ocr_v1.json。

## Q6 飞线(多个同名测试点)怎么识别和输出?

**一句话**:同页同名非电源标签 ≥2 处逻辑合并为一个 net,edges 输出零长度段(与官方示例一致);VCC/GND 走电源网四道门,不画物理长线。
展开:电源网合并门:单 tip 归属 / ≥3 独立视图同字形 / 无冲突拼写 / 无未决交叉。
依据:network_labels.py;power_networks.py:95-168;赛题附录一 §8。

## Q7 bbox 20px 怎么保证?两套口径是什么关系?

**一句话**:canonical 框约定(40 页公开子集,达标率 0.905→1.000)与按页缩放(13 页 released,150/163→159/163)是**两套独立实验、不同机制、同时生效但不可相乘**;IC/LED 走各自实框/模板逻辑。
依据:export_geometry.py:1-30;agent.py:988-994;adapter.py:28-64。

## Q8 Name 为什么有时是位号、有时是型号?

**一句话**:官方约定是"Name==key,box 类除外"——200 页 9186 组件统计 box 类只有 24.2% 等于位号,所以 box 保留芯片印刷型号;d/led 有近失读修正规则。
依据:adapter.py:28-55; EXPORT_SEMANTICS.md。

## Q9 两小时跑得完吗?现场能演示吗?

**一句话**:按 GPU 3 并发设计(显存≥24GB 自动);CPU 单例实测 2000.5 秒(官方示例页);官方机时未实测——如实回答,可申请赛前复测。演示优先用已归档 decision_log/Pred JSON 或 30 秒录屏。
依据:agent.py:557-611;defense_supplement.md §五。

## Q10 结果的可信度和可复现性怎么保证?

**一句话**:逐文件 SHA256 manifest(249 项)+ 3 权重指纹锁定 + 预检 12 项资产 + 626 单测 + 构建期断言 + 每图 20+ 审计日志 + 失败回退最近有效预测。
展开:权重被替换会显式失败;第 6 次评测 0 分教训促成 overlay_check.py 构建期门禁。
依据:package_manifest.json;export_resilience.py;overlay_check.py。

## Q11 训练数据从哪来?有没有用到评测标注?

**一句话**:检测器用本地 detection_balanced_v2;U-Net 用源注释几何栅格化 + 人工复核掩码;JunctionCNN 用 1024 合成渲染 + 人工复核裁片;运行时输入合同是"PNG 与冻结模型/OCR 输出,不读任何标注/CAD/Golden"。
依据:checkpoint 元数据;training_dependency.py;run_observed_net_fields.py input_contract。

## Q12 与其他队/开源方案的核心差异?

**一句话**:别人堆模型,我们堆"纪律"——模型只产生候选,结论由证据体系给出;再加面向官方标注约定的统计校准;全链可审计。
依据:source_audit.md §6;docs/*.md(30 篇逐页溯源)。

---

## 备询快答

- **"下一步?"** 结点分类器转正、黑盒难例数据扩充、跨域泛化(Part IV 页)。
- **"数据贡献 10 分?"** 提交材料中无自采数据提交记录——如实回答(待确认清单)。
- **"为什么封面是 EDA2606K5?"** 由本地提交目录名推断,报名记录为准(适配说明中已标注)。
- **"Word 技术报告?"** 由 defense_supplement.md §三 + assets/ 图组生成,答辩必带。
- **"页 13 案例的整页前后对比图?"** 原始页 13 PNG 未随包,当前用开发仓库真实裁剪 + 前后数字面板,来源已标注在图内。

---

## 叙事统一版补充问答(story-polish)

### Q13 你们的核心主张用一句话说是什么?

**一句话**:From Pixels to Connectivity —— We Don't Just Detect Symbols. We Reconstruct Electrical Connectivity(我们不是在堆视觉模型,而是在重建电气连接关系)。
展开:三段主线 Perceive(识别)→ Connect(建几何与拓扑)→ Verify(证据修复与校准);四个官方难点的对策全部挂在这条线上。
依据:PPT P3 横幅与三段线索行;`defense_slide_plan` 叙事统一节。

### Q14 "Repair Without Guessing" 的四步是什么?为什么强调"不猜"?

**一句话**:Detect(多视角定向 OCR,硬预算)→ Verify(≥2 视角@0.98 + 几何拓扑约束)→ Commit|Reject(原子事务、冲突全方否决)→ Rollback(失败回退最近有效预测)。
展开:"不猜"指宁可漏也不编造——每个字段采纳都要证据、每次改动可审计可回滚;OCR 多视角被明确声明"非统计独立证据"。
依据:repair_engine.py 能力白名单;agent.py remember();PPT P7 四步流程。

### Q15 页 13 的 58.72%→85.19% 是官方整体指标吗?

**一句话**:不是——这是**单页开发集案例**(冻结缓存回放、单页消融),官方整体成绩是 74 分。
展开:其余 12 页逐字节不变证明修复是局部的;官方分项口径我们看不到。
依据:PPT P7 斜体声明;docs/network_junction_reconciliation.md。

### Q16 主页的四层架构和备份页的 11 阶段是什么关系?

**一句话**:四层是叙事压缩(Perceive/Connect/Verify 的工程映射),11 阶段是完整实现;两级编排(S 阶段链 + 15 个 finalize)细节在备份页 A。
依据:PPT P4 四层图;备份 B1;source_audit.md §4。

### Q17 本地回放 98.93 为什么不在主页?

**一句话**:与官方 74 分口径不同(13 页公开集、开发评估),放主页会误导评委;已移至备份页 D 并标注"development only"。
依据:defense_supplement.md §一.2;PPT P9 底部指引行。

### Q18 页 13 展示用的是哪一套数据?为什么不用早期快照?

**一句话**:用的是提交镜像内 `docs/network_junction_reconciliation.md` 记录的**同一次结点归并实验**——修复前 32/28/17(F1 58.72%、连通 67.14%、总分 82.871)→ 修复后 46/13/3(85.19%、87.53%、92.656)。
展开:开发仓库的 `page13_status_20261020/status.json`(wire 13/22/36、F1 30.95%、总分 65.58)是**更早管线状态**的快照(pin_row_library 部署版),不是本次实验的"修复前";两套数据不得混用,展示图已按同一次实验统一。
依据:提交镜像 docs/network_junction_reconciliation.md 第 18-19 行;status.json 原件。
