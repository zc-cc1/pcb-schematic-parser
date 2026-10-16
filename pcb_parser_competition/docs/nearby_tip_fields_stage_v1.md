# S20：近邻tip的编号/名称补全

S19定位到0018/U3：编号8在`PIN_CAND_0102`，名称INT在`PIN_CAND_0113`。本轮在S18后加入独立预测阶段，以模型物理导线和实际OCR判断两点是否属于同一引线，只把已有名称补到已有编号的点。

```powershell
.\scripts\run_nearby_tip_fields.ps1 -Image .\data\standardized\0018\0018.png -Device cpu
```

冻结旧开发集回放及离线评价：

```powershell
& .\.venv-gpu\Scripts\python.exe scripts/batch_nearby_tip_fields.py
& .\.venv-gpu\Scripts\python.exe scripts/evaluate_nearby_tip_fields.py
& .\.venv-gpu\Scripts\python.exe scripts/run_regression_tests.py
```

`configs/nearby_tip_fields_v1.json`规定：两点属于同一已识别器件、方向一致，距离≤5px，各tip置信度≥0.85；共享实际物理网络、投影距离≤4px、唯一高置信物理导线段概率≥0.9，接线距离≤3px；无交点、断线或逻辑合并歧义。编号和名称均须是未分割的实际OCR字面，置信度≥0.98、同字框至少2个不同视图、原分配间隔≥5。第三个tip、重复编号、同字框冲突拼写或字框被其他pin占用时扣留。多次模型OCR视图并非独立观测。

方向从模型点到器件框边界的距离取得，两个最近边界相差不足2px时扣留；已有方向与几何冲突也扣留。原始PNG验证发现次级tip方向元数据可能为空，不能依赖缓存中的text_side；现在空值不影响几何判断，也不会改写原方向字段。

26页只接受0018/U3.8的1个名称补全：两点4.183px、最近第三点12.666px；同一W00039线段，编号8与名称INT各有3个不同OCR视图。原图可见字面和引线已另存助手视觉复核，未用于推理或训练。

| 出口 | S18 | S20 |
|---|---:|---:|
| 严格完整pin | 122 | 123 |
| 候选完整pin | 138 | 139 |
| 严格net | 0 | 0 |
| 候选net | 1 | 1 |

保留1477个原tip、原编号/名称/归属及全部网络成员。原次级tip仍只有INT、缺编号，所在网络继续扣留；没有删成员导出自环，也没有把编号复制给两个点。新名称证据注明原观察tip和共享线段，标记`new_OCR_observation=false`。

固定位置/归属一对一匹配的条件字段计数保持S18。严格已导出pin中，编号＋名称＋位置同时符合完整草稿字段的计数27→28；参考完整字段161条，草稿缺失不计算为已认证正确。全量名称＋位置代理因同时保留两个tip的INT而略降：该评价原本已用更近的次级tip匹配INT，本轮不能把同一参考重复计分。连接对代理完全不变。这是输出完整性改进，不是新增OCR、独立测试或官方PinF1提升。

26项新增回归覆盖共享物理线段、逻辑合并不代替物理证据、近邻竞争、同字框拼写冲突、方向缺元数据/角点歧义、编号不复制、已有字段不覆盖、输入不修改及重跑幂等；全套304项通过。0018原始PNG全流程54.2秒，补全字段、出口、OCR支持和网络成员与回放一致，最大tip数值偏差0.0171px，容差0.05px。证据见`runs/s20_nearby_tip_fields/runtime_verification.json`。

阶段输出在`runs/s20_nearby_tip_fields/`，逐页保留`field_link_audit.json`、完成字段、旧/新出口和严格/候选缺口队列，`visual_review_queue.json`索引原图对照。独立入口保留，默认入口未切换。下一步需要验证同一引脚多tip假设的完整出口表达方式，仍不能仅凭近距离删除模型点或缩减真实网络成员。
