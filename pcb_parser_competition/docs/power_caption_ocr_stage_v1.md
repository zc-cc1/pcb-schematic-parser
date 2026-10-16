# S16：电源文字局部 OCR

2026-10-13，在冻结 S14 图像预测上增加电源文字复读，复用 S15 逻辑组网与出口门槛。入口为 `scripts/run_power_caption_ocr.ps1`，属于独立试验，原默认入口保留。

对缺文字证据、但已有单一模型归属且唯一接线的电源符号，裁剪原 PNG 周围区域，执行 2/3/4 倍和 0/90 度带定位框的 RapidOCR。实际文字框还原到原图坐标；裁剪边界文字被拒绝。保留原始文字及所有局部冲突，不按词典修字、不把整块识别结果冒充定位框。

文字必须通过原电源名称规则和符号空间归属规则。邻近两个符号即便已在同一物理网络，也保持归属歧义；元件/引脚文字不挪作电源名。置信度≥0.98、至少3种同位置视图、完整字面名、交点/断线/漏电源成员等组网门槛继续使用。多个视图来自同一模型，并非独立证据。

26页旧开发集共复读134个ROI，满足证据门槛的电源节点28→56，候选组3→7，减少网络分组9→26。在同一冻结电源草稿参考上，条件连接对F1 56.05%→71.84%，全量配对代理25.41%→33.97%。条件FP 3→12，全量代理FP 599→732，尚有误合并风险，保留独立试验。这不是官方NetF1或独立测试。

严格部分出口仍为213元件、116个完整pin、0个net；候选仍131个pin、1个net。所有原tip、字段、归属及物理线段保留，未遗漏或删减网络成员。新增电源组仍为内部provisional候选，不能作为完整赛事网络导出。

```powershell
# 在项目根目录，从 PNG 运行整个冻结图像模型流程
.\scripts\run_power_caption_ocr.ps1 -Image .\data\standardized\0018\0018.png -Device cpu

# 固定开发集回放，仅提供已有且经哈希校验的模型预测
& .\.venv-gpu\Scripts\python.exe scripts/batch_power_caption_ocr.py
# 离线评价使用原 S15 草稿参考快照
& .\.venv-gpu\Scripts\python.exe scripts/evaluate_power_caption_ocr.py
& .\.venv-gpu\Scripts\python.exe scripts/run_regression_tests.py
```

配置：`configs/power_caption_ocr_v1.json`。主要模块：`src/pcb_parser/power_caption_ocr.py`。主要产物：`runs/s16_power_caption_ocr/` 的 `report.md`、`evaluation.json`、`visual_review_queue.json` 和逐例 `recovered_texts.json`、`power_caption_recovery.json`、旧/新部分出口。

新增12项测试覆盖旋转/尺度坐标还原、边界截断、低置信度、冲突拼写、同网邻近符号歧义、文字角色屏蔽、预算和输入尺寸。现有回归工具缺少tests目录导入路径，已补齐；全套229项回归通过。原始PNG复测证据由 `scripts/verify_power_caption_runtime.py` 写入 `runtime_verification.json`。

0018原始PNG完整入口CPU运行45.6秒，未传入预测缓存；字段/候选统计与网络成员匹配开发回放，tip最大数值差0.0171px，小于0.05px的CPU/GPU数值容差。新28处文字裁剪已由助手目视核对字面名与符号角色，记录在 `data/reviews/power_caption_ocr_v1.json`，不进入预测或训练，也不认证完整远端路径。

后续重点仍是缺归属tip、电源规范身份与pin字段，以及0148/0149等新增候选的完整路径复核。局部文字视觉复核不等于认证远端电气网络。
