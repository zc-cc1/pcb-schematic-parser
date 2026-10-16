# S18：局部引脚字段 OCR

新增独立入口 `scripts/run_targeted_pin_ocr.ps1`。先执行冻结的S17图像流程，再按字段缺口与已有未分配OCR候选选择ROI，做2/3/4倍、0/90度带定位框复读。配置 `configs/targeted_pin_ocr_v1.json`，每页最多12个ROI。

仅填空编号/名称，不改已有字段，不改tip/归属/物理网络成员，不推断JTAG的规范身份，不根据空间顺序生成编号。新字段至少两个同位置强视图、置信度≥0.98、分配间隔≥5，并且必须包含该引脚局部ROI的实际观察。既有字段的字框被占用保护；同元件编号冲突、封装/非法被动编号、同位置拼写冲突和裁剪边界文字继续扣留。观察视图来自同一模型，不当作独立证据。

新增片段门槛：0105局部旋转OCR曾将IN截读为N；同位置存在强完整字面IN时，新增N被扣留，不用猜测修字补IN。不同位置的合法单字符F不受该门槛影响。

26页旧开发集共127个ROI，补回9个可见编号、2个可见名称。严格完整pin116→122，候选131→138；严格net仍0、候选仍1。现有1477个tip和所有既有字段、归属、网络成员保持，派生引脚引用同步增加。默认入口未切换。

同一草稿参考中，编号正确/已输出/可评价为90/119/202→92/121/202，名称186/199/220→187/200/220。新增11字段中3个有位置/归属及字段参考且一致；3个参考字段缺失，5个无法在归属/5px位置上匹配参考，仍需核查。局部图片文字已由助手视觉复核，但不能据此认证tip位置或完整网络，也不是官方PinF1/独立测试。

同时修复电源出口门槛漏报：即使网络已经因为字段缺失被扣留，也必须记录漏电源成员风险，不仅记录实际从JSON中移除的网络。两个列表分别保留“全部受门槛阻断”与“实际移除”。C87所在上方网络有缺电源tip风险，候选已经具有pinname=1；此前将其描述为仅缺名称不完整。按修正门槛，S17与S18仅字段未决网络均为296，S17旧报告的305偏高。

```powershell
.\scripts\run_targeted_pin_ocr.ps1 -Image .\data\standardized\0018\0018.png -Device cpu
& .\.venv-gpu\Scripts\python.exe scripts/batch_targeted_pin_ocr.py
& .\.venv-gpu\Scripts\python.exe scripts/evaluate_targeted_pin_ocr.py
& .\.venv-gpu\Scripts\python.exe scripts/run_regression_tests.py
```

产物在 `runs/s18_targeted_pin_ocr/`：`report.md`、`evaluation.json`、`visual_review_queue.json`，以及逐页局部文字、恢复/拒绝证据、原/新部分出口、复核队列。

266项回归通过。0018原始PNG完整CPU入口在并行批量回放时约72.2秒；未传入模型预测缓存，新增字段、出口、网络成员与开发回放一致，tip最大数值差0.0171px。运行记录见 `runtime_verification.json`。

下一步应核查5个未匹配新增字段的模型tip/源参考位置，处理真正缺少电源tip的网络，再补规范元件身份与没有可见文字的引脚规则。不能用删成员、猜J1或将候选名称约定提升为严格规则来制造完整网络。
