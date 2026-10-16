# S17：编号角色过滤与完整网络缺口排序

S18后续勘误：S17队列仍漏报“已因字段扣留、同时存在电源漏成员风险”的网络；305个仅字段未决应修正为296。C87对应网络也有上方电源缺tip风险，不能描述为仅缺pinname。S18已修复门槛记录，详见 `targeted_pin_ocr_stage_v1.md`。

2026-10-13，新增 `scripts/run_network_completion.ps1`，在S16之后进行被动器件编号角色过滤并生成完整出口复核队列。是独立试验，原S12/S16入口保持可用。

电阻、电容、电感中由OCR绑定且字面为常见封装代码的编号被撤回，原字符、置信度、框、观察视图和绑定证据保存到旁路审计；不猜测替代编号。IC/元件库编号不使用该规则。六处0402误绑定已目视核对：其中电感两处的文字来自附近电容的封装说明。同步清除这些编号的派生引用，原1477个tip、归属、物理线段和全部网络成员保留。

字段队列新增完整出口拦截原因、非字段拦截原因和共享任务。同一元件的身份缺口不再按每个pin重复计算；含电源provisional或漏成员风险的网络不会因交点/断线列表为空而优先进入“只需补字段”队列。修复导出接口对已完成字段的重复补全错误，原调用方式不变。

26页旧开发集共有378个多引脚网络候选：305个没有已记录非字段拦截但字段不全，73个有交点/断线/电源候选限制。“没有已记录拦截”不等于电气拓扑正确。严格部分出口仍213元件、116pin、0net，候选仍131pin、1net。连接对指标和有明确参考的条件字段计数与S16一致；六处编号撤回没有新增可计算的官方精确率。

最优先的两例：0002的C87.1缺pinname；0042的JTAG连接器缺规范身份。JTAG/JTAG-20PIN是图片标题，不能凭空补J1；数字作为被动器件名称仍保留在候选约定。不得删除未决成员凑完整hyperGraph。

另存0148/0149的离线参考差异：0149一个5V源组件没有进入电源参考分组；0148一个模型电源tip未与参考在归属/5px位置上匹配，需进一步区分模型位置错误与参考缺失。没有修改参考，不能据此认证合并正确或把所有差异直接宣称为已核验误合并。

```powershell
.\scripts\run_network_completion.ps1 -Image .\data\standardized\0018\0018.png -Device cpu
& .\.venv-gpu\Scripts\python.exe scripts/batch_network_completion.py
& .\.venv-gpu\Scripts\python.exe scripts/build_network_completion_report.py
& .\.venv-gpu\Scripts\python.exe scripts/run_regression_tests.py
```

配置：`configs/pin_number_validation_v1.json`。报告：`runs/s17_network_completion/report.md`。任务与排序：`completion_tasks.json`、`prioritized_networks.json`、`aggregate_blockers.json`。已视觉复核的六处封装文字与两个网络字段缺口另存 `data/reviews/network_completion_v1.json`，不进入推理或训练。

246项回归通过；0018原始PNG完整CPU入口约48.5秒，未传入模型预测缓存，字段/任务队列/网络成员及派生引用与回放一致，tip最大数值差0.0171px，小于0.05px CPU/GPU容差。验证证据见 `runtime_verification.json`。下一步优先处理编号/名称规范与准确型号引脚字段，再处理完整路径复核。
