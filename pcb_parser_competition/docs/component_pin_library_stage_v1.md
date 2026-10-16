# 准确型号的公共引脚库试验

新增入口：`scripts/run_component_pin_library.ps1`。它执行当前 YOLO/SAHI、U-Net、RapidOCR 和 S11 文字绑定，再对已有准确型号应用公共厂家引脚表，最后运行导线、网络标签和 S12 字段出口。原 S12 默认入口保持可用，本阶段是独立试验。

目前只支持 **STM32F103C8T6 / LQFP48**。该完整型号至少两种同位置强 OCR 视图支持，才应用厂家引脚映射。不能仅凭 STM32、F103 或外形选择封装。后续增加型号时，需记录厂家链接、版本、封装、公开 PDF SHA256 和逐脚复核。

名称来自图中的完整文字。比如 `PA5/AN5/SPI1_SCK` 是名称，15 是厂家映射补出的编号。编号状态为 `public_library_from_observed_pinname`，证据说明 `number_observed_in_image=false`，不能归入编号 OCR 识别数。`PC13/TEMPER_RTC` 保留原图字面拼写，不用手册字符串覆盖。

完整名称至少两种强 OCR 视图支持，完整文字框须在元件框内并与 U-Net 引脚行对齐；编号一对一分配。重复点假设不取得第二份编号；已占用编号、合法编号与名称冲突、同位置文字拼写冲突均保留未决。准确 LQFP48 型号上的非法编号读数（如 PB5）移入审计，原证据仍保留。没有通用左右顺序编号规则，也不处理未知无文字 R/C 编号。

```powershell
.\scripts\run_component_pin_library.ps1 -Image .\data\standardized\0018\0018.png -Device cpu
```

PNG 入口不读取数据集源注释、转换参考 JSON 或 Golden。公开元件表作为静态配置读取；所有原引脚坐标/归属原样保留。缓存模式须同时提供 `--pin-folder`、`--wire-cache`、`--wire-metadata`、`--baseline-network` 并通过图像、配置、模型及输入 SHA 核验。

主要输出：

- `library_binding_audit.json`：每个补全/拦截的旧值、新值、公开依据与实际文字证据。
- `pin_text/pin_text_result.json`：库校验后的预测，区分可用编号与图片观察编号。
- `output_rules/partial_submission.json`：严格部分出口。
- `output_rules/submission_candidate.json`：沿用 S12 被动器件候选名称规则，仍有假设，不能当成完整赛事答案。

26 页旧开发回放只命中0018的一颗U1。补19个编号、4个名称和1个名称片段；严格字段齐全pin从93到116，候选从108到131，严格net仍0、候选仍1。新增19个编号在原位置/归属草稿参考上都匹配，但参考编号均缺失，所以没有新增编号的草稿准确率。助手对图片文字的视觉复核单独留在 `data/reviews/component_pin_library_v1.json`；它不进入预测或训练，不充当独立测试或官方参考。

完整结果见 `runs/s14_component_library/report.md`、`evaluation.json` 和运行验证文件。当前整个比赛流程仍未完成：官方格式含义、R/C编号约定、电源成员、交点/断线和完整网络出口仍待完善。
