# Dataset300 原始格式与赛事契约差异

资料根目录：`C:/Users/Pc/Desktop/新建文件夹/Dataset300`。此目录只读；项目保存引用和哈希，不移动或改写原件。

300例均包含同名图像与 `_CVJsonStd.json`，另有 EDS 和 `_spice.net`。已读取全部图像头和JSON；审计结果在 `data/dataset_audit.json`，逐例记录在 `data/manifest.jsonl`。

## 字段不能直接作为提交答案

| 内容 | CVJsonStd 原始表示 | 赛事要求 | 转换前的核验 |
|---|---|---|---|
| 元件身份 | C_1等内部ID | U1/R1等真实位号，重复者按官方规则加后缀 | 解析关联文字的designator；核验缺失与重复编号 |
| 元件类别 | Box、C、R、Gnd等34种原始类别 | type，例如ic/resistor/capacitor | 建映射和未知类型表，不把原始类别等同官方枚举 |
| 名称和值 | searchNameByOCR/VLM/Lib指向T_*，associatedTexts含details/content | Name/value字符串或允许的null | 解析文字引用；多值选择、错误OCR、空值规则要核验 |
| 引脚 | 根级P_*及组件局部pins；pinNumberByOCR等可能指向文字ID | 每器件下pin_<number>，pinname、point | 核验所属元件、实际文字、空编号、多编号及自动编号 |
| 框 | location和realBBox对象 | 四数bbox数组 | 区分outer/body；抽查图像及Golden |
| 坐标 | tip/base数组、location等 | 输出左下原点像素坐标 | 两个调试例的原始坐标在视觉上与左上图像坐标吻合；这不是正式出口H/H−1约定的验证 |
| 网络 | 顶层edges，connect、netlabel、isFlywire、keyPointPaths | 顶层nets，每网hyperGraph及两点edges | 按官方规则分组并线、归网及分段，不能重命名edges冒充nets |
| 状态 | active/deleted，某些对象无state | 仅有效输出 | 删除对象处理须记录；审计统计与正式转换策略分开 |

原始标注具有 OCR/VLM 候选字段，不能假定每个字段均是人工金标准。例如0001/C_1的searchNameByOCR指向场景标题T_1，而图中该IC显示RP2350B；该字段直接作为Name会产生明显错误。离线转换器必须保存歧义和人工核验状态。

## 资料同级的示例

`case_001_扩展示例.json` 和 `submission/case_001/case_001.json` 采用 components/pins/nets，但其 hyperGraph 包含未在自身组件/引脚段出现的引用。故当前仅作为字段例子；不能据目录名submission把它视为完整官方Golden。

## 当前转换结果与使用边界

300例已生成同尺寸PNG及 components/pins/nets 参考JSON，全部通过项目结构和引用复查。逐例映射/警告、复核队列及转换配置已保存，详见 `docs/conversion_report.md`。

92例非PNG图像已经另存PNG并检查解码像素相同。原始标注缺失的身份和字段仍有未确定项：4546个元件键使用UNRESOLVED占位；自动编号、未确认type、网络标签和分段策略均有记录。转换器没有把结构合法等同于语义正确。

1. 只在离线数据准备中读取标注；推理入口只读图像及允许的模型/库，不访问Golden或原始标注。
2. 非PNG内容已经解码后另存项目缓存中的PNG，源/目标哈希、宽高和像素检查均已保存；原件保持不变。
3. 原始类别、位号、引脚和文本的映射另存可追溯表；无法确定的项目不偷偷补答案。
4. 0002和0001已先转换并复核部分关键元件和编号；修正单独存入annotation_overrides.json。全部300例现已批处理，但仍需针对复核队列及正式转换器/Golden逐字段确认。
5. 转换器产生的结果只能标为本地参考标注，未经正式核验不声称是官方Golden。官方评测器未拿到前，任何替代分数明确标为proxy。
6. 先建立源工程分组，再决定训练/开发/保留集；已发现的三组精确重复必须同组，近似与不同EDA导出也需复核。
