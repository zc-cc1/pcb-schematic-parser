# OCR 阶段分析脚本

这些脚本**只读**缓存产物与公开 GT，不训练、不写预测、不改任何数据。全部用
`pcb_parser_competition\.venv-gpu\Scripts\python.exe` 运行。

| 脚本 | 作用 |
|---|---|
| `replay_binder.py` | 用冻结的 v5 检测框 + OCR 结果离线重放 `component_text.bind_components`（秒级，不需要 GPU/OCR） |
| `offline_eval.py` | 重放 + `adapter.to_target` + 官方口径打分（key/type/Name/value + 四角 20px；电源按几何匹配），支持 `--set` 覆盖 binding 开关做 A/B |
| `unbound_probe2.py` | 未绑定到元件的 GT 位号三级分类：标签在符号上 / 标签在别处 / 无文本 |
| `ocr_ceiling_probe.py` | GT 位号字符串在 OCR 输出里的可见率与绑定率（OCR 天花板） |
| `error_budget.py` | 命中 key 之后的字段级失败预算与无源器件 value 决策统计 |
| `value_probe.py` | 逐个差异元件的 value 证据（设计框、value token、最近的其他位号） |
| `reject_probe.py` | 未分配位号 token 的最近检测、前缀白名单判定与 `rejected_matches` 原因 |
| `token_probe.py` | tokenizer 视角：角色分布、融合串切分结果、model_candidate 列表 |
| `gt_convention_probe.py` | 200 页 GT 的标注约定（`Name == key` 比例、value 形态） |
| `decompose_v6.py` | v5 → v6 端到端逐元件比对，区分"字段翻转"与"检测/几何漂移" |
| `pin_text_budget.py` | 引脚口径预算：GT/pred pin 数、pinname 精确率、tip≤5px 比例、pin TP |
| `network_guard_budget.py` | 每例算出多少条多引脚网络、被哪道闸挡住、实际导出几条 |
| `label_type_geometry.py` | 被塌缩的标签类（pin/testpoint/net_input/crystal_*）的 GT 框尺寸可分性 |
| `ab_types.py` | 导出期类型词表 A/B（golden10 + big6，key 口径逐字段比较） |
| `gt_pin_labels.py` | GT 引脚编号形态（十进制 / BGA 球名 / 垃圾标签），判断哪一页能当引脚标尺 |
| `pin_withhold_reasons.py` | 引脚字段被扣住的原因计数（从各阶段日志汇总） |
| `pin_replay.py` | 用 `--keep-work` 的中间产物离线重放 pin_text v2，秒级 A/B 引脚文字 |
| `tip_offset.py` / `tip_calibrate.py` | tip 相对 GT 的偏差统计与全局平移网格搜索（先平移再重配，不偷看） |
| `tip_overlay.py` | 把 GT tip（红圈）与我方 tip（蓝=≤5px／橙=>5px）画在同一张放大图上 |

`scores/` 保存了四个官方口径打分结果（`v5/v6` × 是否 `--require-fields`），
供复核 `docs/ocr_stage_v1.md` 第 5 节的表格。

## 依赖的外部路径（本机）

| 变量 | 默认值 | 内容 |
|---|---|---|
| `BASE` | `D:\pcb5_submission` | 从第 5 次评测镜像里 cp 出来的部署代码（含 `adapter.py`） |
| `BATCH` | `D:\pcb_v5_batch` | v5 在 6 个公开页上的 `preds/`、`output/component_results/`、`output/logs/*_ocr_texts.json` |
| `GT_ROOT` | `..\..\赛题六公开数据集\赛题六公开数据集\200_train_cases` | 公开 200 页与 `*_target_named.json` |

缓存缺失时先跑一次 `agent.py`（见 `docs/ocr_stage_v1.md` 第 7 节）。

## 典型用法

```powershell
$py = 'C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition\.venv-gpu\Scripts\python.exe'

& $py analysis_ocr\offline_eval.py                 # 本轮默认配置
& $py analysis_ocr\offline_eval.py --legacy-name `
       --set leftover_reference_pass=false --set passive_value_allow_numeric=true   # v5 行为
& $py analysis_ocr\offline_eval.py --explain        # 逐个打印剩余失配原因
& $py analysis_ocr\unbound_probe2.py                # 位号未绑定的三级归因
```

## 已知限制

- 检测框来自 v5 冻结产物，所以 `0029` 这类被保守出口丢掉的元件无法在重放里恢复；
  重放分数是**同口径对比用的下界**，不是官方分。
- `adapter.to_target` 之后的 `ic_geometry` / `led_geometry` / `component_identity_recovery`
  不在重放链路里，因此解析数比真实管线少（0018 16 vs 20、0080 60 vs 64）。
- 公开 GT 的位号在部分页面与印刷体不同（见 `docs/ocr_stage_v1.md` 第 4 节），
  key 口径对这几页不可达。
