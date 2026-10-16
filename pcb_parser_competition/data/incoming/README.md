# 待接收的正式资料

建议分为 `images/`、`golden/`、`evaluator/`、`symbol_library/`、`qa/` 和 `runtime/`。原件不改写，解压后保留来源和版本。

也可以直接使用其他已下载目录：把绝对路径登记到 `configs/project.json`，刷新清单后按源工程写入 manifest。无需为了接入而移动原文件。

manifest 已由 `scripts/audit_dataset300.py` 登记 300 条原始案例，包括图片、原始 JSON、画布、哈希、计数和警告。golden_path、difficulty、source_group 未确认时保持 null。原始标注可用于获准的离线训练/诊断，不传入实际推理。
