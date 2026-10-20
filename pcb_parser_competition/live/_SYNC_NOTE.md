# live/ — D 盘线上代码快照

- 来源：`D:\pcb5_submission`（第 5 次评测镜像里 cp 出来的部署副本，也是 10-20 各次评测实际运行的代码）
- 快照时间：2026-10-20 23:13（该目录当时最新文件时间）
- 拷贝范围：`agent.py`、`adapter.py`、`component_export.py`、`agent_system`、`run.sh`、
  `requirements.txt`、`Dockerfile*`、`overlay_check.py`、`package_manifest.json`、
  `README*.md`、`src/`、`scripts/`、`configs/`、`data/`、`assets/fonts/`、`docs/`、`tests/`
- 未包含（体积大或已被 .gitignore 排除）：`weights/`（.pt）、`assets/ocr/`（.onnx）、
  `__pycache__/`、`runs/`

## 与 pcb_parser_competition/ 其它目录的关系

- **本目录是权威代码**：10-20 新增的 `pin_numeric_connector.py`、`pin_row_library.py`、
  `pin_vertical_repair.py`、`pin_nc_recovery.py`、`pin_number_cycle.py`、`pin_evidence_repair.py`、
  `pin_pairing.py`、`pin_relocation.py`、`pin_geometry_recovery.py`、`pin_field_recovery.py`、
  `pin_position_consensus.py`、`power_wire_export.py` 及对应 configs 只在这里。
- `../src`、`../scripts`、`../configs` 是较早的仓库快照，两者有 32 个文件内容不同
  （26 个是 live 更新，6 个是仓库侧更新：`export_geometry.py`、`unet_model.py`、
  `run_baseline.py`、`configs/wire_graph_v2.json`、`configs/pin_text_v1.json`、
  `configs/bbox_convention_v1.json`），合并前请先 diff。
- 用仓库重建镜像前，必须用本目录的代码，否则会丢模块（第 6 次评测 0 分即此原因）。