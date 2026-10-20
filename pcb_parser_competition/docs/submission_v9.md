# 提交包 v9：相对上次提交改了什么

日期：2026-10-18
镜像：`pcb-parser:v9`（由 `D:\pcb5_submission` 构建）
本地口径（官方协议复现，非平台分）：**0.5345 = 53.45 分**，上次提交（第 7 次）平台 33.52 分。

## 1. 相对 v5 镜像的改动

| 改动 | 文件 | 默认 | 依据（13 例 open_GT，平台协议口径） |
|---|---|---|---|
| IC 引脚 tip 推到引线末端 | `agent.py` `pin_lead_snap()` | 18px | 参考点在 body 边缘 ±20px，模型质心只到 ±12–14px；pin F1 0.031 → 0.403 |
| 元件库补 IC 引脚号/名 | `src/pcb_parser/box_ic_library.py`、`scripts/run_engineering_networks.py` | 开 | 34 个 IC 里认出 17 个，+47 个编号；配合上一条 pin F1 → 0.549 |
| 非 IC 引脚补容器编号 | `scripts/run_engineering_networks.py` | 开 | 连通身份里普通元件只看 key，编号只是容器；让网络能出口 |
| 出口按连通身份放行 | `src/pcb_parser/wire_graph_artifacts.py` | 开 | 名字不再作为网络出口条件（IC 除外） |
| 路口闸门打开 | `wire_graph_artifacts.py` | 开 | 连通性 0.188 → 0.259，fp 仅 1 |
| 部分成员照发 | `wire_graph_artifacts.py`、`whole_network_export.py`、`run_engineering_networks.py` | 开 | 连通性 0.259 → 0.463，总分 45.22 → 53.45 |
| 被动元件框按检测框比例 | `agent.py` `PCB_BBOX_SCALE_PASSIVE` | **关** | 离线 +0.02，端到端 −0.011，已回退 |

全部可用环境变量关闭：`PCB_PIN_LEAD_PX=0`、`PCB_BOX_LIBRARY_CONFIG=off`、
`PCB_PASSIVE_NUMBERING=off`、`PCB_NET_JUNCTION_GATE=closed`、`PCB_NET_PARTIAL_MEMBERS=off`。

## 2. 自带资源（镜像内，不再依赖宿主机路径）

| 路径 | 大小 | 用途 |
|---|---:|---|
| `data/component_library/BoxICLibrary.json` | 3.1 MB | 官方补发的元件库（1000 个器件） |
| `data/component_library/cpntLibrary` | 1.7 MB | 公开数据集自带的那份（337 个器件） |
| `configs/box_ic_library_v1.json` | — | 指向上面两份，相对路径 |

以前这两份库放在工作区，容器里根本不存在；这次放进镜像，`library_from_config` 用相对路径解析
（cwd 或 `configs/` 的父目录），容器 `WORKDIR /submission` 下可直接命中。

## 3. 验证

1. **宿主机**（`D:\pcb5_submission`，默认开关，无环境变量）：13 例全跑，0 失败，480 s；
   评分 0.5345。
2. **单例冒烟**（宿主机）：case 3 → 0.752。
3. **容器内冒烟**（`docker run --rm -v ...:/in -v ...:/out pcb-parser:v9`）：case 3 → **0.759**，
   CPU 107 s，元件 25 / 引脚 38 / 网络 6。CPU 与 GPU 的差异在检测浮点，量级正常。
4. 镜像构建成功：`pcb-parser:v9`，`COPY . /submission/` 后 import 自检通过。

## 3.1 产物

| 文件 | 大小 | 说明 |
|---|---:|---|
| `D:\pcb_submission_out\pcb-parser-v9.zip` | 3.55 GB | 提交包，内含 `pcb-parser-v9.tar`（与 v5 同结构） |
| `D:\pcb_submission_out\pcb-parser-v9.tar` | 3.32 GB | 镜像 tar，单独保留便于直接 `docker load` |
| `D:\pcb_submission_out\pcb-parser-v9.zip.sha256` | — | `7EBB2737014995C0877B25F0A0293CD51B6C64418ED23648811F95831919BECF` |

构建/导出/打包三步都实际执行过：`docker build` → `docker save` → `tar -a` 生成 zip
（`Compress-Archive` 对 >2 GB 的条目会报 "stream too long"，改用系统自带 bsdtar）。
上传这一步需要你用自己的账号在平台完成，我这里没有提交入口。

## 4. 已知风险

- **部分成员照发**是策略放宽：导出网络可能缺成员，靠软 F1 拿部分分。若平台口径与本地复现不同，
  这一项可能反噬；一条环境变量即可退回。
- 元件/连通仍以 13 例为唯一标尺；`open_GT` 是否就是平台评测集尚未由官方确认（本地 component 与
  平台第 7 次相差 0.004，connectivity 相差 0.005，很接近）。
- 案例 12/13 仍低（0.067/0.096）：12 缺被动件 value 的 OCR，13 缺被动框约定。
