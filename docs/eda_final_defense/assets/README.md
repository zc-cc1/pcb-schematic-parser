# 答辩可视化素材说明(assets/ 索引)

每张图记录:来源 / 对应实验 / 代码模块 / 展示内容 / 指标。自产图的生成脚本在 `_audit_workspace/scripts/gen_asset_*.py`,可复现;所有数字见 `defense_supplement.md` §三统一数据表。

## 一、系统输出与检测(冒烟复现,官方示例页 1190×845)

| 文件 | 来源 | 代码模块 | 技术内容 | 指标 |
|---|---|---|---|---|
| `fig1_detection_overlay.png` | 本机端到端运行 `output_v102/final_results/rp2350_page11_Pred.json` 叠加渲染 | `agent.py`→`adapter.to_target` | 全部 89 个导出元件(含电源符号 UNRESOLVED_DET_* 键、类型分布 gnd29/v11/c16…) | 89/165/63;类型分布真实 |
| `fig1b_detection_clean.png` | 同上(仅 32 个绘图类元件) | 同上 | 检测框+位号+识别出的型号(RP2350A/W25Q128JVSIQ/TLV62569DBVT/PTS8153…/10kΩ) | U3 bbox 与官方示例标注差 1~2px |
| `fig6_final_output.png` | 同上 | `validate_target` 后的最终 JSON 回绘 | 上=元件标注,下=63 网络彩色编码边 | 最终交付形态 |

## 二、U-Net 双头分割(真实管线产物)

| 文件 | 来源 | 代码模块 | 技术内容 | 指标 |
|---|---|---|---|---|
| `fig2c_unet_real_artifacts.png` ⭐ | keep-work 运行保存的 `probabilities.npz`(真实管线概率输出) | `unet_inference.predict_configured`/`decode_pins` | 三联:输入图 / wire 概率图(0.35 阈值)/ pin 热图(**431 候选=287 已归属+115 歧义+29 无主**) | 候选统计出自 `ownership/summary.json` |
| `fig2_unet_segmentation.png` | 本机直接前向(同一权重) | `unet_checkpoint.load_wire_pin_checkpoint` | 三联带标注版(演示用) | — |
| `fig2b_unet_operating.png` | 同上 | 同上 | 工作阈值叠加单图 | wire px 22,078 / pin px 37,167 |

## 三、拓扑重建(真实管线产物)

| 文件 | 来源 | 代码模块 | 技术内容 | 指标 |
|---|---|---|---|---|
| `fig3c_topology_real.png` | 真实 wire 掩码(0.35 阈值)→骨架 | `wire_graph.compress_skeleton`(等效重建) | 骨架 7,210px + 39 个结点(度≥3 聚类) | — |
| `fig3d_graph_networks_real.png` ⭐ | `work/.../p_stage23/export/engineering_graph_proposals.json` | `wire_graph.build_wire_graph` | 管线**物理网络候选 238 网 / 362 条折线边**按网着色(合并前的真实中间表示) | 图含 junction_candidates/gap_candidates 字段 |
| `fig3_topology_skeleton.png` / `fig3b_network_overlay.png` | final JSON 63 网渲染 | `network_labels`/`engineering_network_export` | 最终网络叠加(合并后) | 63 nets |
| `fig4b_ownership_overlay_pipeline.png` | 管线自产 `ownership_overlay.png` 原样拷贝 | `pin_ownership` | **管线自带**的引脚归属叠加图(开发期可视化能力佐证) | 287/115/29 归属统计 |

## 四、修复案例

| 文件 | 来源 | 代码模块 | 技术内容 | 指标 |
|---|---|---|---|---|
| `fig4_page13_repair_case.png` | 裁剪=`vision-fetch/gap_crop_case13.png`(开发仓库真实裁剪);数字=修复前 `page13_status_20261020/status.json`(wire 13/22/36,连通 31.08%,65.58 分)+ 修复后 `docs/network_junction_reconciliation.md`(46/13/3,58.72%→85.19%,82.871→92.656) | `network_junction_reconcile` | 结点归并机制:60px 共线重合 / 红点+95% 墨迹;仅合并成员与边,元件引脚不动;每页≤8 组合 | ⚠️ 整页前后叠加渲染需原始页 13 PNG(未随包),已用真实裁剪+状态数字替代 |
| `fig5_evidence_gated_ocr.png` | 冒烟 `logs/rp2350_page11_export_semantics.json` 中 `observed_switch_model` 事件,OCR bbox 叠加在原图裁剪上 | `export_semantics._switch_model` | SW1/SW2 型号恢复:'SW1'→'PTS8153JK250SMTRLFS'(0.998)/'SW2'→同型号(0.9923),旋转视角 tile_0_77_r90 | 审计含 evidence bbox/confidence/view |

## 五、架构图

| 文件 | 内容 |
|---|---|
| `arch_pipeline.png` | 端到端 11 阶段架构(输入→感知→U-Net→归属→建图→网络→补全→修复→校准→导出→JSON),每格含模型/算法、I/O、代码入口 |
| `arch_modules.png` | 六大模块关系(视觉模型层 / OCR 证据体系 / 几何拓扑 / 规则引擎 / 字段配对修复 / 输出适配器)+ 数据流标签 |

## 六、开发期真实记录(dev_repo/,来自 zc-cc1/vision 仓库)

| 文件 | 说明 |
|---|---|
| `unet_trend_chart.png` | U-Net 训练过程指标趋势(开发期) |
| `wire_overlay_example.png` | 导线叠加示例(开发期) |
| `gt_wire_annotation.png` | GT 导线标注示例 |
| `eval8_scoreboard.png` / `task_scoreboard.png` | 开发期评分板(第 8 次评测 / 任务评分) |
| `gap_crop_case13.png` | 页 13 缺口案例原始裁剪(fig4 素材) |
| `H1局部.png` | 页 13 H1 连接器 3 号引脚漏检局部(docs/第13页现状.md 的图像证据) |
| `pin_failures.png` / `pin_miss_example_1_USB1.png` / `unoutput_pins.png` | 引脚失败模式分析图 |
| `第1页_USB1_问题点.png` / `第1页_问题分布.png` | 页 1 USB1 引脚问题定位 |

## 七、成绩证明(待人工补)

`official/` 目录:平台成绩页截图(74 分)、排名/晋级名单(第 47 名/前 61)、提交记录。**当前为空,由用户补充。**
