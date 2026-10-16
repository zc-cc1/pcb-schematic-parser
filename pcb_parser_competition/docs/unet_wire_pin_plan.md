# U-Net导线与引脚方案：当前默认路线

2026-10-11用户明确要求使用U-Net、不采用传统方法。导线与引脚的位置识别改由学习模型完成；之前的传统分割/模板路线不再是默认实现。本文记录已完成的接口和仍需完成的训练与网络识别。

## 选用模型及来源

采用PyTorch U-Net，复用 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 的DoubleConv、Down、Up与OutConv结构模块，固定提交 `21d7850f2af30a9695bbeea75f3136aa538cfc4a`。上游文件原样保存于 `src/pcb_parser/vendor/pytorch_unet/`；许可证GPL-3.0、每个源码哈希和版本记录均保留。

PCB适配器是 `src/pcb_parser/unet_model.py`：RGB输入、4层下采样与对应跳跃连接、基础通道32，双输出共4318434个参数。第一个输出是导线像素概率，第二个是引脚tip热图，两个任务共用编码/解码器。这里的轻量通道配置和tip任务是本项目适配，不是上游已经训练好的PCB模型。

导线损失采用加权BCEWithLogits与Soft Dice，引脚Gaussian热图采用加权BCE。独立通道不采用互斥Softmax：接点与导线可以在同位置重合。像素padding不参与损失。

当前没有经过识别质量验证的U-Net正式权重。原有YOLO与OCR权重不变，不能拿它们当作U-Net初始化权重。

## 图像入口与输出

`scripts/run_unet.py` 只读原PNG与U-Net checkpoint，不读同页标注、CAD或转换参考。采用原分辨率512×512切片、stride384，重叠概率平均回原图，避免整图缩小使细线消失。

产物为 `probabilities.npz`、`wire_probability.png`、`wire_mask.png`、`pin_probability.png`与`pin_candidates.json`。掩膜阈值与热图峰值提取作用于模型概率，不对原图执行颜色阈值、形态学分割、Hough直线检测或模板定位。

tip候选目前只有坐标和概率，尚未恢复元件归属、编号和名称。身份关联将结合实际检测框与OCR，保留歧义和失败原因。当前没有实现完整引脚与网络输出。

分割之后仍需将预测像素/接点转换成线段与图关系；这属于输出结构处理。交叉处的电气连接不能只由像素相接判断。连接点分支需要独立核验标签后再训练；当前只启用导线和tip两个输出，尚未训练交叉连接识别。

## 数据准备的实际结果

`scripts/prepare_unet_dataset.py` 已从Dataset300的离线 `keyPointPaths.wire` 和所属器件引脚tip生成300页标签草稿。它绘制的是源标注几何，不是对图片运行传统分割后制造标签。

- 300页生成成功，失败0页。
- 132页通过当前结构完整性门槛：103页既有训练划分、28页既有验证划分、1页原排除划分。
- 逻辑飞线与deleted记录被排除；缺少路径时不根据两个接点虚构直线。
- 缺路径、越界、无效tip或未归属根级pin的页面不作为默认可用草稿。
- 当前人工批准的完整导线/tip标签页为0。132页结构检查通过不等于像素与tip全部正确，也不等于最终正式可训练页数。
- 导线监督目前是源路径按3px线宽绘制；需要复核实际线宽、漏线、符号引线与交叉。tip使用sigma2px的Gaussian热图。
- 复用原有case划分，工程级分组仍待核验；正式泛化评估须补充未参与调参的工程。

数据位于 `data/segmentation_unet_v1/`，全量清单为manifest.jsonl，summary.json记录门槛结果。0001/0002/0131保存了label_review.png：红色是源路径监督，蓝圈是源tip。它们是标签审查图，不是模型预测。

## 已完成的运行验证

8项新增测试通过：排除逻辑飞线及删除记录、缺路径不补线、tip/base区分、画布校验、padding排除损失、真实梯度、奇数尺寸还原和原尺寸切片覆盖。

CPU在0002草稿上完成4个真实优化步骤，128×128调试切片、双输出模型4318434参数，末层权重确实更新；随后从0002原PNG完成512切片的预测入口。该运行是 `cpu_smoke_only`，没有验证集或质量指标。随机初始化仅更新4步时预测很差属于预期，不能宣称U-Net已训练完成或已提升准确率。

记录：`runs/training/unet_wire_pin_smoke/summary.json`；预测入口验证：`runs/s4_unet_smoke/0002/summary.json`。烟雾测试checkpoint不会被设为正式默认权重，推理使用它必须显式提供`--allow-smoke`。

检查时RTX5060显存7683/8151MiB、GPU利用率100%，其他训练仍在运行。因此当前没有启动正式GPU训练。训练脚本在可用显存不足配置门槛时直接拒绝启动，不中断其他进程；该显存门槛不代表已完成实际GPU批量性能测试。

## 接下来的顺序与指标

1. 核验标签：选已通过结构检查的灰度/彩色、水平/竖直、密集/稀疏页，复核导线覆盖、线宽及tip。缺失区域应补标或明确ignore，不能当作确定的背景。将批准案例及证据登记到 `configs/unet_label_reviews.json`。
2. 首轮训练：配置 `configs/unet_wire_pin.json`，60epochs、batch2、AdamW lr0.0003、AMP；真实时间与显存以运行实测为准。默认只使用人工批准且结构可用的训练/验证页。`--allow-draft`只用于明确标记的草稿监督实验。
3. 评价导线：在固定阈值下报告像素Precision/Recall/Dice/IoU，另外统计细线漏段与断裂。建议首轮导线Dice≥90%作为内部目标；不能仅用像素分数代表电气拓扑。
4. 评价tip：先建立至少50个核验点，5px一对一匹配Precision≥95%、Recall≥90%作为内部目标，另外报告上游元件归属、编号与名称错误。
5. 加入连接点/交叉分类监督，评价错合并、错拆分与未决交叉；再接入NetworkX与完整pins/nets导出。正式PinF1、NetHypergraphF1和NetLineF1均仍待官方评测器。

上述目标均未实测。类别与位号遗漏仍来自真实YOLO/OCR输出，正式全链路评价不能用参考框或答案替换。

## 可复现入口

```powershell
# 项目根目录：生成草稿，原件只读
& .\.venv-gpu\Scripts\python.exe scripts/prepare_unet_dataset.py
# CPU只验证训练和推理链路，明确使用草稿，不报告识别成绩
& .\.venv-gpu\Scripts\python.exe scripts/train_unet.py --smoke --allow-draft --case-id 0002 --output runs/training/unet_wire_pin_smoke
& .\.venv-gpu\Scripts\python.exe scripts/run_unet.py --image data/standardized/0002/0002.png --weights runs/training/unet_wire_pin_smoke/best.pt --device cpu --allow-smoke --output runs/s4_unet_smoke/0002
# 标签复核后、GPU空闲且显存充足时启动正式训练
& .\.venv-gpu\Scripts\python.exe scripts/train_unet.py --device cuda:0
```

配置中的checkpoint仍为null。当前训练/预测只依赖已经安装的PyTorch、NumPy与Pillow；无需为了分割安装scikit-image或Skan。开源导入工具 `scripts/import_unet_source.py` 可核验本地源码哈希，无需每次训练重新下载。
