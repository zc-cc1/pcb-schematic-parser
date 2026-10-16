# U-Net 旋转增强独立对照

目标是检验0036中的方向敏感性是否可通过训练增强改善，使模型直接读取原方向PNG时也恢复细横线。诊断中旋转90°的高分不是本次采用指标，推理不旋转、不加深、不融合多视角、不补线。

## 固定设置

- 使用冻结v1第57轮权重作为初始模型。
- 沿用v2复核草稿：103训练页、26去重验证页；0036不进入训练。没有新增/修改标签。
- 同步旋转RGB、导线目标、引脚热图及有效区域，0/90/180/270度各25%。torch.rot90无插值，不改变颜色、线宽或取样值。
- 与上一轮普通微调对照保持相同种子、切片/顺序、batch2、512/384、AdamW、16轮、学习率1e-4余弦至2e-5、AMP以及原损失函数。
- 复用已经完成的v2 control组，避免重复运行同一组训练。其初始权重、manifest、训练/验证case顺序及全部影响训练的匹配参数在启动前核对。
- 仅旋转增强是组间差异；没有同时增加灰度增强、位移增强、连通损失或更换模型结构。
- checkpoint按原方向验证损失最小选择。每轮0036监测仅用于进度解释，不参与选轮次。

## 验证

固定原模型阈值wire0.5、pin0.8及K5二次细化。原模型、普通微调对照、旋转组在相同原方向26页上比较导线像素P/R/Dice、源路径附近2px的连通代理和引脚坐标5px匹配F1。0036另外检查X1两侧、DOUT横线连通及已人工确认的非导线IC边框区域。

648个原始来源文件在运行前后核对SHA256。正式权重及推理配置没有自动替换。验证页参与过模型选择和后续开发，标签未完整人工认证，因此指标不等同于官方PinF1/NetF1或独立工程测试。

## 运行

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv-gpu\Scripts\python.exe -X utf8 -m unittest discover -s tests -p 'test_unet_rotation_training.py' -q
.\.venv-gpu\Scripts\python.exe -X utf8 scripts/retrain_unet_rotation_v3.py --allow-draft --smoke-steps 4
.\.venv-gpu\Scripts\python.exe -X utf8 scripts/retrain_unet_rotation_v3.py --allow-draft
.\.venv-gpu\Scripts\python.exe -X utf8 scripts/evaluate_unet_rotation_v3.py
```

不覆盖已有rotation训练或evaluation目录。默认输出`runs/unet_retrain_rotation_v3`。summary、history、protocol、各轮0036原方向概率及最终评估留在该目录。完整结果以evaluation/report.md和evaluation/conclusion.md为准。
