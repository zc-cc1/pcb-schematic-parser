# PCB 原理图解析项目工作目录

本仓库的文件来源为 `C:\Users\Pc\Desktop\新建文件夹`，按该目录中的实际版本保存项目源码和资料。

## 目录

| 路径 | 内容 |
| --- | --- |
| `pcb_parser_competition/` | PCB 原理图解析项目的源码、配置、测试、文档和资料 |
| `_sample2606/` | 赛题示例代码和说明 |
| `评测结果/` | 评测记录、结果及相关代码，排除模型权重和缓存 |
| `赛题六公开数据集/` | 数据集说明文档，原始数据在本地保留 |
| `case_001_扩展示例.json` | 扩展示例 |
| `schematic_rp2350_pico2.png` | 原理图示例图片 |

项目状态及运行方式见 [PCB 项目说明](pcb_parser_competition/README.md)。

## 保存范围

按用户选择，本仓库保存源码、配置、测试、文档和资料。Python 环境、第三方依赖、缓存、临时文件、模型权重、训练运行产物、压缩包、Docker 镜像及原始或派生训练数据不上传；这些文件仍保留在本机。具体规则见各级 `.gitignore`。

`Dataset300/`、公开数据集原始文件和 `edarace-agent-base_ubuntu24.tar` 需要在运行时另行准备。仅克隆仓库不能还原本机全部运行环境，文档中的部分本地数据和模型路径需要相应配置。

## 后续同步

在 `C:\Users\Pc\Desktop\新建文件夹` 打开 PowerShell：

```powershell
git status
git add .
git commit -m "更新项目代码与资料"
git push
```

仓库设为私有。Git 不保存空目录；已有第三方代码和资料的许可声明随原文件保留。
