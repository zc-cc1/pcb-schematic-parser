# C# 调用 HALCON 20.11 示例

`HalconBatchDemo.cs` 演示最小的批量读取、分割、状态输出和资源释放。它不是 HDevelop 导出代码，而是直接使用 `HalconDotNet.HOperatorSet`。

使用步骤：

1. 新建与 HALCON 20.11 位数一致的 .NET Framework 控制台项目。
2. 引用 `%HALCONROOT%/bin/dotnet35/halcondotnet.dll`。
3. 将平台目标设为与 HALCON 运行库一致的 x64 或 x86；通常使用 x64。
4. 把 `HalconBatchDemo.cs` 加入项目后运行。
5. 部署时确认目标机器的 HALCON Runtime、许可证和环境变量均有效。

工程重点：所有 `HObject` 在 `finally` 中释放；每张图独立捕获异常；上层接收稳定状态码，而不是依赖窗口显示。
