# 在VS Code中运行basics.cpp

## 现在直接这样运行

1. 在VS Code中选择“文件 → 从文件打开工作区”，打开本目录的 `第一周C++.code-workspace`。也可以通过“打开文件夹”打开 `01_cpp_basics`，不要只打开单独的cpp文件。
2. 修改代码后先按Ctrl+S保存。
3. 按 **Ctrl+Shift+B**，执行默认任务“第一周：CMake 编译并运行 basics”。
4. 在下方“终端”面板查看结果。窗口会保留输出，程序无需添加暂停语句。

预期输出：

```text
100% tests passed out of 1
value=20
value=30
width=300, height=180
add(3,5)=8
int sum=260, narrowed=4
CMake build, test and run succeeded. OpenCV was not required.
```

## 为什么之前选了编译器仍可能不能运行

`C_Cpp.default.compilerPath`配置主要服务于代码分析与补全。构建运行任务由 `.vscode/tasks.json` 定义，现在调用 `run_cmake_basics.ps1`，自动执行配置、编译、CTest与程序运行。

`CMakePresets.json`中的basics预设显式设置 `WEEK01_WITH_OPENCV=OFF`，使用GCC 16.1.0和项目本地Ninja。构建结果位于项目根目录 `.build/week01-basics`，避开中文输出路径和旧build缓存。Ninja位于项目根目录 `.tools/ninja/bin`，不需要修改系统安装。

旧build目录缓存过 `WEEK01_WITH_OPENCV=ON`。只修改CMakeLists.txt中的默认值不会自动覆盖已经缓存的值，这就是之前仍查找OpenCV的原因。新任务使用明确关闭开关的预设与独立构建目录。

如果想使用CMake扩展按钮，先按Ctrl+Shift+P，执行 `CMake: Select Configure Preset`，选择“第一周：纯C++（不需要OpenCV）”，再执行 `CMake: Configure` 和 `CMake: Build`。若VS Code仍显示旧任务，执行 `Developer: Reload Window` 后重试。

运行时只在任务进程中加入 `C:/msys64/ucrt64/bin`，让程序能找到GCC运行库；不会修改系统PATH。PowerShell的Bypass参数仅作用于这次任务进程，不修改持久执行策略。

目前未检测到GDB，所以本操作使用 **Ctrl+Shift+B编译运行**。F5及带“Debug/调试”的按钮需要额外安装并配置GDB。普通运行不要求先安装调试器。

本任务专门运行纯C++的basics.cpp。依赖OpenCV的demo.cpp与exercise_solution.cpp目前不会参与构建。以后安装匹配GCC的OpenCV SDK后，再单独配置开启该开关的工程。

验证记录（2026-09-21）：CMake 4.4.3、GCC 16.1.0、项目本地Ninja 1.13.2已完成配置、编译和运行，CTest的cpp_basics测试1/1通过。使用与VS Code任务相同的脚本执行验证；未通过界面点击验证。OpenCV C++示例不在此次验证范围内。
