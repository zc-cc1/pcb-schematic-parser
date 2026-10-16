# Docker Linux 引擎 500：固件虚拟化未启用

2026-10-16 已做只读检查，并正常重启过一次 Docker Desktop。没有执行工厂重置、删除镜像或注销 WSL 分发版。

检查结果：

- CPU：AMD Ryzen 5 5600。
- 主板：ASUS PRIME A520M-R。
- CPU 支持虚拟机扩展和二级地址转换。
- `VirtualizationFirmwareEnabled = false`。
- `HypervisorPresent = false`。
- Docker Desktop 选用了 WSL 2，但引擎状态是 stopped。
- Docker 后端日志报告 `WslVirtualizationDisabledInFirmware` 和 `HCS_E_HYPERV_NOT_INSTALLED`。
- 当前没有设置 DOCKER_API_VERSION、DOCKER_HOST 或 DOCKER_CONTEXT 环境变量。
- Windows“虚拟机平台”的状态查询需要管理员权限，本次未确认该功能状态。

这说明当前必须先解决固件虚拟化开关，不能仅通过重启 Docker 或改 API 版本来完成启动。

## 操作

1. 在管理员 PowerShell 中确保 Windows 虚拟机平台已启用：

   ```powershell
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

2. 保存工作后重启电脑，在出现华硕开机画面时连续按 Delete（Del）进入 BIOS。
3. 如处于简易模式，按 F7 进入 Advanced Mode。华硕 AMD 主板的常见路径为 `Advanced → CPU Configuration → SVM Mode`，将它设为 `Enabled`。具体界面随 BIOS 版本变化，以实际选项为准。
4. 按 F10 保存更改并退出，让 Windows 正常启动。
5. 任务管理器 → 性能 → CPU，确认“虚拟化”为“已启用”。
6. 启动 Docker Desktop，再运行 `docker version`，确认同时显示 Client 和 Server。

如找不到 SVM Mode，可提供 BIOS 页面照片后继续定位。无需安装额外 Ubuntu 分发版来替代这个固件开关；Docker Desktop 会创建自己的运行环境。

官方资料：

- [华硕主板启用虚拟化](https://www.asus.com/support/faq/1045141/)
- [微软 WSL 虚拟机启动故障说明](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting#installation-issues)
- [Docker Desktop WSL 2 后端](https://docs.docker.com/desktop/features/wsl/)
