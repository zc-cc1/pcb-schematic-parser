param(
    [string[]]$Devices = @('cuda:0'),
    [string[]]$Modes = @('fused'),
    [int]$Repeats = 3,
    [string]$RunName = 's3_y11s_baseline_quick_dev',
    [string]$Config = 'configs/baseline_y11s.json'
)
$ErrorActionPreference = 'Stop'
$pcbRoot = Split-Path -Parent $PSScriptRoot
$pcbPython = Join-Path $pcbRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pcbPython)) { throw 'GPU 环境尚未建立：.venv-gpu' }
Push-Location $pcbRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $pcbArguments = @('scripts/run_baseline.py', '--config', $Config, '--image', 'data/standardized/0002/0002.png',
        '--image', 'data/standardized/0001/0001.png', '--output', "runs/$RunName", '--repeats', "$Repeats", '--devices')
    $pcbArguments += $Devices
    $pcbArguments += '--modes'
    $pcbArguments += $Modes
    & $pcbPython @pcbArguments
    if ($LASTEXITCODE -ne 0) { throw "基线运行失败，退出码：$LASTEXITCODE" }
} finally {
    Pop-Location
}
