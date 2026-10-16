param(
    [string]$Device = 'cpu',
    [string]$Config = 'configs/baseline_v3_adopted.json',
    [string]$DetectorRun = 's3_v3_detector_quick_dev',
    [string]$RunName = 's3_component_text_quick_dev',
    [string[]]$Images = @('data/standardized/0001/0001.png', 'data/standardized/0002/0002.png'),
    [switch]$ReuseDetections,
    [switch]$ReuseTexts
)
$ErrorActionPreference = 'Stop'
$pcbRoot = Split-Path -Parent $PSScriptRoot
$pcbPython = Join-Path $pcbRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pcbPython)) { throw 'Python 环境不存在：.venv-gpu' }
Push-Location $pcbRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:OMP_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    $pcbImageArgs = @()
    foreach ($pcbImage in $Images) { $pcbImageArgs += @('--image', $pcbImage) }
    if (-not $ReuseDetections) {
        $pcbArgs = @('scripts/run_baseline.py', '--config', $Config, '--output', "runs/$DetectorRun",
            '--devices', $Device, '--modes', 'fused', '--repeats', '1', '--skip-ocr') + $pcbImageArgs
        & $pcbPython @pcbArgs
        if ($LASTEXITCODE -ne 0) { throw "v3 检测失败：$LASTEXITCODE" }
    }
    $pcbArgs = @('scripts/run_component_text.py', '--config', $Config, '--detector-run', "runs/$DetectorRun",
        '--detector-device', $Device.Replace(':', '_'), '--mode', 'fused', '--output', "runs/$RunName") + $pcbImageArgs
    if ($ReuseTexts) { $pcbArgs += '--reuse-texts' }
    & $pcbPython @pcbArgs
    if ($LASTEXITCODE -ne 0) { throw "OCR 绑定失败：$LASTEXITCODE" }
} finally {
    Pop-Location
}
