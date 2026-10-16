param(
    [ValidateSet('Check','Prepare','Smoke','Train')][string]$Mode = 'Check',
    [string]$Device = 'cuda:0',
    [string]$Config = 'configs/unet_wire_pin.json',
    [string]$RunName = 'unet_wire_pin_v1',
    [switch]$AllowDraft,
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [string]$Resume = ''
)
$ErrorActionPreference = 'Stop'
$pcbUnetRoot = Split-Path -Parent $PSScriptRoot
$pcbUnetPython = Join-Path $pcbUnetRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pcbUnetPython)) { throw '缺少 .venv-gpu Python 环境' }
Push-Location $pcbUnetRoot
try {
    $env:PYTHONIOENCODING='utf-8'
    $env:OMP_NUM_THREADS='4'
    $env:MKL_NUM_THREADS='4'
    if ($Mode -eq 'Prepare') {
        & $pcbUnetPython scripts/prepare_unet_dataset.py
    } elseif ($Mode -eq 'Check') {
        & $pcbUnetPython scripts/verify_unet_runtime.py
    } elseif ($Mode -eq 'Smoke') {
        & $pcbUnetPython scripts/train_unet.py --smoke --allow-draft --device cpu --case-id 0002 --output runs/training/unet_wire_pin_smoke
    } else {
        $pcbUnetArgs=@('scripts/train_unet.py','--config',$Config,'--device',$Device,'--output',"runs/training/$RunName")
        if ($AllowDraft) { $pcbUnetArgs+='--allow-draft' }
        if ($Epochs -gt 0) { $pcbUnetArgs+=@('--epochs',"$Epochs") }
        if ($BatchSize -gt 0) { $pcbUnetArgs+=@('--batch-size',"$BatchSize") }
        if ($Resume) { $pcbUnetArgs+=@('--resume',$Resume) }
        & $pcbUnetPython @pcbUnetArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "U-Net $Mode 失败：$LASTEXITCODE" }
} finally {
    Pop-Location
}
