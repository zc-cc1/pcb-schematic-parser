param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Device = 'cpu',
    [string]$Weights,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python environment not found: $pythonPath" }
if (-not [System.IO.Path]::IsPathRooted($Image)) { $Image = Join-Path $projectRoot $Image }
if (-not $Weights) {
    $Weights = Join-Path $projectRoot 'weights\unet_wire_pin_v1.pt'
    if (-not (Test-Path -LiteralPath $Weights)) {
        $Weights = Join-Path $projectRoot 'runs\training\unet_wire_pin_v1\best.pt'
    }
}
if (-not [System.IO.Path]::IsPathRooted($Weights)) { $Weights = Join-Path $projectRoot $Weights }
if (-not $Output) {
    $caseName = [System.IO.Path]::GetFileNameWithoutExtension($Image)
    $Output = Join-Path $projectRoot "runs\unet_inference_v2_predictions\$caseName"
}
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output = Join-Path $projectRoot $Output }
$arguments = @(
    (Join-Path $projectRoot 'scripts\run_unet.py'),
    '--image', $Image,
    '--weights', $Weights,
    '--device', $Device,
    '--inference-config', (Join-Path $projectRoot 'configs\unet_inference_v2.json'),
    '--output', $Output
)
& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) { throw "U-Net inference failed with exit code $LASTEXITCODE" }
