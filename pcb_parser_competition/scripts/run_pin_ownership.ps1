param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Device='cpu',
    [string]$Output
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Image)) { $Image=Join-Path $projectRoot $Image }
if (-not $Output) {
    $caseName=[System.IO.Path]::GetFileNameWithoutExtension($Image)
    $Output=Join-Path $projectRoot "runs\pin_ownership_y11s_predictions\$caseName"
}
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $projectRoot $Output }
& (Join-Path $projectRoot '.venv-gpu\Scripts\python.exe') (Join-Path $projectRoot 'scripts\run_pin_ownership.py') --image $Image --device $Device --output $Output
if ($LASTEXITCODE -ne 0) { throw "Pin ownership inference failed: $LASTEXITCODE" }
