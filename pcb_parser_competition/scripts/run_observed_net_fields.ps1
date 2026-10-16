param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Device='cpu',
    [string]$Output
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Image)) { $Image=Join-Path $projectRoot $Image }
if (-not $Output) { $Output=Join-Path $projectRoot ('runs\observed_net_fields_y11s_predictions\'+[System.IO.Path]::GetFileNameWithoutExtension($Image)) }
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $projectRoot $Output }
& (Join-Path $projectRoot '.venv-gpu\Scripts\python.exe') (Join-Path $projectRoot 'scripts\run_observed_net_fields.py') --image $Image --output $Output --device $Device
if ($LASTEXITCODE -ne 0) { throw 'Observed net field inference failed' }
