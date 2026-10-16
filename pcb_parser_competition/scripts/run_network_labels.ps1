param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Device='cpu',
    [string]$Output,
    [switch]$LearnedJunctionPilot
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Image)) { $Image=Join-Path $projectRoot $Image }
if (-not $Output) { $Output=Join-Path $projectRoot ('runs\network_label_predictions\'+[System.IO.Path]::GetFileNameWithoutExtension($Image)) }
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $projectRoot $Output }
$arguments=@('--image',$Image,'--device',$Device,'--output',$Output)
if ($LearnedJunctionPilot) { $arguments+='--learned-junction-pilot' }
& (Join-Path $projectRoot '.venv-gpu\Scripts\python.exe') (Join-Path $projectRoot 'scripts\run_network_labels.py') @arguments
if ($LASTEXITCODE -ne 0) { throw "Network label inference failed: $LASTEXITCODE" }
