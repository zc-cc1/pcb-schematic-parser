param([Parameter(Mandatory=$true)][string]$Image,[string]$Device='cpu',[string]$Output)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Image)) { $Image=Join-Path $projectRoot $Image }
if (-not $Output) { $Output=Join-Path $projectRoot ('runs\component_tip_predictions\'+[System.IO.Path]::GetFileNameWithoutExtension($Image)) }
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $projectRoot $Output }
& (Join-Path $projectRoot '.venv-gpu\Scripts\python.exe') (Join-Path $projectRoot 'scripts\run_component_tip_refinement.py') --image $Image --output $Output --device $Device
if ($LASTEXITCODE -ne 0) { throw "Tip refinement experiment failed: $LASTEXITCODE" }
