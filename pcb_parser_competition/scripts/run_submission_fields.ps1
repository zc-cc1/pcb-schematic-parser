param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Device='cpu',
    [string]$Output,
    [switch]$SkipLabelRoi
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Image)) { $Image=Join-Path $projectRoot $Image }
if (-not $Output) { $Output=Join-Path $projectRoot ('runs\submission_field_predictions\'+[System.IO.Path]::GetFileNameWithoutExtension($Image)) }
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $projectRoot $Output }
$arguments=@('--image',$Image,'--device',$Device,'--output',$Output)
if ($SkipLabelRoi) { $arguments+='--skip-label-roi' }
& (Join-Path $projectRoot '.venv-gpu\Scripts\python.exe') (Join-Path $projectRoot 'scripts\run_submission_fields.py') @arguments
if ($LASTEXITCODE -ne 0) { throw "Submission field inference failed: $LASTEXITCODE" }
