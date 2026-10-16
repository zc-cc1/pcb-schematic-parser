param(
    [Parameter(Mandatory=$true)][string]$Image,
    [string]$Device='cpu',
    [string]$Output,
    [switch]$LocalOCR
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Image)) { $Image=Join-Path $projectRoot $Image }
if (-not $Output) {
    $caseName=[System.IO.Path]::GetFileNameWithoutExtension($Image)
    $Output=Join-Path $projectRoot "runs\pin_text_predictions\$caseName"
}
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output=Join-Path $projectRoot $Output }
$pinTextArgs=@((Join-Path $projectRoot 'scripts\run_pin_text.py'),'--image',$Image,'--device',$Device,'--output',$Output)
if ($LocalOCR) { $pinTextArgs+='--enable-local' }
& (Join-Path $projectRoot '.venv-gpu\Scripts\python.exe') @pinTextArgs
if ($LASTEXITCODE -ne 0) { throw "Pin text inference failed: $LASTEXITCODE" }
