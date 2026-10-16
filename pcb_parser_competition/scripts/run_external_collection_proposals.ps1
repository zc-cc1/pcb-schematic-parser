param([string]$RunName = 'external_collection_v1_proposals')
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunName must be a simple folder name' }
$pcbProjectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $pcbProjectRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $pcbReviewRows = Get-Content data/external_collection_v1/review_manifest.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
    $pcbCandidateRows = @($pcbReviewRows | Where-Object { $_.reserved_split -eq 'train_candidate_pending_annotation' })
    if ($pcbCandidateRows.Count -eq 0) { throw 'No training candidate pages in review manifest' }
    $pcbArguments = @('scripts/run_baseline.py', '--config', 'configs/baseline_finetuned.json',
        '--output', "runs/$RunName", '--devices', 'cuda:0', '--modes', 'sliced', '--repeats', '1', '--skip-ocr')
    foreach ($pcbRow in $pcbCandidateRows) { $pcbArguments += @('--image', $pcbRow.image) }
    & .venv-gpu/Scripts/python.exe @pcbArguments
    if ($LASTEXITCODE -ne 0) { throw "Proposal generation failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
