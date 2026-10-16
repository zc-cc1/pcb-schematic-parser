param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$pcbExportRoot = (Resolve-Path -LiteralPath 'D:\pcb_submission_export').Path.TrimEnd('\')
$pcbProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path.TrimEnd('\')
$pcbLatestRoot = Join-Path $pcbExportRoot 'pcb_image_20261016_155125_dec67ea7'
$pcbKeepZip = Join-Path $pcbLatestRoot 'submission_official_slim.zip'
$pcbExpectedHash = 'C8CFE06E8F047692A640EAF22DDEA70C1298C1611041442AD629306F1863B26A'
$pcbArchiveNames = @('submission_image.zip','submission_target.zip','submission.tar','source_snapshot.zip')
if ((Get-Item -LiteralPath $pcbExportRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Export root must not be a link.'
}
$pcbCandidates = @(
    foreach ($pcbDir in Get-ChildItem -LiteralPath $pcbExportRoot -Directory) {
        if ($pcbDir.Name -notmatch '^pcb_image_\d{8}_\d{6}_[0-9a-f]{8}$') { continue }
        if ($pcbDir.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Unexpected linked directory.' }
        if (-not $pcbDir.FullName.StartsWith($pcbExportRoot + '\',[StringComparison]::OrdinalIgnoreCase)) {
            throw 'Directory escaped export root.'
        }
        foreach ($pcbFile in Get-ChildItem -LiteralPath $pcbDir.FullName -File) {
            if ($pcbFile.Name -notin $pcbArchiveNames) { continue }
            if ($pcbDir.FullName -eq $pcbLatestRoot -and $pcbFile.Name -ne 'submission.tar') { continue }
            $pcbFile
        }
    }
    foreach ($pcbOldName in @('submission.zip','submission_target_source.zip','submission_publicfix_source.zip')) {
        $pcbOldPath = Join-Path $PSScriptRoot $pcbOldName
        if (Test-Path -LiteralPath $pcbOldPath -PathType Leaf) { Get-Item -LiteralPath $pcbOldPath }
    }
)
foreach ($pcbFile in $pcbCandidates) {
    $pcbAbsolute = (Resolve-Path -LiteralPath $pcbFile.FullName).Path
    $pcbInsideApprovedRoot = $pcbAbsolute.StartsWith($pcbExportRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or
        $pcbAbsolute.StartsWith($pcbProjectRoot + '\dist\',[StringComparison]::OrdinalIgnoreCase)
    if (-not $pcbInsideApprovedRoot -or $pcbAbsolute -eq $pcbKeepZip -or
        ($pcbFile.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unsafe cleanup target.' }
}
$pcbBefore = @(Get-PSDrive -Name C,D | Select-Object Name,Free)
$pcbReport = [ordered]@{
    generated_at=(Get-Date -Format o); executed=[bool]$Execute
    retained_upload=$pcbKeepZip; retained_sha256=$pcbExpectedHash.ToLowerInvariant()
    files=@($pcbCandidates | Select-Object FullName,Length)
    planned_deleted_bytes=($pcbCandidates | Measure-Object Length -Sum).Sum
    free_before=$pcbBefore
    preserved='Latest upload ZIP, latest source snapshot, source code, model weights, logs, predictions, official input TAR and Docker images'
}
$pcbReportPath = Join-Path $PSScriptRoot 'submission_archive_cleanup_20261016.json'
if ($Execute) {
    if ((Get-Item -LiteralPath $pcbKeepZip).Length -ne 3562329214) { throw 'Latest ZIP size changed.' }
    if ((Get-FileHash -LiteralPath $pcbKeepZip -Algorithm SHA256).Hash -ne $pcbExpectedHash) {
        throw 'Latest ZIP hash changed; cleanup cancelled.'
    }
    $pcbReport | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $pcbReportPath -Encoding utf8
    foreach ($pcbFile in $pcbCandidates) { Remove-Item -LiteralPath $pcbFile.FullName -Force }
    $pcbRemaining = @($pcbCandidates | Where-Object { Test-Path -LiteralPath $_.FullName })
    if ($pcbRemaining.Count) { throw 'Some archive deletions did not complete.' }
    $pcbReport['free_after'] = @(Get-PSDrive -Name C,D | Select-Object Name,Free)
    $pcbReport['verified_deleted_files'] = $pcbCandidates.Count
    $pcbReport['completed_at'] = Get-Date -Format o
}
$pcbReport | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $pcbReportPath -Encoding utf8
$pcbReport | ConvertTo-Json -Depth 6
