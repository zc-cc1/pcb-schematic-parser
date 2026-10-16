param(
    [string]$ImageTag = 'pcb-parser:local-cuda-0e4a7241',
    [string]$OutputDirectory = 'D:\pcb_submission_export',
    [string]$SourceZip,
    [string]$FrozenSource,
    [string]$PreviousBuild,
    [string]$ExpectedSourceZipSha256 = '20b1bb6f783c83b619ba4c6962d1b1750e951a17149977e728573616a9f2ed53',
    [ValidatePattern('^[A-Za-z0-9_-]+\.zip$')][string]$ArchiveName = 'submission_ic_led_gpu.zip',
    [string]$ResumeBuildRoot,
    [switch]$SkipLoadCheck
)
# Finish an export that was interrupted after a fully verified image had already
# been built. Nothing is rebuilt and no dependency is re-downloaded: the image is
# re-verified in place, then saved, loaded back, zipped and checksummed.
# Pass -ResumeBuildRoot to continue inside an existing build directory: an
# already saved submission.tar is reused instead of writing another multi-GB copy.
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 does not load these by default; ZipArchive needs both.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $SourceZip) { $SourceZip = Join-Path $projectRoot 'dist\submission_icled_gpu_source.zip' }
if (-not $FrozenSource) { $FrozenSource = Join-Path $projectRoot 'submission\platform_test_20261016_icled_gpu_1' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot 'dist\local_image_exports' }
$dockerExecutable = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $dockerExecutable) { throw 'Docker CLI is unavailable.' }
$pythonExecutable = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExecutable) { throw 'Python is unavailable on the host; the archive verifier needs it.' }
$engine = (& $dockerExecutable info --format '{{.OSType}}' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $engine -ne 'linux') { throw 'Docker Desktop must be running in Linux container mode.' }

function Invoke-DockerLogged {
    param([string[]]$CommandArguments, [string]$LogPath)
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $dockerExecutable @CommandArguments 2>&1 |
            ForEach-Object { $_.ToString() } | Tee-Object -FilePath $LogPath
        $commandExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($commandExitCode -ne 0) { throw "Docker command failed ($commandExitCode). Inspect $LogPath" }
}

# The image tag under test must be the one the previous build actually produced.
$imageId = (& $dockerExecutable image inspect $ImageTag --format '{{.Id}}' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $imageId) { throw "Image not present locally: $ImageTag" }
$imagePlatform = (& $dockerExecutable image inspect $ImageTag --format '{{.Os}}/{{.Architecture}}' | Out-String).Trim()
$entrypoint = (& $dockerExecutable image inspect $ImageTag --format '{{json .Config.Entrypoint}}' | Out-String).Trim()
$imageConfig = (& $dockerExecutable image inspect $ImageTag --format '{{json .Config.Env}}' | Out-String).Trim()
if ($imagePlatform -ne 'linux/amd64') { throw "Expected linux/amd64, got $imagePlatform" }
if ($entrypoint -ne '["/submission/run.sh"]') { throw "Entrypoint must be /submission/run.sh, got $entrypoint" }
if ($imageConfig -notmatch 'PCB_DEVICE=auto') { throw 'Image must default to PCB_DEVICE=auto.' }

if (-not $PreviousBuild) {
    $candidates = Get-ChildItem -LiteralPath $OutputDirectory -Directory -Filter 'pcb_image_*' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    foreach ($candidate in $candidates) {
        $log = Join-Path $candidate.FullName 'docker_build.log'
        if (-not (Test-Path -LiteralPath $log)) { continue }
        if (Select-String -LiteralPath $log -SimpleMatch -Pattern "naming to docker.io/library/$ImageTag" -Quiet) {
            $PreviousBuild = $candidate.FullName
            break
        }
    }
    if (-not $PreviousBuild) { throw "No previous build log under $OutputDirectory names $ImageTag." }
}
$PreviousBuild = (Resolve-Path -LiteralPath $PreviousBuild).Path
$previousTag = (Select-String -LiteralPath (Join-Path $PreviousBuild 'docker_build.log') -SimpleMatch -Pattern 'naming to docker.io/library/' |
    Select-Object -Last 1).Line.Trim()
Write-Host "Reusing image $ImageTag from $PreviousBuild ($previousTag)."
$previousTestOutput = Join-Path $PreviousBuild 'test-output'
if (-not (Test-Path -LiteralPath (Join-Path $previousTestOutput 'final_results\decision_log.json'))) {
    throw 'The previous build has no completed inference output to re-verify.'
}

$sourceZipAbsolute = (Resolve-Path -LiteralPath $SourceZip).Path
$sourceZipHash = (Get-FileHash -LiteralPath $sourceZipAbsolute -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ExpectedSourceZipSha256 -and $sourceZipHash -ne $ExpectedSourceZipSha256) {
    throw "Source ZIP hash $sourceZipHash does not match the verified $ExpectedSourceZipSha256."
}
$frozenSourceAbsolute = (Resolve-Path -LiteralPath $FrozenSource).Path
foreach ($fileName in @('agent.py','run.sh','requirements.txt','package_manifest.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $frozenSourceAbsolute $fileName))) {
        throw "Frozen source is missing $fileName."
    }
}

if ($ResumeBuildRoot) {
    $buildRoot = (Resolve-Path -LiteralPath $ResumeBuildRoot).Path
    if (-not (Test-Path -LiteralPath (Join-Path $buildRoot 'submission.tar'))) {
        throw "Resume directory has no submission.tar: $buildRoot"
    }
    Write-Host "Resuming in existing build directory $buildRoot."
    # Keep the recorded scripts identical to the ones that produced the artifacts.
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $buildRoot 'complete_export_from_verified_image.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'verify_submission_archive.py') -Destination $buildRoot -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'pack_submission_zip.py') -Destination $buildRoot -Force
} else {
    $buildName = 'pcb_image_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
    $buildRoot = [IO.Path]::GetFullPath((Join-Path $OutputDirectory $buildName))
    New-Item -ItemType Directory -Path $buildRoot -ErrorAction Stop | Out-Null
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $buildRoot 'complete_export_from_verified_image.ps1')
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'verify_submission_archive.py') -Destination $buildRoot
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'pack_submission_zip.py') -Destination $buildRoot
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'build_test_export.ps1') -Destination $buildRoot
    foreach ($name in @('docker_build.log','docker_run.log','image_manifest_check.log','target_contract.log','Dockerfile.used')) {
        $previous = Join-Path $PreviousBuild $name
        if (Test-Path -LiteralPath $previous) { Copy-Item -LiteralPath $previous -Destination (Join-Path $buildRoot "previous_$name") }
    }
    # Small, durable slice of the CPU smoke evidence (skip the Ultralytics scratch).
    $recordRoot = Join-Path $buildRoot 'cpu_smoke_records'
    New-Item -ItemType Directory -Path $recordRoot | Out-Null
    foreach ($relative in @('decision_log.json','final_results','logs','project_results','component_results','runtime_environment.json')) {
        $source = Join-Path $previousTestOutput $relative
        if (-not (Test-Path -LiteralPath $source)) { continue }
        Copy-Item -LiteralPath $source -Destination $recordRoot -Recurse
    }
}

# 1. The frozen source snapshot must still describe the image exactly.
$manifestCheck = @'
import hashlib,json
from pathlib import Path
root=Path('/submission')
actual=(root/'package_manifest.json').read_bytes()
assert actual==Path('/expected/package_manifest.json').read_bytes(), 'Image manifest differs from the frozen source'
manifest=json.loads(actual)
for relative,expected in manifest['files'].items():
    data=(root/relative).read_bytes()
    assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'], relative
print('Image /submission matches the frozen source manifest and every bundled file hash.')
'@
Invoke-DockerLogged -CommandArguments @('run','--rm','--network','none','--entrypoint','python3',
    '--mount',"type=bind,source=$frozenSourceAbsolute,target=/expected,readonly",$ImageTag,'-c',$manifestCheck) `
    -LogPath (Join-Path $buildRoot 'image_manifest_recheck.log')

# 2. Re-run the Target contract check over the recorded CPU smoke output.
$contractCheck = @'
import json
import hashlib
from pathlib import Path
from adapter import COMPONENT_OUTPUT_POLICY, to_target, validate_target
manifest=json.loads(Path('/submission/package_manifest.json').read_text())
for relative,expected in manifest['files'].items():
    assert hashlib.sha256((Path('/submission')/relative).read_bytes()).hexdigest()==expected['sha256'], relative
for case in ['0036','0042']:
    data=json.loads((Path('/verify/final_results')/(case+'_Pred.json')).read_text())
    raw=json.loads((Path('/verify/project_results')/(case+'_Pred.json')).read_text())
    augmented=Path('/verify/component_results')/(case+'_Pred.json')
    if augmented.exists():
        original=raw
        raw=json.loads(augmented.read_text())
        assert raw['nets']==original['nets'], 'Power component export altered networks'
        assert all(raw['components'][key]==value for key,value in original['components'].items())
        assert all(raw['pins'][key]==value for key,value in original['pins'].items())
    validate_target(data)
    expected=to_target(raw)
    name_audit_path=Path('/verify/logs')/(case+'_ic_name_selection.json')
    if name_audit_path.exists():
        names=json.loads(name_audit_path.read_text())
        for entry in names['components']:
            key=entry['component_id']
            assert expected['components'][key]['type'] in {'ic','box'}
            assert entry['before_Name']==expected['components'][key]['Name']
            expected['components'][key]['Name']=entry['after_Name']
    ic_audit_path=Path('/verify/logs')/(case+'_ic_geometry.json')
    if ic_audit_path.exists():
        ic_audit=json.loads(ic_audit_path.read_text())
        for entry in ic_audit['entries']:
            key=entry['component_id']
            assert expected['components'][key]['type'] in {'ic','box'}
            assert entry['before_bbox_bottom_left']==expected['components'][key]['bbox']
            expected['components'][key]['bbox']=entry['after_bbox_bottom_left']
    led_audit_path=Path('/verify/logs')/(case+'_led_geometry.json')
    if led_audit_path.exists():
        audit=json.loads(led_audit_path.read_text())
        for entry in audit['entries']:
            key=entry['component_id']
            assert expected['components'][key]['type']=='led'
            assert entry['before_bbox_bottom_left']==expected['components'][key]['bbox']
            expected['components'][key]['bbox']=entry['after_bbox_bottom_left']
    assert data==expected, 'Final output differs outside declared component, IC Name and IC/LED geometry policies'
    assert sum(len(pins) for pins in data['pins'].values())>0
    assert not any(c['type'] in {'resistor','capacitor','inductor','diode','ic'} for c in data['components'].values())
decision=json.loads(Path('/verify/decision_log.json').read_text())
assert decision['component_output_policy']==COMPONENT_OUTPUT_POLICY
print('Both PNG outputs passed Target layout, public component policy and reference validation.')
'@
Invoke-DockerLogged -CommandArguments @('run','--rm','--network','none','--entrypoint','python3',
    '--mount',"type=bind,source=$previousTestOutput,target=/verify,readonly",$ImageTag,'-c',$contractCheck) `
    -LogPath (Join-Path $buildRoot 'target_contract_recheck.log')

# 3. Export the verified image verbatim.
$tarPath = Join-Path $buildRoot 'submission.tar'
if (Test-Path -LiteralPath $tarPath) {
    Write-Host "Reusing the existing submission.tar; not saving the image again."
} else {
    Invoke-DockerLogged -CommandArguments @('image','save','--output',$tarPath,$ImageTag) -LogPath (Join-Path $buildRoot 'docker_save.log')
}
if ((Get-Item -LiteralPath $tarPath).Length -lt 1GB) { throw 'Saved TAR is implausibly small.' }

# 4. The archive must load back as the very same image.
if (-not $SkipLoadCheck) {
    Invoke-DockerLogged -CommandArguments @('image','load','--input',$tarPath) -LogPath (Join-Path $buildRoot 'docker_load.log')
    $loadedImageId = (& $dockerExecutable image inspect $ImageTag --format '{{.Id}}' | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $loadedImageId -ne $imageId) { throw "Archive did not load as $imageId (got $loadedImageId)." }
    $loadedPlatform = (& $dockerExecutable image inspect $ImageTag --format '{{.Os}}/{{.Architecture}}' | Out-String).Trim()
    $loadedEntrypoint = (& $dockerExecutable image inspect $ImageTag --format '{{json .Config.Entrypoint}}' | Out-String).Trim()
    if ($loadedPlatform -ne 'linux/amd64' -or $loadedEntrypoint -ne '["/submission/run.sh"]') { throw 'Reloaded image lost its platform or entrypoint.' }
}

# 5. ZIP with submission.tar as the only member. Python's zipfile writes a genuine
#    STORED member; .NET's CompressionLevel.NoCompression would emit DEFLATE
#    level-0 framing instead, growing an already-compressed payload by ~0.5 KB per MB.
$zipPath = Join-Path $buildRoot $ArchiveName
# Reuse a ZIP that is already a single stored member of this exact TAR, so a
# resumed run does not rewrite several gigabytes for an identical result.
$reuseZip = $false
if (Test-Path -LiteralPath $zipPath) {
    try {
        $existing = [IO.Compression.ZipFile]::OpenRead($zipPath)
        try {
            $member = if ($existing.Entries.Count -eq 1) { $existing.Entries[0] } else { $null }
            if ($member -and $member.FullName -eq 'submission.tar' -and
                $member.Length -eq (Get-Item -LiteralPath $tarPath).Length -and
                $member.CompressedLength -eq $member.Length) { $reuseZip = $true }
        } finally { $existing.Dispose() }
    } catch { $reuseZip = $false }
}
if ($reuseZip) {
    Write-Host 'Reusing the existing upload ZIP; it is already a stored member of this TAR.'
} else {
    & $pythonExecutable (Join-Path $PSScriptRoot 'pack_submission_zip.py') $tarPath $zipPath
    if ($LASTEXITCODE -ne 0) { throw 'Packing the upload ZIP failed.' }
}
$readBack = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    if ($readBack.Entries.Count -ne 1 -or $readBack.Entries[0].FullName -ne 'submission.tar' -or
        $readBack.Entries[0].Length -ne (Get-Item -LiteralPath $tarPath).Length) {
        throw 'Unexpected ZIP layout or uncompressed size.'
    }
} finally { $readBack.Dispose() }

# 6. Independently verify layout, CRC, digests and image identity. Redirect the
#    streams to files: with $ErrorActionPreference='Stop' a single stderr line
#    from Python becomes a terminating error and would hide the traceback.
$verifyLog = Join-Path $buildRoot 'archive_verification.json'
$verifyErr = Join-Path $buildRoot 'archive_verification.stderr.txt'
$previousPreference = $ErrorActionPreference
try {
    # The verifier writes its own UTF-8 report; only console output is captured
    # here, because a PowerShell `1>` redirect would encode the JSON as UTF-16.
    $ErrorActionPreference = 'Continue'
    $verifyOutput = & $pythonExecutable (Join-Path $buildRoot 'verify_submission_archive.py') $tarPath $zipPath `
        '--expect-image-id' $imageId '--out' $verifyLog 2>&1
    $verifyExit = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference }
if ($verifyExit -ne 0) {
    ($verifyOutput | Out-String) | Set-Content -LiteralPath $verifyErr -Encoding UTF8
    throw "Archive verification failed ($verifyExit). Inspect $verifyErr"
}
Remove-Item -LiteralPath $verifyErr -Force -ErrorAction SilentlyContinue
$verify = (Get-Content -LiteralPath $verifyLog -Raw) | ConvertFrom-Json
if (-not $verify.all_checks_passed) { throw 'Archive verification did not report a clean pass.' }
Write-Host ($verifyOutput | Out-String).Trim()
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$tarHash = (Get-FileHash -LiteralPath $tarPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($verify.zip.zip_sha256 -ne $zipHash -or $verify.tar.tar_sha256 -ne $tarHash) { throw 'Hash mismatch between the verifier and the host checksums.' }
$gpuVerification = Join-Path $projectRoot 'dist\submission_update_20261016\gpu_verification.json'
$gpu = if (Test-Path -LiteralPath $gpuVerification) { Get-Content -LiteralPath $gpuVerification -Raw | ConvertFrom-Json } else { $null }

$report = [ordered]@{
    export_kind = 'completed interrupted export from a previously verified image'
    image = $ImageTag; image_id = $imageId; platform = $imagePlatform; entrypoint = $entrypoint
    image_torch_runtime = '2.5.1+cu121 (official Ubuntu 24 / PyTorch official base)'
    rebuilt_from_scratch = $false
    previous_verified_build = $PreviousBuild
    previous_build_image_line = $previousTag
    source_zip = $sourceZipAbsolute; source_zip_sha256 = $sourceZipHash
    frozen_source_dir = $frozenSourceAbsolute
    image_matches_frozen_source = $true
    target_contract_rechecked = $true
    tar_sha256 = $tarHash; tar_bytes = (Get-Item -LiteralPath $tarPath).Length
    tar_members = $verify.tar.tar_members
    tar_layer_entries = $verify.tar.layer_entries
    tar_unique_layer_blobs = $verify.tar.unique_layer_blobs
    tar_layer_blob_digests_verified = $verify.tar.layer_blobs_verified
    tar_content_addressed_blobs_verified = $verify.tar.content_addressed_blobs_verified
    tar_oci_index_manifests = $verify.tar.oci_index_manifests
    tar_format = $verify.tar.tar_format
    archive_load_returned_same_image = (-not $SkipLoadCheck)
    upload_zip = $zipPath
    upload_zip_bytes = (Get-Item -LiteralPath $zipPath).Length
    upload_zip_sha256 = $zipHash
    upload_zip_members = $verify.zip.zip_entries
    zip_crc_verified = $verify.zip.zip_crc_verified
    cpu_verification = 'Two PNGs (0036, 0042) ran offline in this image with the Target contract re-checked against the recorded output.'
    gpu_verification = if ($gpu) {
        "Frozen source on the local RTX 5060 via the cu128 image: device=$($gpu.selected_device), cuda_probe.success=$($gpu.cuda_probe.success), cases=$($gpu.successful_cases)/$($gpu.cases.Count)."
    } else { 'Local GPU verification record not present.' }
    gpu_verification_is_not_official = $true
    official_score = $null
    remaining_limitations = @(
        'No A800 GPU was available: the uploaded cu121 image has not been executed on the competition GPU.',
        'The deployment needs a writable /submission/runs (Ultralytics predictor save_dir); a read-only /submission mount breaks inference.',
        'Public training/Golden IC Name conventions differ; the observed priority is not guaranteed to match every version.',
        'Network and pin coverage remain limited.',
        'Online IC/LED failing images are unavailable; the new official score is pending.'
    )
}
# Set-Content -Encoding UTF8 writes a BOM in Windows PowerShell, which makes the
# file fail a plain utf-8 read; write it explicitly without one.
[IO.File]::WriteAllText((Join-Path $buildRoot 'build_report.json'),
    ($report | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding $false))
Write-Host "READY TO UPLOAD: $zipPath"
Write-Host "SHA256: $zipHash"
Write-Host "Only upload $ArchiveName, whose sole member is submission.tar."