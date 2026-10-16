param(
    [ValidateSet('cuda', 'cpu')][string]$Runtime = 'cuda',
    [string]$OutputDirectory,
    [string]$SourceZip,
    [string]$BaseImage,
    [string]$BuildRecipe,
    [ValidateSet('auto','cpu')][string]$SmokeDevice = 'auto',
    [ValidatePattern('^[A-Za-z0-9_-]+\.zip$')][string]$ArchiveName = 'submission_target.zip',
    [ValidateRange(1,3)][int]$SmokeWorkers = 2,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $SourceZip) { $SourceZip = Join-Path $projectRoot 'dist\submission_y11s_main_source.zip' }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $projectRoot 'dist\local_image_exports' }

$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCommand) {
    throw 'Docker CLI is unavailable. Install WSL 2 and Docker Desktop, start Docker Desktop, then open a new PowerShell window.'
}
$script:dockerExecutable = $dockerCommand.Source
$engine = & $script:dockerExecutable info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable. Start Docker Desktop and wait until its engine is running.' }
if (($engine | Out-String).Trim() -ne 'linux') { throw 'Linux containers are required. Switch Docker Desktop to Linux containers.' }
if ($CheckOnly) { Write-Host 'Docker Linux engine is ready.'; exit 0 }
if (-not (Test-Path -LiteralPath $SourceZip -PathType Leaf)) { throw "Source ZIP not found: $SourceZip" }
$sourceZipAbsolute = (Resolve-Path -LiteralPath $SourceZip).Path

# Always use a new directory; earlier exports and test logs are retained.
$buildName = 'pcb_image_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$buildRoot = [IO.Path]::GetFullPath((Join-Path $OutputDirectory $buildName))
New-Item -ItemType Directory -Path $buildRoot -ErrorAction Stop | Out-Null
$sourceSnapshot = Join-Path $buildRoot 'source_snapshot.zip'
Copy-Item -LiteralPath $sourceZipAbsolute -Destination $sourceSnapshot
$sourceZipHash = (Get-FileHash -LiteralPath $sourceSnapshot -Algorithm SHA256).Hash
$context = Join-Path $buildRoot 'source'
$testInput = Join-Path $buildRoot 'test-input'
$testOutput = Join-Path $buildRoot 'test-output'
New-Item -ItemType Directory -Path $context,$testInput,$testOutput | Out-Null
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::ExtractToDirectory($sourceSnapshot, $context)
# Ensure an extracted same-size manifest cannot retain a previous context's
# metadata in Docker's incremental file transfer cache.
$contextTime = [DateTime]::UtcNow
Get-ChildItem -LiteralPath $context -Recurse -File | ForEach-Object { $_.LastWriteTimeUtc = $contextTime }
foreach ($fileName in @('agent.py','run.sh','requirements.txt','package_manifest.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $context $fileName))) {
        throw "Source ZIP must contain $fileName at its root."
    }
}
foreach ($caseId in @('0036','0042')) {
    $sourceImage = Join-Path $projectRoot ("data\standardized\$caseId\$caseId.png")
    if (-not (Test-Path -LiteralPath $sourceImage)) { throw "Smoke-test PNG is missing: $sourceImage" }
    Copy-Item -LiteralPath $sourceImage -Destination (Join-Path $testInput "$caseId.png")
}

function Invoke-DockerLogged {
    param([string[]]$CommandArguments, [string]$LogPath)
    # Docker writes progress to stderr. Capture it without treating successful
    # progress output as a terminating PowerShell 5 NativeCommandError.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $script:dockerExecutable @CommandArguments 2>&1 |
            ForEach-Object { $_.ToString() } | Tee-Object -FilePath $LogPath
        $commandExitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($commandExitCode -ne 0) { throw "Docker command failed ($commandExitCode). Inspect $LogPath" }
}

$tag = 'pcb-parser:local-' + $Runtime + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$index = if ($Runtime -eq 'cuda') { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
$recipe = Join-Path $PSScriptRoot 'Dockerfile.local'
if ($BuildRecipe) { $recipe = (Resolve-Path -LiteralPath $BuildRecipe).Path }
$extraBuildArguments = @()
if ($BaseImage) {
    # Overlay only deployment files when the runtime, models and image pipeline
    # match the tested base exactly; never silently mix different algorithms.
    $baseManifestText = (& $script:dockerExecutable run --rm --network none --entrypoint cat $BaseImage /submission/package_manifest.json | Out-String)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the requested base-image manifest.' }
    $baseManifest = $baseManifestText | ConvertFrom-Json
    $newManifest = Get-Content -LiteralPath (Join-Path $context 'package_manifest.json') -Raw | ConvertFrom-Json
    $allowedChanges = @('agent.py','adapter.py','README.md','Dockerfile')
    $allPaths = @($baseManifest.files.PSObject.Properties.Name) + @($newManifest.files.PSObject.Properties.Name) | Sort-Object -Unique
    foreach ($relativePath in $allPaths) {
        if ($relativePath -in $allowedChanges) { continue }
        $oldFile = $baseManifest.files.PSObject.Properties[$relativePath]
        $newFile = $newManifest.files.PSObject.Properties[$relativePath]
        if (-not $oldFile -or -not $newFile -or $oldFile.Value.sha256 -ne $newFile.Value.sha256) {
            throw "Runtime file differs from base: $relativePath. Use a full build instead."
        }
    }
    $recipe = Join-Path $PSScriptRoot 'Dockerfile.patch'
    $extraBuildArguments = @('--build-arg',"BASE_IMAGE=$BaseImage",'--pull=false')
}
$savedRecipe = Join-Path $buildRoot 'Dockerfile.used'
Copy-Item -LiteralPath $recipe -Destination $savedRecipe
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $buildRoot 'build_test_export.ps1')
Write-Host "Building $tag. The first local download can take time; full output will be saved."
$buildArguments = @('build','--platform','linux/amd64','--progress=plain',
    '--build-arg',"TORCH_INDEX_URL=$index",'-f',$savedRecipe,'-t',$tag) + $extraBuildArguments + @($context)
Invoke-DockerLogged -CommandArguments $buildArguments -LogPath (Join-Path $buildRoot 'docker_build.log')

$imagePlatform = (& $script:dockerExecutable image inspect $tag --format '{{.Os}}/{{.Architecture}}' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $imagePlatform -ne 'linux/amd64') { throw 'Expected a linux/amd64 image.' }
$entrypoint = (& $script:dockerExecutable image inspect $tag --format '{{json .Config.Entrypoint}}' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $entrypoint -ne '["/submission/run.sh"]') { throw 'Image entrypoint must be /submission/run.sh.' }

$manifestCheck = @'
import hashlib,json
from pathlib import Path
root=Path('/submission')
actual=(root/'package_manifest.json').read_bytes()
assert actual==Path('/expected/package_manifest.json').read_bytes(), 'Image manifest differs from source snapshot'
manifest=json.loads(actual)
for relative,expected in manifest['files'].items():
    data=(root/relative).read_bytes()
    assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'], relative
print('Source manifest and all bundled file hashes match the built image.')
'@
Invoke-DockerLogged -CommandArguments @('run','--rm','--network','none','--entrypoint','python3',
    '--mount',"type=bind,source=$context,target=/expected,readonly",$tag,'-c',$manifestCheck) -LogPath (Join-Path $buildRoot 'image_manifest_check.log')

# Disable the container network to verify that inference uses bundled models.
$runArguments = @('run','--rm','--network','none')
if ($Runtime -eq 'cuda' -and $SmokeDevice -ne 'cpu') { $runArguments += @('--gpus','all') }
else { $runArguments += @('-e','PCB_DEVICE=cpu') }
$runArguments += @('--mount',"type=bind,source=$testInput,target=/input,readonly",
    '--mount',"type=bind,source=$testOutput,target=/output",$tag,'-input','/input','-output','/output','--workers',"$SmokeWorkers")
Invoke-DockerLogged -CommandArguments $runArguments -LogPath (Join-Path $buildRoot 'docker_run.log')
$decision = Get-Content -LiteralPath (Join-Path $testOutput 'final_results\decision_log.json') -Raw | ConvertFrom-Json
if ($decision.successful_cases -ne 2 -or $decision.failed_cases -ne 0) { throw 'Both smoke-test cases must succeed before export.' }
if ($Runtime -eq 'cuda' -and $SmokeDevice -ne 'cpu' -and $decision.device -notlike 'cuda*') { throw 'CUDA image did not use a GPU. Check Docker Desktop GPU support.' }
if ($decision.output_format -ne 'target') { throw 'Deployment must default to the live Target protocol.' }
if ($decision.workers -ne [math]::Min($SmokeWorkers,2)) { throw 'Requested concurrent smoke test did not run.' }
foreach ($caseId in @('0036','0042')) {
    $prediction = Get-Content -LiteralPath (Join-Path $testOutput "final_results\${caseId}_Pred.json") -Raw | ConvertFrom-Json
    if (@($prediction.components.PSObject.Properties).Count -eq 0) { throw "No components produced for $caseId." }
}
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
    '--mount',"type=bind,source=$testOutput,target=/verify,readonly",$tag,'-c',$contractCheck) -LogPath (Join-Path $buildRoot 'target_contract.log')

$tarPath = Join-Path $buildRoot 'submission.tar'
Invoke-DockerLogged -CommandArguments @('image','save','--output',$tarPath,$tag) -LogPath (Join-Path $buildRoot 'docker_save.log')
$savedImageId = (& $script:dockerExecutable image inspect $tag --format '{{.Id}}' | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the image before archive verification.' }
Invoke-DockerLogged -CommandArguments @('image','load','--input',$tarPath) -LogPath (Join-Path $buildRoot 'docker_load.log')
$loadedImageId = (& $script:dockerExecutable image inspect $tag --format '{{.Id}}' | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $loadedImageId -ne $savedImageId) { throw 'Exported archive did not load as the original image.' }
$zipPath = Join-Path $buildRoot $ArchiveName
# A genuine STORED member, written by Python. .NET's ZipArchive maps
# CompressionLevel.NoCompression to DEFLATE level 0, which re-frames the already
# compressed `docker save` payload as deflate blocks and adds ~0.5 KB per MB;
# Compress-Archive cannot handle a 2-GB member at all.
$pythonExecutable = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExecutable) { throw 'Python is unavailable on the host; packing the upload ZIP needs it.' }
& $pythonExecutable (Join-Path $PSScriptRoot 'pack_submission_zip.py') $tarPath $zipPath
if ($LASTEXITCODE -ne 0) { throw 'Packing the upload ZIP failed.' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$readBack = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    if ($readBack.Entries.Count -ne 1 -or $readBack.Entries[0].FullName -ne 'submission.tar' -or
        $readBack.Entries[0].Length -ne (Get-Item -LiteralPath $tarPath).Length) {
        throw 'Unexpected ZIP layout or uncompressed size.'
    }
} finally { $readBack.Dispose() }
$report = [ordered]@{
    image=$tag; platform=$imagePlatform; runtime=$Runtime; entrypoint=$entrypoint
    source_zip=$sourceZipAbsolute; source_zip_snapshot=$sourceSnapshot; source_zip_sha256=$sourceZipHash
    docker_build_passed=$true; docker_inference_passed=$true; docker_archive_load_passed=$true
    source_manifest_matches_image=$true; all_bundled_file_hashes_verified=$true
    image_id=$loadedImageId; successful_cases=$decision.successful_cases
    output_format=$decision.output_format; workers_tested=$decision.workers; target_layout_verified=$true
    smoke_device=$decision.device; gpu_inference_tested=($decision.device -like 'cuda*')
    base_image=$BaseImage; build_recipe_sha256=(Get-FileHash -LiteralPath $savedRecipe -Algorithm SHA256).Hash
    type_mapping_status='six aliases verified in submission 8180 and public labels'
    component_output_policy=$decision.component_output_policy
    remaining_limitations=@('Public training/Golden IC Name conventions differ; observed model/reference/description priority is not guaranteed to match all versions','Network and pin coverage remain limited','Online IC/LED failing images are unavailable; new official score pending','CUDA use requires platform GPU visibility and compatible drivers',"Local smoke device: $($decision.device)")
    token_usage=$decision.token_usage; upload_zip=$zipPath
    upload_zip_sha256=(Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
    upload_zip_bytes=(Get-Item -LiteralPath $zipPath).Length; official_score=$null
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $buildRoot 'build_report.json') -Encoding UTF8
Write-Host "READY TO UPLOAD: $zipPath"
Write-Host "Build log: $(Join-Path $buildRoot 'docker_build.log')"
Write-Host "Only upload $ArchiveName, whose sole member is submission.tar."
