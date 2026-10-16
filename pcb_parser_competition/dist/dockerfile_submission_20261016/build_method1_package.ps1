param(
    [string]$Package = 'dist\dockerfile_submission_20261016\package',
    [string]$ReferenceOutput = 'D:\pcb_submission_export\pcb_image_20261016_203548_16454252\cpu_smoke_records',
    [string]$OutputRoot = 'dist\dockerfile_submission_20261016'
)
$ErrorActionPreference = 'Stop'
# Prove that 方式1 (platform builds from the Dockerfile) is viable: time a cold,
# cacheless build the way the platform will run it, then show the resulting image
# produces byte-identical predictions to the verified multi-GB prebuilt package.
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
$pkg = Join-Path $projectRoot $Package
$outRoot = Join-Path $projectRoot $OutputRoot
$dockerExecutable = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$tag = 'pcb-parser:method1-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$testInput = Join-Path $outRoot 'test-input'
$testOutput = Join-Path $outRoot 'test-output'
if (Test-Path -LiteralPath $testOutput) { Remove-Item -LiteralPath $testOutput -Recurse -Force }
New-Item -ItemType Directory -Path $testInput,$testOutput -Force | Out-Null
foreach ($caseId in @('0036','0042')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot "data\standardized\$caseId\$caseId.png") -Destination (Join-Path $testInput "$caseId.png")
}

Write-Host "Cold cacheless build of $tag (this is what the platform's ingestion does)..."
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $dockerExecutable build --no-cache --progress=plain --platform linux/amd64 -t $tag $pkg 2>&1 |
        ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $outRoot 'docker_build_cold.log')
    $buildExit = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference }
$stopwatch.Stop()
if ($buildExit -ne 0) { throw "Cold build failed ($buildExit). Inspect docker_build_cold.log" }
$buildSeconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 1)
Write-Host "Cold build finished in $buildSeconds seconds (platform cap is 1800 s)."

$imagePlatform = (& $dockerExecutable image inspect $tag --format '{{.Os}}/{{.Architecture}}' | Out-String).Trim()
$entrypoint = (& $dockerExecutable image inspect $tag --format '{{json .Config.Entrypoint}}' | Out-String).Trim()
if ($imagePlatform -ne 'linux/amd64') { throw 'Unexpected platform.' }
if ($entrypoint -ne '["/submission/run.sh"]') { throw 'Unexpected entrypoint.' }

# Offline inference, exactly as the platform runs it.
$previousPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $dockerExecutable run --rm --network none -e PCB_DEVICE=cpu `
        --mount "type=bind,source=$testInput,target=/input,readonly" `
        --mount "type=bind,source=$testOutput,target=/output" `
        $tag -input /input -output /output --workers 2 2>&1 |
        ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $outRoot 'docker_run.log')
    $runExit = $LASTEXITCODE
} finally { $ErrorActionPreference = $previousPreference }
if ($runExit -ne 0) { throw 'Inference failed. Inspect docker_run.log' }

# Byte-for-byte comparison against the verified prebuilt package's output.
$compared = @()
foreach ($caseId in @('0036','0042')) {
    $mine = Join-Path $testOutput "final_results\${caseId}_Pred.json"
    $theirs = Join-Path $ReferenceOutput "final_results\${caseId}_Pred.json"
    $mineHash = (Get-FileHash -LiteralPath $mine -Algorithm SHA256).Hash
    $theirHash = (Get-FileHash -LiteralPath $theirs -Algorithm SHA256).Hash
    if ($mineHash -ne $theirHash) { throw "Prediction differs for $caseId ($mineHash vs $theirHash)." }
    $compared += [ordered]@{ case = $caseId; sha256 = $mineHash.ToLowerInvariant(); identical_to_prebuilt = $true }
}
$decision = Get-Content -LiteralPath (Join-Path $testOutput 'final_results\decision_log.json') -Raw | ConvertFrom-Json

# The upload ZIP for 方式1: package files at the archive root.
$zipPath = Join-Path $outRoot 'submission.zip'
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::CreateFromDirectory($pkg, $zipPath, [IO.Compression.CompressionLevel]::Optimal, $false)
$entryNames = @()
$reader = [IO.Compression.ZipFile]::OpenRead($zipPath)
try { $entryNames = @($reader.Entries | ForEach-Object { $_.FullName }) } finally { $reader.Dispose() }
if ($entryNames -notcontains 'Dockerfile' -or $entryNames -notcontains 'agent.py' -or $entryNames -notcontains 'run.sh') {
    throw "submission.zip must hold Dockerfile/agent.py/run.sh at its root; got $($entryNames[0..([Math]::Min(3,$entryNames.Count-1))] -join ', ')"
}
$pythonExecutable = (Get-Command python -ErrorAction SilentlyContinue).Source
$sourceTreeCheck = & $pythonExecutable (Join-Path $outRoot 'compare_with_frozen_source.py')
if ($LASTEXITCODE -ne 0) { throw 'Package tree differs from the frozen source beyond the Dockerfile.' }

$report = [ordered]@{
    submission_mode = '方式1 平台构建 Dockerfile（不上传预构建镜像）'
    tag = $tag
    cold_build_seconds = $buildSeconds
    platform_build_cap_seconds = 1800
    build_headroom_seconds = 1800 - $buildSeconds
    base_image_reused_from_platform = 'edarace-agent-base:ubuntu24'
    image_platform = $imagePlatform; image_entrypoint = $entrypoint
    cpu_smoke_cases = @('0036','0042'); successful_cases = $decision.successful_cases; failed_cases = $decision.failed_cases
    predictions_identical_to_prebuilt_package = $true
    prediction_comparison = $compared
    upload_zip = $zipPath
    upload_zip_bytes = (Get-Item -LiteralPath $zipPath).Length
    upload_zip_sha256 = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    upload_zip_root_entries = $entryNames.Count
    package_tree_vs_frozen_source = $sourceTreeCheck
    limitations = @(
        '本地构建时长含本地带宽，平台侧只应作为量级参考，不是保证；护栏是 constraints 里不下载 torch。',
        '未实测该镜像在 A800 上跑 GPU；CPU 推理已用断网容器验证。',
        '平台构建器是否支持 BuildKit 未知，因此配方里去掉了 --mount=type=cache 这类 BuildKit 专属语法。'
    )
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outRoot 'build_report.json') -Encoding UTF8
Write-Host "ZIP: $zipPath  $((Get-Item -LiteralPath $zipPath).Length) bytes"
Write-Host "Cold build: $buildSeconds s / 1800 s cap"