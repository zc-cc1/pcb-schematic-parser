# Build, verify and export the competition submission image.
#
# The submission contract has two accepted shapes; this script produces both from
# one source tree:
#   * a prebuilt image tar (submission.tar inside a zip) - what the 5th evaluation ran
#   * the staged build context, if the platform prefers to build a Dockerfile itself
#
# Preconditions it checks instead of assuming:
#   * the Docker daemon is reachable (Windows: Docker Desktop must be running)
#   * the base image edarace-agent-base:ubuntu24 is loaded (auto-loads the local tar)
#   * the frozen weights and the configs the entry point references exist
#
# Usage (from pcb_parser_competition):
#   .\scripts\build_submission.ps1                     # build + save + zip
#   .\scripts\build_submission.ps1 -Smoke              # also run one case in the container
#   .\scripts\build_submission.ps1 -SkipSave           # build only (fast iteration)

[CmdletBinding()]
param(
    [string]$Tag = "pcb-parser:v4",
    [string]$BaseTar = "..\edarace-agent-base_ubuntu24.tar",
    [string]$BaseImage = "edarace-agent-base:ubuntu24",
    [switch]$SkipSave,
    [string]$OutDir = "",
    [switch]$Smoke,
    [string]$SmokeCase = "0002"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$submission = Join-Path $root "submission"
$build = Join-Path $submission "build"
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $submission "out" }
$out = $OutDir

function Step($message) { Write-Host "`n=== $message ===" -ForegroundColor Cyan }
function Fail($message) { Write-Host "ERROR: $message" -ForegroundColor Red; exit 1 }

Step "preflight: docker"
# stderr from a native command becomes an error record under PS 5.1; silence it
# here and rely on the exit code so the message below is the one users see.
$serverVersion = (docker version --format '{{.Server.Version}}' 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    Fail @"
Docker daemon is not reachable. Start Docker Desktop, wait for "Engine running", then re-run.
  Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
"@
}
Write-Host ("docker server " + $serverVersion)

Step "preflight: base image $BaseImage"
docker image inspect $BaseImage 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    if (-not (Test-Path $BaseTar)) { Fail "base image $BaseImage missing and $BaseTar not found" }
    Write-Host "loading $BaseTar (a few GB, one time)..."
    docker load -i $BaseTar
    if ($LASTEXITCODE -ne 0) { Fail "docker load failed" }
}

Step "preflight: frozen artefacts"
$cfg = Get-Content (Join-Path $root "configs\baseline_v3.json") -Raw | ConvertFrom-Json
$weights = Join-Path $root $cfg.weights
if (-not (Test-Path $weights)) { Fail "weights missing: $weights" }
foreach ($f in "configs\bbox_convention_v1.json", "configs\export_vocabulary.json",
               "configs\class_thresholds.json", "scripts\run_baseline.py") {
    if (-not (Test-Path (Join-Path $root $f))) { Fail "missing $f" }
}
Write-Host ("weights  " + (Split-Path $weights -Leaf) + "  " + [math]::Round((Get-Item $weights).Length/1MB,1) + " MB")
Write-Host ("geometry " + $cfg.bbox_convention.mode + " / vocabulary " + $cfg.export_vocabulary.path)

Step "preflight: offline wheel set"
$wheelDir = Join-Path $submission "out\wheels"
if (-not (Test-Path $wheelDir)) {
    Fail @"
No wheel set at $wheelDir. The Docker VM on this workstation cannot reach any package
index (TCP connects, TLS ClientHello is dropped), so the image is built offline.
Fetch the wheels on the host first:
  .venv-gpu\Scripts\python.exe scripts\fetch_submission_wheels.py
"@
}
$wheels = Get-ChildItem $wheelDir -Filter *.whl
if ($wheels.Count -lt 30) { Fail "wheel set looks incomplete: $($wheels.Count) wheels" }
Write-Host ("wheels  " + $wheels.Count + "  " + [math]::Round((($wheels | Measure-Object Length -Sum).Sum)/1MB,1) + " MB")

$modelSrc = Join-Path (Split-Path $root -Parent) "vision\pcb_parser_competition\.venv-gpu\Lib\site-packages\rapidocr\models"
if (-not (Test-Path $modelSrc)) {
    # fall back to the project-local venv junction, if that is how this copy is set up
    $modelSrc = Join-Path $root ".venv-gpu\Lib\site-packages\rapidocr\models"
}
if (-not (Test-Path $modelSrc)) { Fail "RapidOCR model directory not found (looked in $modelSrc)" }
$onnx = Get-ChildItem $modelSrc -Filter *.onnx
if ($onnx.Count -lt 3) { Fail "expected 3 RapidOCR models, found $($onnx.Count) in $modelSrc" }
Write-Host ("ocr     " + $onnx.Count + " models  " + [math]::Round((($onnx | Measure-Object Length -Sum).Sum)/1MB,1) + " MB")

Step "stage build context -> $build"
if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory -Path $build | Out-Null
foreach ($dir in "src", "scripts", "configs", "weights") {
    Copy-Item (Join-Path $root $dir) (Join-Path $build $dir) -Recurse
}
Copy-Item (Join-Path $submission "Dockerfile") $build
Copy-Item (Join-Path $submission "agent.py") $build
# run.sh must keep LF endings; copy explicitly as bytes and note the line ending
$runSh = [IO.File]::ReadAllBytes((Join-Path $submission "run.sh"))
[IO.File]::WriteAllBytes((Join-Path $build "run.sh"), $runSh)
# offline install inputs
Copy-Item $wheelDir (Join-Path $build "wheels") -Recurse
New-Item -ItemType Directory -Path (Join-Path $build "ocr_models") | Out-Null
Copy-Item (Join-Path $modelSrc "*.onnx") (Join-Path $build "ocr_models")
Copy-Item (Join-Path $submission "out\requirements-extras.txt") $build -ErrorAction SilentlyContinue
Write-Host ("staged files: " + (Get-ChildItem $build -Recurse -File).Count +
            ", size " + [math]::Round(((Get-ChildItem $build -Recurse -File | Measure-Object Length -Sum).Sum)/1MB,1) + " MB")

Step "docker build $Tag"
docker build -t $Tag $build
if ($LASTEXITCODE -ne 0) { Fail "docker build failed" }

if ($Smoke) {
    Step "smoke: run one case inside the container"
    $tmp = Join-Path $env:TEMP ("pcb_docker_smoke_" + [guid]::NewGuid().ToString("N").Substring(0,8))
    New-Item -ItemType Directory -Path (Join-Path $tmp "input\$SmokeCase") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $tmp "output") -Force | Out-Null
    Copy-Item (Join-Path $root "data\standardized\$SmokeCase\$SmokeCase.png") (Join-Path $tmp "input\$SmokeCase\$SmokeCase.png")
    $mountIn = $tmp.Replace('\', '/')
    # CPU on purpose: the base image ships torch 2.5.1+cu121, and this workstation's
    # RTX 5060 is sm_120, which that CUDA build does not support. The scoring host is
    # an A800 (sm_80), which cu121 does support. Running the smoke on CPU validates
    # the image (deps, OCR weights, config path, output contract) without pretending
    # to validate the GPU path.
    docker run --rm -v "${mountIn}/input:/input:ro" -v "${mountIn}/output:/output:rw" $Tag -input /input -output /output -limit 1 -device cpu
    if ($LASTEXITCODE -ne 0) { Fail "container run failed" }
    $pred = Join-Path $tmp "output\final_results\${SmokeCase}_Pred.json"
    if (-not (Test-Path $pred)) { Fail "container produced no $SmokeCase_Pred.json" }
    $n = (Get-Content $pred -Raw | ConvertFrom-Json).components.PSObject.Properties.Count
    Write-Host ("container output: $n components in $pred")
    if (-not (Test-Path (Join-Path $tmp "output\final_results\decision_log.json"))) {
        Fail "container produced no decision_log.json"
    }
}

if ($SkipSave) {
    Step "done (image built, not exported)"
    Write-Host "docker run --rm -v <in>:/input:ro -v <out>:/output:rw $Tag -input /input -output /output"
    exit 0
}

Step "export image tar"
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
$imageGb = [double]((docker image inspect $Tag --format '{{.Size}}' | Select-Object -First 1))
$imageGb = [math]::Round($imageGb/1GB, 2)
$drive = (Get-Item $out).PSDrive
Write-Host ("image " + $imageGb + " GB; target drive " + $drive.Name + " has " + [math]::Round($drive.Free/1GB,1) + " GB free")
# the tar plus its zip needs roughly 1.7x the image size
if ($drive.Free/1GB -lt ($imageGb * 1.7)) {
    Fail ("not enough space on " + $drive.Name + ": need about " + [math]::Round($imageGb*1.7,1) +
          " GB for the tar and zip, have " + [math]::Round($drive.Free/1GB,1) + " GB. " +
          "Use -OutDir <path on a larger drive>.")
}
$tarName = ($Tag -replace '[:/]', '-') + ".tar"
$tar = Join-Path $out $tarName
docker save $Tag -o $tar
if ($LASTEXITCODE -ne 0) { Fail "docker save failed" }
$sizeGb = [math]::Round((Get-Item $tar).Length/1GB, 2)
Write-Host "saved $tar ($sizeGb GB)"
$sha = (Get-FileHash $tar -Algorithm SHA256).Hash
"$sha  $tarName" | Set-Content (Join-Path $out "$tarName.sha256")

Step "compress for upload"
$zip = Join-Path $out ([IO.Path]::GetFileNameWithoutExtension($tarName) + ".zip")
if (Test-Path $zip) { Remove-Item $zip -Force }
if (Get-Command tar -ErrorAction SilentlyContinue) {
    Push-Location (Split-Path $tar -Parent)
    tar -a -cf (Split-Path $zip -Leaf) (Split-Path $tar -Leaf)
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Fail "tar -a zip failed (exit $code)" }
} else {
    Compress-Archive -Path $tar -DestinationPath $zip -CompressionLevel Optimal
}
Write-Host ("zip " + [math]::Round((Get-Item $zip).Length/1MB,1) + " MB -> $zip")

Step "summary"
[pscustomobject]@{
    image    = $Tag
    tar      = $tar
    zip      = $zip
    sha256   = $sha
    evidence = "ENTRYPOINT must be /submission/run.sh; outputs /output/final_results/<case>_Pred.json"
} | Format-List
