$ErrorActionPreference='Stop'
# GPU / automatic-device validation for the frozen submission source.
#
# Mount rationale: /submission is bind-mounted read-only, which is the strict
# contract this test exists to prove. Ultralytics' predictor still creates its
# scratch directory `runs/detect/predict` relative to the process CWD, and the
# pipeline runs with cwd=/submission, so a writable `runs` entry is mandatory
# even for pure inference (without it: OSError [Errno 30] Read-only file system:
# 'runs'). A tmpfs can only be mounted there if the mountpoint already exists in
# the read-only bind source, so the test uses a staged copy of the frozen tree
# plus one empty `runs/` directory. The staged files are hash-verified against
# package_manifest.json before the run, which proves the tested code and weights
# are byte-identical to the frozen source.
#
# The result checks live in verify_gpu_outputs.py: PowerShell 5.1 cannot parse
# the IC Name / geometry audit files because they repeat JSON keys.
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$frozen=Join-Path $projectRoot 'submission\platform_test_20261016_icled_gpu_1'
$staging=Join-Path $PSScriptRoot 'gpu-source-staging'
$testInput=Join-Path $PSScriptRoot 'gpu-input'
$testOutput=Join-Path $PSScriptRoot 'gpu-output'
$verifier=Join-Path $PSScriptRoot 'verify_gpu_outputs.py'
$image='pcb-parser:local-cuda-68d6bcb8'
# 0036 exercises IC box geometry and IC Name priority; 0147 and 0107 add the LED
# box/reference stage (0107 is the 9-LED page used in the LED verification).
$cases=@('0036','0147','0107')

if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
Copy-Item -LiteralPath $frozen -Destination $staging -Recurse
New-Item -ItemType Directory -Path (Join-Path $staging 'runs') -Force | Out-Null

# The staged tree must be file-identical to the frozen package.
$manifest=Get-Content -LiteralPath (Join-Path $staging 'package_manifest.json') -Raw | ConvertFrom-Json
$verified=0
foreach ($property in $manifest.files.PSObject.Properties) {
    $relative=$property.Name
    $target=Join-Path $staging ($relative -replace '/','\')
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Staged file missing: $relative" }
    $item=Get-Item -LiteralPath $target
    if ($item.Length -ne $property.Value.bytes) { throw "Staged size mismatch: $relative" }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value.sha256) {
        throw "Staged hash mismatch: $relative"
    }
    $verified++
}
# package_manifest.json is the one file the manifest cannot describe for itself.
$stagedRelative=@(Get-ChildItem -LiteralPath $staging -Recurse -File | ForEach-Object {
    $_.FullName.Substring($staging.Length + 1) -replace '\\','/'
})
$described=@($manifest.files.PSObject.Properties | ForEach-Object { $_.Name })
$unexpected=@($stagedRelative | Where-Object { $_ -notin $described -and $_ -ne 'package_manifest.json' })
$missing=@($described | Where-Object { $_ -notin $stagedRelative })
if ($unexpected.Count -or $missing.Count) {
    throw "Staged tree differs from the manifest (unexpected: $($unexpected -join ', '); missing: $($missing -join ', '))."
}
Write-Host "Staged tree verified against package_manifest.json: $verified manifest files plus the manifest itself."

if (Test-Path -LiteralPath $testOutput) { Remove-Item -LiteralPath $testOutput -Recurse -Force }
New-Item -ItemType Directory -Path $testInput,$testOutput -Force | Out-Null
foreach ($caseId in $cases) {
    Copy-Item -LiteralPath (Join-Path $projectRoot "data\standardized\$caseId\$caseId.png") -Destination (Join-Path $testInput "$caseId.png")
}
$dockerExecutable='C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$arguments=@('run','--rm','--gpus','all','--network','none','--read-only',
    '--tmpfs','/tmp:rw,size=2g','--tmpfs','/submission/runs:rw,size=256m',
    '--entrypoint','python3','-e','PCB_DEVICE=auto',
    '--mount',"type=bind,source=$staging,target=/submission,readonly",
    '--mount',"type=bind,source=$testInput,target=/input,readonly",
    '--mount',"type=bind,source=$testOutput,target=/output",
    $image,'-u','/submission/agent.py',
    '-input','/input','-output','/output','--workers','1')
& $dockerExecutable @arguments 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'gpu_run.log')
if ($LASTEXITCODE -ne 0) { throw "GPU inference failed: $LASTEXITCODE" }

# The read-only mount must also mean the pipeline left its own tree untouched.
foreach ($property in $manifest.files.PSObject.Properties) {
    $target=Join-Path $staging ($property.Name -replace '/','\')
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value.sha256) {
        throw "Staged file changed during the run: $($property.Name)"
    }
}
Write-Host "Staged tree unchanged by the run: $verified files still match."

& python $verifier $testOutput (Join-Path $PSScriptRoot 'gpu_verification.json')
if ($LASTEXITCODE -ne 0) { throw 'GPU output verification failed.' }