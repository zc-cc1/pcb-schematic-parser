$ErrorActionPreference = 'Stop'
$projectRoot = 'C:\Users\Pc\Desktop\vision\pcb_parser_competition'
$testRoot = Join-Path $projectRoot 'dist\led_bbox_fix_20261016'
$testInput = Join-Path $testRoot 'container-input'
$testOutput = Join-Path $testRoot 'container-output'
New-Item -ItemType Directory -Path $testInput,$testOutput -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'data\standardized\0107\0107.png') -Destination (Join-Path $testInput '0107.png')
$dockerExecutable = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$dockerArguments = @('run','--rm','--network','none','--read-only',
    '--tmpfs','/tmp:rw,size=2g','--tmpfs','/submission/runs:rw,size=256m',
    '-e','PCB_DEVICE=cpu','-e','PYTHONPATH=/submission/src:/submission/scripts:/submission',
    '--mount',"type=bind,source=$testInput,target=/input,readonly",
    '--mount',"type=bind,source=$testOutput,target=/output")
$sources = [ordered]@{
    'submission\platform_adapter\agent.py'='/submission/agent.py'
    'submission\platform_adapter\adapter.py'='/submission/adapter.py'
    'submission\platform_adapter\component_export.py'='/submission/component_export.py'
    'src\pcb_parser\component_text.py'='/submission/src/pcb_parser/component_text.py'
    'src\pcb_parser\led_geometry.py'='/submission/src/pcb_parser/led_geometry.py'
}
$hashes = [ordered]@{}
foreach ($relative in $sources.Keys) {
    $source = Join-Path $projectRoot $relative
    $dockerArguments += @('--mount',"type=bind,source=$source,target=$($sources[$relative]),readonly")
    $hashes[$relative] = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
}
$hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $testRoot 'container_source_sha256.json') -Encoding utf8
$dockerArguments += @('pcb-parser:local-cuda-28da9146','-input','/input','-output','/output','--device','cpu','--workers','1')
& $dockerExecutable @dockerArguments 2>&1 | Tee-Object -FilePath (Join-Path $testRoot 'container_run.log')
if ($LASTEXITCODE -ne 0) { throw "Container verification failed: $LASTEXITCODE" }
