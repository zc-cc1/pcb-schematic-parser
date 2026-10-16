$ErrorActionPreference='Stop'
$projectRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$testRoot=$PSScriptRoot
$testInput=Join-Path $testRoot 'container-input'
$testOutput=Join-Path $testRoot 'container-output'
$compatOutput=Join-Path $testRoot 'compatibility-output'
New-Item -ItemType Directory -Path $testInput,$testOutput,$compatOutput -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'data\standardized\0036\0036.png') -Destination (Join-Path $testInput '0036.png')
$dockerExecutable='C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$sources=[ordered]@{
    'submission\platform_adapter\agent.py'='/submission/agent.py'
    'submission\platform_adapter\adapter.py'='/submission/adapter.py'
    'submission\platform_adapter\component_export.py'='/submission/component_export.py'
    'src\pcb_parser\component_text.py'='/submission/src/pcb_parser/component_text.py'
    'src\pcb_parser\ic_name_selection.py'='/submission/src/pcb_parser/ic_name_selection.py'
    'src\pcb_parser\ic_geometry.py'='/submission/src/pcb_parser/ic_geometry.py'
    'src\pcb_parser\led_geometry.py'='/submission/src/pcb_parser/led_geometry.py'
}
$hashes=[ordered]@{}
$mounts=@()
foreach ($relative in $sources.Keys) {
    $source=Join-Path $projectRoot $relative
    $mounts+=@('--mount',"type=bind,source=$source,target=$($sources[$relative]),readonly")
    $hashes[$relative]=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
}
$hashes | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $testRoot 'container_source_sha256.json') -Encoding utf8
$arguments=@('run','--rm','--network','none','--read-only',
    '--tmpfs','/tmp:rw,size=2g','--tmpfs','/submission/runs:rw,size=256m',
    '-e','PCB_DEVICE=cpu','-e','PYTHONPATH=/submission/src:/submission/scripts:/submission',
    '--mount',"type=bind,source=$testInput,target=/input,readonly",
    '--mount',"type=bind,source=$testOutput,target=/output")
$arguments+=$mounts
$arguments+=@('pcb-parser:local-cuda-28da9146','-input','/input','-output','/output','--device','cpu','--workers','1')
& $dockerExecutable @arguments 2>&1 | Tee-Object -FilePath (Join-Path $testRoot 'container_run.log')
if ($LASTEXITCODE -ne 0) { throw "PNG container test failed: $LASTEXITCODE" }
$arguments=@('run','--rm','--network','none','--read-only','--entrypoint','python3',
    '-e','PYTHONPATH=/submission/src:/submission',
    '--mount',"type=bind,source=$testRoot\compatibility-input,target=/fixtures,readonly",
    '--mount',"type=bind,source=$compatOutput,target=/output",
    '--mount',"type=bind,source=$testRoot\compatibility_check.py,target=/check.py,readonly")
$arguments+=$mounts
$arguments+=@('pcb-parser:local-cuda-28da9146','/check.py')
& $dockerExecutable @arguments 2>&1 | Tee-Object -FilePath (Join-Path $testRoot 'compatibility_run.log')
if ($LASTEXITCODE -ne 0) { throw "Image-only runtime check failed: $LASTEXITCODE" }
foreach ($relative in $hashes.Keys) {
    $actual=(Get-FileHash -LiteralPath (Join-Path $projectRoot $relative) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $hashes[$relative]) { throw "Source changed during validation: $relative" }
}
