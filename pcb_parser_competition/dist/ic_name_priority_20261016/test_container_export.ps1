$ErrorActionPreference='Stop'
$projectRoot='C:\Users\Pc\Desktop\vision\pcb_parser_competition'
$testRoot=Join-Path $projectRoot 'dist\ic_name_priority_20261016'
$testOutput=Join-Path $testRoot 'container-output'
New-Item -ItemType Directory -Path $testOutput -Force | Out-Null
$dockerExecutable='C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$arguments=@('run','--rm','--network','none','--read-only','--entrypoint','python3',
    '-e','PYTHONPATH=/submission/src:/submission',
    '--mount',"type=bind,source=$testRoot\container-input,target=/fixtures,readonly",
    '--mount',"type=bind,source=$testOutput,target=/output",
    '--mount',"type=bind,source=$testRoot\container_export_check.py,target=/check.py,readonly")
$sources=[ordered]@{
    'submission\platform_adapter\agent.py'='/submission/agent.py'
    'submission\platform_adapter\adapter.py'='/submission/adapter.py'
    'src\pcb_parser\ic_name_selection.py'='/submission/src/pcb_parser/ic_name_selection.py'
    'src\pcb_parser\component_text.py'='/submission/src/pcb_parser/component_text.py'
}
foreach ($relative in $sources.Keys) {
    $source=Join-Path $projectRoot $relative
    $arguments+=@('--mount',"type=bind,source=$source,target=$($sources[$relative]),readonly")
}
$arguments+=@('pcb-parser:local-cuda-28da9146','/check.py')
& $dockerExecutable @arguments 2>&1 | Tee-Object -FilePath (Join-Path $testRoot 'container_export.log')
if ($LASTEXITCODE -ne 0) { throw "Container export check failed: $LASTEXITCODE" }
