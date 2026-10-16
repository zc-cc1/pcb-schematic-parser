param(
    [ValidateSet('run', 'build')]
    [string]$Mode = 'run'
)

$ErrorActionPreference = 'Stop'
$compiler = 'C:\msys64\ucrt64\bin\g++.exe'
if (-not (Test-Path -LiteralPath $compiler)) {
    throw "Compiler not found: $compiler"
}

# Use relative paths for GCC/ld, which may mishandle Chinese absolute paths.
$previousPath = $env:PATH
$resultCode = 0
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:PATH = (Split-Path -Parent $compiler) + ';' + $previousPath
    New-Item -ItemType Directory -Force -Path 'build_basics_vscode' | Out-Null
    & $compiler '-std=c++17' '-Wall' '-Wextra' '-g' 'basics.cpp' '-o' 'build_basics_vscode/basics.exe'
    if ($LASTEXITCODE -ne 0) {
        throw "Compilation failed (exit code $LASTEXITCODE). See compiler output above."
    }
    Write-Host 'Build succeeded.'
    if ($Mode -eq 'run') {
        & '.\build_basics_vscode\basics.exe'
        $resultCode = $LASTEXITCODE
        if ($resultCode -ne 0) {
            throw "Program failed (exit code $resultCode)."
        }
        Write-Host 'Program completed successfully (exit code 0).'
    }
}
catch {
    Write-Host $_ -ForegroundColor Red
    $resultCode = 1
}
finally {
    Pop-Location
    $env:PATH = $previousPath
}
exit $resultCode
