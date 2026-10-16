$ErrorActionPreference = 'Stop'
$cmake = 'C:\Program Files\CMake\bin\cmake.exe'
$ctest = 'C:\Program Files\CMake\bin\ctest.exe'
$previousPath = $env:PATH
$resultCode = 0
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath $cmake)) { throw "CMake not found: $cmake" }
    $env:PATH = 'C:\msys64\ucrt64\bin;' + $previousPath
    & $cmake --preset basics
    if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
    & $cmake --build --preset basics
    if ($LASTEXITCODE -ne 0) { throw 'CMake build failed.' }
    & $ctest --preset basics
    if ($LASTEXITCODE -ne 0) { throw 'CTest failed.' }
    & '../../.build/week01-basics/basics.exe'
    if ($LASTEXITCODE -ne 0) { throw 'basics.exe failed.' }
    Write-Host 'CMake build, test and run succeeded. OpenCV was not required.'
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
