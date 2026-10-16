param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(4, 12)]
    [int]$Week,

    [ValidateRange(1, 6)]
    [int]$Day,

    [switch]$PrintOnly
)

$courseRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$weekPrefix = ([char]0x7B2C) + $Week + ([char]0x5468) + '_'
$weekFolder = Get-ChildItem -LiteralPath $courseRoot -Directory |
    Where-Object { $_.Name.StartsWith($weekPrefix) } |
    Select-Object -First 1

if (-not $weekFolder) {
    throw "Week $Week course folder was not found."
}

if (-not $PSBoundParameters.ContainsKey('Day')) {
    if ($PrintOnly) {
        Write-Output $weekFolder.FullName
    }
    else {
        Start-Process explorer.exe -ArgumentList $weekFolder.FullName
    }
    return
}

$dayPrefix = "{0:D2}_" -f $Day
$program = Get-ChildItem -LiteralPath $weekFolder.FullName -Recurse -Filter '*.hdev' |
    Where-Object {
        $_.Name.StartsWith($dayPrefix) -or
        ($Day -eq 6 -and $_.Name -eq 'main_project_template.hdev')
    } |
    Select-Object -First 1

if (-not $program) {
    throw "Week $Week day $Day HDevelop program was not found."
}

if ($PrintOnly) {
    Write-Output $program.FullName
    return
}

$hdevelopCandidates = @()
if ($env:HALCONROOT) {
    $hdevelopCandidates += Join-Path $env:HALCONROOT 'bin\x64-win64\hdevelop.exe'
}
$hdevelopCandidates += 'C:\Program Files\MVTec\HALCON-20.11-Steady\bin\x64-win64\hdevelop.exe'
$hdevelopCandidates = $hdevelopCandidates | Where-Object { Test-Path -LiteralPath $_ }

$hdevelop = $hdevelopCandidates | Select-Object -First 1
if (-not $hdevelop) {
    throw 'hdevelop.exe was not found. Check HALCONROOT or the HALCON installation.'
}

Start-Process -FilePath $hdevelop -ArgumentList ('"{0}"' -f $program.FullName)
Write-Output ("Opened: {0}" -f $program.FullName)
