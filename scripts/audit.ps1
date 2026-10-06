<#
.SYNOPSIS
    Runs the whole quality and security audit and collects the results in one folder and one zip file.

.DESCRIPTION
    Runs, in this order: tool versions, ruff (lint and format check), mypy (source and tests), pytest with coverage,
    a compile check, bandit (security scan of our own code), pip-audit (known vulnerabilities in the installed
    libraries), the installed-library lists, and a check that uv.lock matches pyproject.toml.

    Each step writes its output to its own text file in  audit-results\<date-time>\  and the folder is also zipped
    to  audit-results\<date-time>.zip  ready to share. Nothing is changed in the project and nothing is deleted.

    Works in Windows PowerShell 5.1 and PowerShell 7. Run it from anywhere:

        powershell -ExecutionPolicy Bypass -File scripts\audit.ps1

.PARAMETER SkipTests
    Leave out pytest and the coverage report (the slowest step), for a quick look at everything else.

.PARAMETER NoZip
    Leave the results as a folder only.

.NOTES
    Why every step goes through cmd.exe: Windows PowerShell 5.1 writes redirected output (the ">" and "*>" operators)
    as UTF-16, which most text tools and chat windows cannot read. The old command prompt writes the bytes exactly as
    the program produced them, and PYTHONUTF8=1 makes Python produce UTF-8 on every Windows version.
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'

# The project folder is the one above this script, wherever the script is run from.
$project = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $project

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Which Python to use: the project's own environment (Windows layout first, then the Linux one).
$py = $null
foreach ($candidate in @('.venv\Scripts\python.exe', '.venv-linux\bin\python')) {
    if (Test-Path -LiteralPath $candidate) { $py = $candidate; break }
}
if (-not $py) {
    Write-Error 'No project environment found (.venv). Create it first, see the README, then run this again.'
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path 'audit-results' $stamp
New-Item -ItemType Directory -Path $out -Force | Out-Null

$steps = New-Object System.Collections.Generic.List[object]

function Invoke-Step {
    param(
        [string]$Name,
        [string]$File,       # the output file name inside the results folder
        [string]$Command     # a command line for cmd.exe; "{out}" stands for the results folder
    )
    $target = Join-Path $out $File
    $line = $Command.Replace('{out}', $out)
    Write-Host ('{0,-34}' -f $Name) -NoNewline
    # The extra outer quotes are how cmd /c keeps a command that itself contains quotes intact.
    cmd /c "$line > `"$target`" 2>&1"
    $code = $LASTEXITCODE
    $status = if ($code -eq 0) { 'ok' } else { "exit code $code" }
    $colour = if ($code -eq 0) { 'Green' } else { 'Yellow' }
    Write-Host $status -ForegroundColor $colour
    $steps.Add([pscustomobject]@{ Step = $Name; Result = $status; File = $File; Code = $code })
}

Write-Host "Audit of $project"
Write-Host "Results: $out"
Write-Host ''

Invoke-Step 'Tool versions' '00-versions.txt' `
    "($py --version & $py -m pip --version & $py -m ruff --version & $py -m mypy --version & $py -m pytest --version & $py -m bandit --version & $py -m pip_audit --version)"

Invoke-Step 'ruff: lint' '01-ruff-check.txt' "$py -m ruff check ."
Invoke-Step 'ruff: format check' '02-ruff-format.txt' "$py -m ruff format --check ."
Invoke-Step 'mypy: src and tests' '03-mypy.txt' "$py -m mypy src tests"

if (-not $SkipTests) {
    Invoke-Step 'pytest with coverage' '04-pytest-coverage.txt' "$py -m pytest -q --cov=lemonrind --cov-report=term-missing"
}

Invoke-Step 'compile check' '05-compileall.txt' "$py -m compileall -q src tests"

# bandit: -q hides its progress chatter, -o puts the report itself in a file. The text report is for reading,
# the JSON one for tools. Its exit code is 1 when it found something.
Invoke-Step 'bandit: security scan' '06-bandit-log.txt' `
    "$py -m bandit -q -c pyproject.toml -r src -f txt -o {out}\06-bandit.txt"
Invoke-Step 'bandit: json report' '06-bandit-json-log.txt' `
    "$py -m bandit -q -c pyproject.toml -r src -f json -o {out}\06-bandit.json"

Invoke-Step 'pip-audit: known vulnerabilities' '07-pip-audit.txt' "$py scripts\audit_dependencies.py"

Invoke-Step 'Installed libraries' '08-pip-list.txt' "$py -m pip list"
Invoke-Step 'Installed libraries (freeze)' '08-pip-freeze.txt' "$py -m pip freeze"

$uv = '.venv\Scripts\uv.exe'
if (Test-Path -LiteralPath $uv) {
    Invoke-Step 'uv.lock matches pyproject' '09-uv-lock-check.txt' "$uv lock --check --system-certs"
}

Copy-Item -LiteralPath 'pyproject.toml' -Destination (Join-Path $out '10-pyproject.toml')

# A short summary, written as UTF-8 without a byte-order mark so every tool can read it.
$summary = @("Audit run $stamp", '')
foreach ($step in $steps) { $summary += ('{0,-34}{1,-14}{2}' -f $step.Step, $step.Result, $step.File) }
[System.IO.File]::WriteAllLines((Join-Path (Get-Location) (Join-Path $out '00-summary.txt')), $summary)

Write-Host ''
if (-not $NoZip) {
    $zip = "$out.zip"
    Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zip -Force
    Write-Host "Zipped: $zip"
}

$failed = @($steps | Where-Object { $_.Code -ne 0 })
if ($failed.Count -eq 0) {
    Write-Host 'Every step passed.' -ForegroundColor Green
    exit 0
}
Write-Host ("{0} step(s) need a look: {1}" -f $failed.Count, (($failed | ForEach-Object { $_.Step }) -join ', ')) -ForegroundColor Yellow
exit 1
