# PowerShell shim so Windows users get the same UX as `make` without needing make.
#
#   .\make.ps1 doctor
#   .\make.ps1 test LAB=1 NPROC=4
#
# Uses the repo venv interpreter when it exists, otherwise whatever `python`
# resolves to on PATH.

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $py = $venvPython
} else {
    $py = "python"
}

& $py (Join-Path $root "tasks.py") @args
exit $LASTEXITCODE
