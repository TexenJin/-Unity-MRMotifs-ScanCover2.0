$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvRoot = Join-Path $ProjectRoot ".venv-scancover"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"

if (!(Test-Path -LiteralPath $PythonExe)) {
    throw "ScanCover Python environment not found: $PythonExe"
}

& $PythonExe @args
