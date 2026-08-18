@echo off
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv-scancover\Scripts\python.exe"
if not exist "%PY%" (
  echo [ScanCover] Missing Python environment: %PY%
  echo Run environment setup before using this viewer.
  exit /b 1
)
if "%~1"=="" (
  echo Usage:
  echo   %~nx0 path\to\file.ply [another.ply ...]
  echo.
  echo You can also drag .ply files onto this .cmd file.
  exit /b 2
)
"%PY%" "%~dp0ScanCoverViewPly.py" %*
