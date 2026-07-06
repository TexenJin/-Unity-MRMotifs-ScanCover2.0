@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "PYTHON_EXE=%PROJECT_ROOT%\.venv-scancover\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo ScanCover Python environment not found: %PYTHON_EXE%
  exit /b 1
)
"%PYTHON_EXE%" %*
