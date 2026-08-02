@echo off
rem cmd.exe shim: `make doctor` works from a plain Command Prompt in this folder.
setlocal
set "REPO=%~dp0"
if exist "%REPO%.venv\Scripts\python.exe" (
    "%REPO%.venv\Scripts\python.exe" "%REPO%tasks.py" %*
) else (
    python "%REPO%tasks.py" %*
)
exit /b %ERRORLEVEL%
