@echo off
setlocal
set "DIR=%~dp0"
set "PYTHONPATH=%DIR%;%PYTHONPATH%"
if exist "%DIR%.venv\Scripts\python.exe" (
    "%DIR%.venv\Scripts\python.exe" -m mandible_rvc %*
) else if exist "%DIR%venv\Scripts\python.exe" (
    "%DIR%venv\Scripts\python.exe" -m mandible_rvc %*
) else (
    python -m mandible_rvc %*
)
