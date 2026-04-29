@echo off
setlocal
cd /d "%~dp0"
title claw-memory-brain memory service
set "PYTHON_EXE="
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".\.venv312\Scripts\python.exe" set "PYTHON_EXE=.\.venv312\Scripts\python.exe"
if not defined PYTHON_EXE if exist ".\.venv\Scripts\python.exe" set "PYTHON_EXE=.\.venv\Scripts\python.exe"

if defined PYTHON_EXE (
  "%PYTHON_EXE%" -u memory_service.py --run-on-start %*
) else (
  py -3 -u memory_service.py --run-on-start %*
)
endlocal
