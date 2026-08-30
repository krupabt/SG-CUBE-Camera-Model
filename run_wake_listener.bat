@echo off
title SG CUBE Background Wake Listener
cd /d "%~dp0"
if exist "%~dp0runtime\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0runtime\Scripts\python.exe"
) else if exist "%~dp0runtime\python.exe" (
    set "PYTHON_EXE=%~dp0runtime\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" wake_listener.py
