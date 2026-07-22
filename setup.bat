@echo off
REM Gen2Train main app installer. Installs torch/accelerate/transformers/diffusers/PySide6/
REM psutil etc. needed for training and the UI. Reuses the kohya_ss shared venv if it works,
REM otherwise creates this project's own venv (Gen2Train\venv). run.bat runs this
REM automatically when required packages are missing, so normally you only need run.bat.
setlocal

set "GEN2TRAIN_DIR=%~dp0"
set "PYTHON_EXE="
set "PYTHON_ARGS="

py -3.11 -c "1" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3.11"
)

if not defined PYTHON_EXE (
    py -3 -c "1" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo [setup] Could not find a Python interpreter. Install Python 3.11+ and try again.
    pause
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% "%GEN2TRAIN_DIR%setup.py"

pause
endlocal
