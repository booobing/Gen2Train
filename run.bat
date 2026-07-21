@echo off
REM Gen2Train launcher.
REM Tries, in order: (1) the kohya_ss sibling venv this project was designed to reuse
REM (already has torch/accelerate/transformers/PySide6 installed), (2) the Windows
REM "py" launcher for any installed Python 3.11/3.x, (3) whatever "python" is on PATH.
setlocal

set "GEN2TRAIN_DIR=%~dp0"
set "PYTHON_EXE="
set "PYTHON_ARGS="

set "SHARED_VENV=%GEN2TRAIN_DIR%..\kohya_ss\kohya_ss\venv"
if exist "%SHARED_VENV%\Scripts\python.exe" (
    set "PYTHON_EXE=%SHARED_VENV%\Scripts\python.exe"
)

if not defined PYTHON_EXE (
    py -3.11 -c "1" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3.11"
    )
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
    echo [Gen2Train] Could not find a Python interpreter anywhere ^(tried the kohya_ss
    echo shared venv, the "py" launcher, and "python" on PATH^).
    echo Install Python 3.11+, or set python_path in gen2train\settings.local.json
    echo to point at an existing environment that has torch/accelerate/PySide6 installed.
    pause
    exit /b 1
)

if "%PYTHON_EXE%"=="%SHARED_VENV%\Scripts\python.exe" (
    echo [Gen2Train] Using shared kohya_ss venv.
) else (
    echo [Gen2Train] kohya_ss shared venv not found next to this project.
    echo Falling back to "%PYTHON_EXE% %PYTHON_ARGS%" - if required packages
    echo ^(torch, accelerate, PySide6, ...^) are missing there, the app will fail to start.
    echo See gen2train\settings.local.json to point python_path at a working environment.
)

"%PYTHON_EXE%" %PYTHON_ARGS% "%GEN2TRAIN_DIR%app.py" %*

endlocal
