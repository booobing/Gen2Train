@echo off
REM Installs the Gen2Train AI helper chatbot backend (venv_chat + llama-cpp-python + GGUF model).
REM Safe to re-run: it skips steps that are already done (existing venv, existing model file).
REM Only used to bootstrap *which* Python builds venv_chat - setup_chat_model.py itself does
REM the real (CUDA/GPU-aware) work once it's running.
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
    echo [setup_chat] Could not find a Python interpreter ^(tried the "py" launcher and
    echo "python" on PATH^). Install Python 3.11+ from python.org and try again.
    pause
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% "%GEN2TRAIN_DIR%chat_backend\setup_chat_model.py"

pause
endlocal
