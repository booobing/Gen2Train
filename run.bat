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

REM A venv's python.exe can exist as a file but still be broken: Windows venvs embed the
REM absolute path of the original base Python in pyvenv.cfg, so if this whole venv folder was
REM copied here from another machine/user, launching it prints "No Python at '...'" pointing at
REM the ORIGINAL machine's path - but (surprisingly) still exits with code 0, so a plain
REM errorlevel check does NOT catch this. Instead, actually capture its stdout via a temp file
REM and check it's the value we expect; a broken venv prints its error to stderr and produces no
REM stdout at all. (A "for /f" command-substitution was tried first but cmd.exe mis-parses it
REM when the captured command string itself starts with a quoted path - the temp file avoids
REM that quoting pitfall entirely.)
set "VENV_CHECK="
set "VENV_CHECK_FILE=%TEMP%\_g2t_venv_check_%RANDOM%.txt"
if exist "%SHARED_VENV%\Scripts\python.exe" "%SHARED_VENV%\Scripts\python.exe" -c "print(1)" >"%VENV_CHECK_FILE%" 2>nul
if exist "%VENV_CHECK_FILE%" set /p VENV_CHECK=<"%VENV_CHECK_FILE%"
del "%VENV_CHECK_FILE%" >nul 2>&1

if "%VENV_CHECK%"=="1" (
    set "PYTHON_EXE=%SHARED_VENV%\Scripts\python.exe"
) else (
    if exist "%SHARED_VENV%\Scripts\python.exe" (
        echo [Gen2Train] Found a kohya_ss shared venv, but it doesn't run here - it was likely
        echo copied from another PC and its pyvenv.cfg still points at that PC's Python install.
        echo Falling back to a local Python instead.
    )
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
    echo [Gen2Train] No usable kohya_ss shared venv next to this project.
    echo Falling back to "%PYTHON_EXE% %PYTHON_ARGS%" - if required packages
    echo ^(torch, accelerate, PySide6, ...^) are missing there, the app will fail to start.
    echo See gen2train\settings.local.json to point python_path at a working environment.
)

"%PYTHON_EXE%" %PYTHON_ARGS% "%GEN2TRAIN_DIR%app.py" %*

endlocal
