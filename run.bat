@echo off
REM Gen2Train launcher.
REM Python priority: (1) kohya_ss shared venv (only if it actually works), (2) this
REM project's own venv (Gen2Train\venv, created by setup.bat/setup.py), (3) the "py"
REM launcher, (4) "python" on PATH. If required packages are missing, setup.py is run
REM automatically before launching the app.
setlocal

set "GEN2TRAIN_DIR=%~dp0"
set "SHARED_VENV=%GEN2TRAIN_DIR%..\kohya_ss\kohya_ss\venv"
set "LOCAL_VENV=%GEN2TRAIN_DIR%venv"

call :ResolvePython
if not defined PYTHON_EXE (
    echo [Gen2Train] Could not find a Python interpreter anywhere ^(tried the kohya_ss
    echo shared venv, this project's own venv, the "py" launcher, and "python" on PATH^).
    echo Install Python 3.11+, or set python_path in gen2train\settings.local.json
    echo to point at an existing environment that has torch/accelerate/PySide6 installed.
    pause
    exit /b 1
)

REM Check whether required packages are actually importable. Results are passed through a
REM temp file rather than a for/f capture (see CheckVenvPython below for why).
set "PKG_CHECK="
set "PKG_CHECK_FILE=%TEMP%\_g2t_pkg_check_%RANDOM%.txt"
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import PySide6, torch, torchvision, accelerate; print(1)" >"%PKG_CHECK_FILE%" 2>nul
if exist "%PKG_CHECK_FILE%" set /p PKG_CHECK=<"%PKG_CHECK_FILE%"
del "%PKG_CHECK_FILE%" >nul 2>&1

if not "%PKG_CHECK%"=="1" (
    echo [Gen2Train] Required packages ^(PySide6, torch, accelerate, ...^) are not installed yet.
    echo Running first-time setup now - this downloads several GB and may take a while...
    "%PYTHON_EXE%" %PYTHON_ARGS% "%GEN2TRAIN_DIR%setup.py"
    if errorlevel 1 (
        echo [Gen2Train] Setup failed. Check the error above, or try running setup.bat directly.
        pause
        exit /b 1
    )
    REM setup.py always ends up using either the shared venv or a freshly created local venv,
    REM never the bare "py"/"python" fallback - re-resolve from scratch.
    set "PYTHON_EXE="
    set "PYTHON_ARGS="
    call :ResolvePython
    if not defined PYTHON_EXE (
        echo [Gen2Train] Setup finished but no usable Python was found afterward. Please check the log above.
        pause
        exit /b 1
    )
)

if "%PYTHON_EXE%"=="%SHARED_VENV%\Scripts\python.exe" (
    echo [Gen2Train] Using shared kohya_ss venv.
) else if "%PYTHON_EXE%"=="%LOCAL_VENV%\Scripts\python.exe" (
    echo [Gen2Train] Using this project's own venv.
) else (
    echo [Gen2Train] Using "%PYTHON_EXE% %PYTHON_ARGS%".
)

"%PYTHON_EXE%" %PYTHON_ARGS% "%GEN2TRAIN_DIR%app.py" %*

endlocal
goto :eof

:ResolvePython
REM A venv's python.exe can exist as a file but still be broken: Windows venvs embed the
REM absolute path of the original base Python in pyvenv.cfg, so if this whole venv folder was
REM copied here from another machine/user, launching it prints "No Python at ..." pointing at
REM the ORIGINAL machine's path - but (surprisingly) still exits with code 0, so a plain
REM errorlevel check does NOT catch this. Instead, actually capture its stdout via a temp file
REM and check it is the value we expect; a broken venv prints its error to stderr and produces
REM no stdout at all.
call :CheckVenvPython "%SHARED_VENV%\Scripts\python.exe"
if "%VENV_CHECK%"=="1" (
    set "PYTHON_EXE=%SHARED_VENV%\Scripts\python.exe"
    set "PYTHON_ARGS="
    goto :eof
)

call :CheckVenvPython "%LOCAL_VENV%\Scripts\python.exe"
if "%VENV_CHECK%"=="1" (
    set "PYTHON_EXE=%LOCAL_VENV%\Scripts\python.exe"
    set "PYTHON_ARGS="
    goto :eof
)

py -3.11 -c "1" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3.11"
    goto :eof
)

py -3 -c "1" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    goto :eof
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    set "PYTHON_ARGS="
    goto :eof
)

goto :eof

:CheckVenvPython
set "VENV_CHECK="
set "VENV_CHECK_FILE=%TEMP%\_g2t_venv_check_%RANDOM%.txt"
if exist "%~1" "%~1" -c "print(1)" >"%VENV_CHECK_FILE%" 2>nul
if exist "%VENV_CHECK_FILE%" set /p VENV_CHECK=<"%VENV_CHECK_FILE%"
del "%VENV_CHECK_FILE%" >nul 2>&1
goto :eof
