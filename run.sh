#!/usr/bin/env bash
# Gen2Train launcher (Linux/WSL2). Mirrors run.bat's Windows logic, minus the kohya_ss shared
# venv reuse - that venv is a Windows venv (Scripts/python.exe), so it can't be reused here.
# Python priority: (1) this project's own venv (Gen2Train/venv, created by setup.sh/setup.py),
# (2) python3.11/python3 on PATH. If required packages are missing, setup.py is run
# automatically before launching the app.
set -e
cd "$(dirname "$0")"
GEN2TRAIN_DIR="$(pwd)"
LOCAL_VENV="$GEN2TRAIN_DIR/venv"

resolve_python() {
    if [ -x "$LOCAL_VENV/bin/python" ] && "$LOCAL_VENV/bin/python" -c "print(1)" >/dev/null 2>&1; then
        echo "$LOCAL_VENV/bin/python"
        return
    fi
    for candidate in python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
}

PYTHON_EXE="$(resolve_python)"
if [ -z "$PYTHON_EXE" ]; then
    echo "[Gen2Train] Could not find a Python interpreter (tried this project's own venv and"
    echo "python3 on PATH). Install Python 3.11+"
    echo "(e.g. sudo apt update && sudo apt install -y python3.11 python3.11-venv) and try again."
    read -r -p "Press Enter to exit..."
    exit 1
fi

if ! "$PYTHON_EXE" -c "import PySide6, torch, torchvision, accelerate" >/dev/null 2>&1; then
    echo "[Gen2Train] Required packages (PySide6, torch, accelerate, ...) are not installed yet."
    echo "Running first-time setup now - this downloads several GB and may take a while..."
    "$PYTHON_EXE" setup.py
    PYTHON_EXE="$(resolve_python)"
    if [ -z "$PYTHON_EXE" ]; then
        echo "[Gen2Train] Setup finished but no usable Python was found afterward. Please check the log above."
        read -r -p "Press Enter to exit..."
        exit 1
    fi
fi

if [ "$PYTHON_EXE" = "$LOCAL_VENV/bin/python" ]; then
    echo "[Gen2Train] Using this project's own venv."
else
    echo "[Gen2Train] Using \"$PYTHON_EXE\"."
fi

exec "$PYTHON_EXE" "$GEN2TRAIN_DIR/app.py" "$@"
