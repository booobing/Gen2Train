#!/usr/bin/env bash
# Gen2Train main app installer (Linux/WSL2). Mirrors setup.bat's Windows logic - installs
# torch/accelerate/transformers/diffusers/PySide6/psutil etc. needed for training and the UI
# into this project's own venv (Gen2Train/venv). run.sh runs this automatically when required
# packages are missing, so normally you only need run.sh.
set -e
cd "$(dirname "$0")"

PYTHON_BIN=""
for candidate in python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "[setup] Could not find a Python interpreter. Install Python 3.11+"
    echo "(e.g. sudo apt update && sudo apt install -y python3.11 python3.11-venv) and try again."
    read -r -p "Press Enter to exit..."
    exit 1
fi

"$PYTHON_BIN" setup.py

read -r -p "Press Enter to exit..."
