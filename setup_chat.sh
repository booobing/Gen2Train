#!/usr/bin/env bash
# Installs the Gen2Train AI helper chatbot backend (venv_chat + llama-cpp-python + GGUF model)
# for Linux/WSL2. Mirrors setup_chat.bat's Windows logic. Safe to re-run: it skips steps that
# are already done (existing venv, existing model file). Only used to bootstrap *which* Python
# builds venv_chat - chat_backend/setup_chat_model.py itself does the real (CUDA/GPU-aware)
# work once it's running. For WSL2, make sure the CUDA toolkit (not the driver - that stays on
# the Windows side) is installed inside WSL2 first: https://developer.nvidia.com/cuda/wsl
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
    echo "[setup_chat] Could not find a Python interpreter. Install Python 3.11+ from your"
    echo "distro's package manager (e.g. sudo apt install python3.11 python3.11-venv) and try again."
    read -r -p "Press Enter to exit..."
    exit 1
fi

"$PYTHON_BIN" chat_backend/setup_chat_model.py

read -r -p "Press Enter to exit..."
