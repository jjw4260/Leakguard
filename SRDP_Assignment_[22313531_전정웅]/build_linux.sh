#!/usr/bin/env bash
set -e
PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    echo "Python 3 was not found. Install Python 3 first."
    exit 1
  fi
fi
"$PYTHON_CMD" -m pip install -r requirements.txt
"$PYTHON_CMD" -m PyInstaller --noconfirm --onefile --windowed --name SRDP_Assignment_GUI src/srdp_gui.py
