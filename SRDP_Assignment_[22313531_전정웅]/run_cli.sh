#!/usr/bin/env bash
set -e
PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    echo "Python 3 was not found. Install Python 3 and run pip install -r requirements.txt."
    exit 1
  fi
fi
"$PYTHON_CMD" src/srdp_full_safe_framework.py --epochs 6 --trials 10 --out-prefix outputs/srdp_assignment
