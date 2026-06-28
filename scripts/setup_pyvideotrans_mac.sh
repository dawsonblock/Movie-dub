#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYVT_DIR="$ROOT_DIR/pyvideotrans-main"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [ -x /opt/homebrew/opt/python@3.10/bin/python3.10 ]; then
    PYTHON_BIN=/opt/homebrew/opt/python@3.10/bin/python3.10
  elif [ -x /usr/local/opt/python@3.10/bin/python3.10 ]; then
    PYTHON_BIN=/usr/local/opt/python@3.10/bin/python3.10
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "python3.10 is required. Install it first, then rerun this script." >&2
  exit 1
fi

cd "$PYVT_DIR"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip wheel setuptools

if command -v uv >/dev/null 2>&1 && [ -f "uv.lock" ]; then
  uv sync
elif [ -f "pyproject.toml" ]; then
  python -m pip install -e .
elif [ -f "requirements.txt" ]; then
  python -m pip install -r requirements.txt
else
  echo "No pyproject.toml or requirements.txt found in $PYVT_DIR" >&2
  exit 1
fi

python -m pip install pytest
python - <<'PY'
import sys
print("Python:", sys.version)
import PySide6
print("PySide6: OK")
import videotrans
print("videotrans: OK")
PY

echo "pyVideoTrans setup complete."
