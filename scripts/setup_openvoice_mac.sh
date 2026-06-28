#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENVOICE_DIR="$ROOT/OpenVoice-main"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="$OPENVOICE_DIR/.venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Install Python 3.10 first, then rerun this script." >&2
  exit 1
fi

cd "$OPENVOICE_DIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel setuptools
python -m pip install -e .
python -m pip install "git+https://github.com/myshell-ai/MeloTTS.git"
python -m unidic download

cat <<EOF

OpenVoice environment is installed at:
  $VENV_DIR

Checkpoints are not committed to git. Place OpenVoice V2 checkpoints here:
  $OPENVOICE_DIR/checkpoints_v2

Required files:
  $OPENVOICE_DIR/checkpoints_v2/converter/config.json
  $OPENVOICE_DIR/checkpoints_v2/converter/checkpoint.pth
  $OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/*.pth

Then run:
  cd "$ROOT"
  python3 scripts/doctor.py
  python3 scripts/smoke_openvoice_bridge.py
EOF
