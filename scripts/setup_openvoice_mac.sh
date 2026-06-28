#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENVOICE_DIR="$ROOT/OpenVoice-main"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="$OPENVOICE_DIR/.venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [ -x /opt/homebrew/opt/python@3.10/bin/python3.10 ]; then
    PYTHON_BIN=/opt/homebrew/opt/python@3.10/bin/python3.10
  elif [ -x /usr/local/opt/python@3.10/bin/python3.10 ]; then
    PYTHON_BIN=/usr/local/opt/python@3.10/bin/python3.10
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing $PYTHON_BIN. Install Python 3.10 first, then rerun this script." >&2
  exit 1
fi

cd "$OPENVOICE_DIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel "setuptools<82"
python -m pip install --no-deps -e .
python -m pip install \
  "numpy<2" \
  torch \
  soundfile \
  librosa \
  pydub \
  wavmark \
  eng_to_ipa \
  inflect \
  unidecode \
  pypinyin \
  cn2an \
  jieba \
  langid
python -m pip install "git+https://github.com/myshell-ai/MeloTTS.git"
python -m unidic download

python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    raise SystemExit(f"torch import failed: {e}")
mps_available = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
print("MPS available:", bool(mps_available))
try:
    import openvoice
    print("openvoice: OK")
except Exception as e:
    raise SystemExit(f"openvoice import failed: {e}")
try:
    import melo
    print("melo: OK")
except Exception as e:
    raise SystemExit(f"melo import failed: {e}")
try:
    import soundfile
    print("soundfile: OK")
except Exception as e:
    raise SystemExit(f"soundfile import failed: {e}")
print("OpenVoice environment OK")
PY

cat <<EOF

OpenVoice environment is installed at:
  $VENV_DIR

Checkpoints are not committed to git. Place OpenVoice V2 checkpoints here:
  $OPENVOICE_DIR/checkpoints_v2

Expected OpenVoice V2 checkpoint layout:
  $OPENVOICE_DIR/checkpoints_v2/
    converter/
      config.json
      checkpoint.pth
    base_speakers/
      ses/
        en-default.pth
        en-newest.pth
        en-us.pth
        es.pth
        fr.pth
        jp.pth
        kr.pth
        zh.pth

Then run:
  cd "$ROOT"
  python3 scripts/doctor.py
  python3 scripts/smoke_openvoice_bridge.py
EOF
