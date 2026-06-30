#!/usr/bin/env bash
# Setup the optional age-regression plugin for Movie-dub speaker profiling.
#
# Installs the griko/voice-age-regression package into a dedicated venv
# (.venv-age) and downloads the trained model into models/ (gitignored).
# The model is NOT cloned into the repo — it is fetched with `hf download`
# so only the weights land in models/, never in git.
#
# What this installs (into .venv-age):
#   voice-age-regressor[full]  AgeRegressionPipeline (SpeechBrain ECAPA + ANN)
#   huggingface_hub            hf CLI for the model download
#   soundfile                  WAV I/O
#   librosa                    acoustic features used by the regressor
#
# What this downloads (into models/age_reg_ann_ecapa_librosa_combined):
#   griko/age_reg_ann_ecapa_librosa_combined  ~ECAPA-TDNN + ANN weights
#
# Model card: https://huggingface.co/griko/age_reg_ann_ecapa_librosa_combined
# Test MAE ~6.93 years on the combined VoxCeleb2 + TIMIT set.
# This is an APPARENT VOCAL AGE estimate, not exact biological age.
#
# Usage:
#   make setup-age
#   # or directly:
#   bash scripts/setup_age_model.sh
#
# Optional: install git-xet first if Hugging Face requires it for large files:
#   brew install git-xet && git xet install
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found. Install Python 3.10 first." >&2
    exit 1
fi

VENV_DIR="$ROOT_DIR/.venv-age"
MODEL_DIR="$ROOT_DIR/models/age_reg_ann_ecapa_librosa_combined"
MODEL_REPO="griko/age_reg_ann_ecapa_librosa_combined"

echo "==> Creating .venv-age with $PYTHON_BIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip / wheel / setuptools"
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing voice-age-regression[full]"
python -m pip install "git+https://github.com/griko/voice-age-regression.git#egg=voice-age-regressor[full]"

echo "==> Installing huggingface_hub soundfile librosa"
python -m pip install huggingface_hub soundfile librosa

echo "==> Downloading model to $MODEL_DIR (hf download, not git clone)"
mkdir -p "$MODEL_DIR"
# Prefer the hf CLI installed above; fall back to uvx if hf is not on PATH.
if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
elif command -v uvx >/dev/null 2>&1; then
    uvx huggingface_hub hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
else
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', local_dir='$MODEL_DIR')"
fi

echo "==> Verifying the model loads"
python - <<'PY'
from pathlib import Path
from age_regressor import AgeRegressionPipeline
model_path = Path("models/age_reg_ann_ecapa_librosa_combined")
if not model_path.exists():
    raise SystemExit("Model dir missing after download: " + str(model_path))
AgeRegressionPipeline.from_pretrained(str(model_path))
print("Age model OK: " + str(model_path))
PY

echo
echo "Age model setup complete."
echo "  venv:  $VENV_DIR"
echo "  model: $MODEL_DIR"
echo
echo "Verify with:"
echo "  make doctor-age"
echo "  make smoke-age"
