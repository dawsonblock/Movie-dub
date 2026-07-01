#!/usr/bin/env bash
# Setup the Qwen3-TTS local engine for the Movie-dub split pipeline.
#
# The Qwen3 bridge (bridge/qwen3_segment_tts.py) runs on the SAME python
# that runs scripts/run_personal_dub.py (sys.executable). The Makefile
# invokes the wrapper with `python3`, so this script installs the
# required packages into that interpreter by default. Override with
# PYTHON_BIN=/path/to/python if you run the wrapper with a different one.
#
# What this installs:
#   mlx-audio        Apple Silicon on-device TTS runtime
#   soundfile        WAV writer (bridge writes 24 kHz WAVs)
#   librosa          audio duration / loading helpers
#   numpy            array backend
#   huggingface_hub  model download
#
# What this downloads:
#   mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 -> models/Qwen3-TTS-12Hz-0.6B-Base-bf16
#
# This does NOT touch the pyVideoTrans venv. Qwen3-local is intentionally
# independent of pyVideoTrans transcription dependencies.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_REPO="${QWEN3_MODEL_REPO:-mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16}"
MODEL_DIR="$ROOT/models/Qwen3-TTS-12Hz-0.6B-Base-bf16"

# Validate MODEL_REPO: must be a HuggingFace repo id (org/name, safe chars
# only) to prevent shell/Python injection via the heredoc below.
if ! echo "$MODEL_REPO" | grep -Eq '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'; then
  echo "Invalid QWEN3_MODEL_REPO: $MODEL_REPO" >&2
  echo "Expected format: org/model-name (alphanumerics, dash, underscore, dot)" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Install Python 3.10+ first, then rerun." >&2
  exit 1
fi

echo "Using python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# --- 1. Install packages into the wrapper python ---
echo "Installing mlx-audio + audio deps into $PYTHON_BIN ..."
"$PYTHON_BIN" -m pip install -U \
  "mlx-audio" \
  "soundfile" \
  "librosa" \
  "numpy" \
  "huggingface_hub"

# --- 2. Import check ---
echo "Verifying imports ..."
"$PYTHON_BIN" - <<'PY'
import sys
for mod in ("mlx_audio", "soundfile", "librosa", "numpy", "huggingface_hub"):
    try:
        m = __import__(mod)
        print(f"  {mod}: OK ({getattr(m, '__version__', '?')})")
    except Exception as e:
        raise SystemExit(f"  {mod}: FAIL -> {e}")
PY

# --- 3. Download the model (skip if already present) ---
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "Model already present at $MODEL_DIR — skipping download."
else
  echo "Downloading $MODEL_REPO -> $MODEL_DIR ..."
  mkdir -p "$ROOT/models"
  "$PYTHON_BIN" - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$MODEL_DIR",
    local_dir_use_symlinks=False,
)
print("Download complete.")
PY
fi

# --- 4. Ensure a default reference WAV exists ---
VOICES_DIR="$ROOT/voices"
DEFAULT_REF="$VOICES_DIR/reference.wav"
MALE_REF="$VOICES_DIR/male_reference.wav"
if [ ! -f "$DEFAULT_REF" ]; then
  mkdir -p "$VOICES_DIR"
  if [ -f "$MALE_REF" ]; then
    echo "Creating default reference from $MALE_REF ..."
    cp "$MALE_REF" "$DEFAULT_REF"
  else
    echo "No reference WAV found at $DEFAULT_REF or $MALE_REF."
    echo "Provide a reference.wav before running dubbing."
  fi
fi

# --- 5. Final summary ---
cat <<EOF

Qwen3-local setup complete.

  Python:  $PYTHON_BIN
  Model:   $MODEL_DIR

Next steps:
  cd "$ROOT"
  python3 scripts/doctor.py            # qwen3-local should now report ready
  make smoke-qwen3                     # one-segment generation proof
  make dub INPUT=movie.mp4 OUTPUT=out.mp4 \\
       SPEAKER_PROFILING=1 TTS_ENGINE=qwen3-local \\
       PREFER_EMBEDDED_SUBTITLES=1

Qwen3-local is the default TTS engine for Movie-dub.
EOF
