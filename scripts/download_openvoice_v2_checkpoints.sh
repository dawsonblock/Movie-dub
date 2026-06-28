#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENVOICE_DIR="$ROOT_DIR/OpenVoice-main"
REPO_ID="${OPENVOICE_HF_REPO:-rsxdalv/OpenVoiceV2}"

mkdir -p "$OPENVOICE_DIR"

required=(
  "$OPENVOICE_DIR/checkpoints_v2/converter/config.json"
  "$OPENVOICE_DIR/checkpoints_v2/converter/checkpoint.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/en-default.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/en-newest.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/en-us.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/es.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/fr.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/jp.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/kr.pth"
  "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/zh.pth"
)

verify_checkpoints() {
  local quiet="${1:-}"
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      [ "$quiet" = "--quiet" ] || echo "MISSING: $path"
      return 1
    fi
    [ "$quiet" = "--quiet" ] || echo "OK: $path"
  done

  local checkpoint="$OPENVOICE_DIR/checkpoints_v2/converter/checkpoint.pth"
  local checkpoint_size
  checkpoint_size="$(stat -f%z "$checkpoint" 2>/dev/null || stat -c%s "$checkpoint")"
  if [ "$checkpoint_size" -lt 100000000 ]; then
    [ "$quiet" = "--quiet" ] || echo "BAD: converter checkpoint is too small: $checkpoint_size bytes"
    [ "$quiet" = "--quiet" ] || echo "This probably means you downloaded a Git LFS/Xet pointer instead of the real checkpoint."
    return 1
  fi

  if LC_ALL=C head -c 200 "$checkpoint" | grep -q "version https://git-lfs.github.com/spec"; then
    [ "$quiet" = "--quiet" ] || echo "BAD: converter checkpoint is a Git LFS pointer, not the real file."
    return 1
  fi
}

echo "Downloading OpenVoice V2 checkpoints"
echo "Repo: $REPO_ID"
echo "Destination: $OPENVOICE_DIR"

if verify_checkpoints --quiet; then
  echo "Existing OpenVoice V2 checkpoints are already installed."
  echo
  echo "Verifying checkpoint layout..."
  verify_checkpoints
  echo
  echo "OpenVoice V2 checkpoints are installed."
  exit 0
fi

if command -v uvx >/dev/null 2>&1; then
  uvx hf download "$REPO_ID" \
    --local-dir "$OPENVOICE_DIR" \
    --include "checkpoints_v2/*"
elif command -v hf >/dev/null 2>&1; then
  hf download "$REPO_ID" \
    --local-dir "$OPENVOICE_DIR" \
    --include "checkpoints_v2/*"
else
  echo "Neither uvx nor hf was found."
  echo "Install uv first, or install huggingface_hub:"
  echo "  brew install uv"
  echo "  uvx hf download $REPO_ID --local-dir $OPENVOICE_DIR --include 'checkpoints_v2/*'"
  echo "  python3 -m pip install -U huggingface_hub"
  exit 1
fi

echo
echo "Verifying checkpoint layout..."
verify_checkpoints

echo
echo "OpenVoice V2 checkpoints are installed."
