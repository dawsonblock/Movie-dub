#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENVOICE_DIR="$ROOT_DIR/OpenVoice-main"
REPO_ID="${OPENVOICE_HF_REPO:-rsxdalv/OpenVoiceV2}"
REVISION="${OPENVOICE_HF_REVISION:-main}"

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

# check_real_file PATH MIN_SIZE
# Rejects missing files, suspiciously small files, and Git LFS/Xet pointer stubs.
check_real_file() {
  local path="$1"
  local min_size="$2"
  if [ ! -f "$path" ]; then
    echo "MISSING: $path"
    return 1
  fi
  local size
  size="$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path")"
  if [ "$size" -lt "$min_size" ]; then
    echo "BAD: $path is too small: $size bytes"
    echo "This may be a Git LFS/Xet pointer instead of the real file."
    return 1
  fi
  if head -n 1 "$path" | grep -q "version https://git-lfs.github.com/spec"; then
    echo "BAD: $path is a Git LFS pointer, not the real file."
    return 1
  fi
  echo "OK: $path"
}

verify_checkpoints() {
  local quiet="${1:-}"
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      [ "$quiet" = "--quiet" ] || echo "MISSING: $path"
      return 1
    fi
  done

  # Validate every .pth is a real file, not a pointer stub.
  # converter checkpoint is ~131 MB; base speaker embeddings are small but real.
  if ! check_real_file "$OPENVOICE_DIR/checkpoints_v2/converter/checkpoint.pth" 100000000; then
    [ "$quiet" = "--quiet" ] || echo "This probably means you downloaded a Git LFS/Xet pointer instead of the real checkpoint."
    return 1
  fi
  for ses in en-default en-newest en-us es fr jp kr zh; do
    if ! check_real_file "$OPENVOICE_DIR/checkpoints_v2/base_speakers/ses/$ses.pth" 1000; then
      return 1
    fi
  done

  [ "$quiet" = "--quiet" ] || echo "All OpenVoice V2 checkpoint files verified."
}

echo "Downloading OpenVoice V2 checkpoints"
echo "Repo: $REPO_ID (revision: $REVISION)"
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
    --revision "$REVISION" \
    --local-dir "$OPENVOICE_DIR" \
    --include "checkpoints_v2/*"
elif command -v hf >/dev/null 2>&1; then
  hf download "$REPO_ID" \
    --revision "$REVISION" \
    --local-dir "$OPENVOICE_DIR" \
    --include "checkpoints_v2/*"
else
  echo "Neither uvx nor hf was found."
  echo "Install uv first, or install huggingface_hub:"
  echo "  brew install uv"
  echo "  uvx hf download $REPO_ID --revision $REVISION --local-dir $OPENVOICE_DIR --include 'checkpoints_v2/*'"
  echo "  python3 -m pip install -U huggingface_hub"
  exit 1
fi

echo
echo "Verifying checkpoint layout..."
verify_checkpoints

echo
echo "OpenVoice V2 checkpoints are installed."
