#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFMPEG_DIR="$ROOT/pyvideotrans-main/ffmpeg"
FFMPEG_VERSION="${FFMPEG_VERSION:-7.1.1}"

mkdir -p "$FFMPEG_DIR"
cd "$FFMPEG_DIR"

curl -L --fail -o ffmpeg.zip "https://evermeet.cx/ffmpeg/ffmpeg-${FFMPEG_VERSION}.zip"
curl -L --fail -o ffprobe.zip "https://evermeet.cx/ffmpeg/ffprobe-${FFMPEG_VERSION}.zip"
unzip -o ffmpeg.zip
unzip -o ffprobe.zip
chmod +x ffmpeg ffprobe

./ffmpeg -version | head -1
./ffprobe -version | head -1

cat <<EOF

Installed local ffmpeg binaries in:
  $FFMPEG_DIR

pyVideoTrans prepends this directory to PATH at runtime.
EOF
