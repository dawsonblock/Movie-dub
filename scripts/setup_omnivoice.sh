#!/usr/bin/env bash
# Install dependencies for the OmniVoice split-pipeline bridge.
#
# OmniVoice is a server-based TTS engine. The bridge script
# (bridge/omnivoice_segment_tts.py) needs:
#   - requests (for OmniVoice-studio mode on port 3900)
#   - gradio_client (for Gradio web-UI mode)
#   - soundfile (for output WAV duration checks)
#
# The bridge runs on the same python that runs run_personal_dub.py (the
# wrapper interpreter), so we install into the current environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

python3 -m pip install --upgrade "gradio_client==2.0.3" requests soundfile

echo "OmniVoice bridge dependencies installed."
echo "Next: start your OmniVoice server and run with --tts-engine omnivoice --omnivoice-url <url>"
