# Movie-dub

Local pyVideoTrans + OpenVoice dubbing workflow.

## What is included

- `bridge/openvoice_segment_tts.py`: subprocess bridge that reads pyVideoTrans-style `queue_tts` JSON, generates OpenVoice V2 WAV segments, and writes a timing/status manifest.
- `pyvideotrans-main/videotrans/tts/_openvoice.py`: pyVideoTrans TTS provider for `OpenVoice V2(Local)`.
- `pyvideotrans-main/tests/test_openvoice_bridge.py`: focused bridge/config tests.
- `assets/pyvideotrans_test_clip.mp4`, `assets/test_clip.wav`, and `voices/openvoice_default_reference.wav`: small local smoke-test inputs.

## Local assets not committed

The Python virtual environments, pyVideoTrans model cache, generated output, and OpenVoice checkpoints are intentionally ignored. The validated local paths are:

- pyVideoTrans venv: `pyvideotrans-main/.venv`
- OpenVoice venv: `OpenVoice-main/.venv`
- OpenVoice V2 checkpoints: `OpenVoice-main/checkpoints_v2`

The checkpoint directory must contain at least:

- `OpenVoice-main/checkpoints_v2/converter/config.json`
- `OpenVoice-main/checkpoints_v2/converter/checkpoint.pth`

## Smoke checks

```bash
python3 -m py_compile bridge/openvoice_segment_tts.py pyvideotrans-main/videotrans/tts/_openvoice.py
cd pyvideotrans-main
.venv/bin/python -m pytest tests/test_openvoice_bridge.py -q
.venv/bin/python cli.py --list providers
```

The OpenVoice provider is registered as provider `34`, displayed as `OpenVoice V2(Local)`.
