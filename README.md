# Movie-dub

Personal Mac dubbing workflow using pyVideoTrans for video orchestration and OpenVoice V2 for cloned segment audio.

## Status

- Personal-use scaffold: yes
- One-click app: no
- Production-ready: no
- Commercial distribution: license review required because pyVideoTrans is GPLv3

## What this is

This repo keeps the two runtimes isolated:

- `pyvideotrans-main`: transcription, translation, subtitles, timing, mixing, export
- `OpenVoice-main`: local OpenVoice V2 voice conversion and MeloTTS base speech
- `bridge/openvoice_segment_tts.py`: JSON/WAV subprocess boundary between them

Do not merge OpenVoice into pyVideoTrans or install both stacks into one venv.

## Requirements

- macOS with Python 3.10
- local `ffmpeg` and `ffprobe`
- `pyvideotrans-main/.venv`
- `OpenVoice-main/.venv`
- OpenVoice V2 checkpoints in `OpenVoice-main/checkpoints_v2`

Expected checkpoint layout:

```text
OpenVoice-main/checkpoints_v2/
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
```

## Install

```bash
make setup-ffmpeg
make setup-pyvt
make setup-openvoice
```

Checkpoints are not committed to git. Place them under `OpenVoice-main/checkpoints_v2`, then run:

```bash
make doctor
```

## Smoke Tests

```bash
make smoke-openvoice
make smoke-e2e
make test-openvoice
```

The OpenVoice provider is registered in pyVideoTrans as provider `34`, displayed as `OpenVoice V2(Local)`.

## Run Your Own Short Clip

Use pyVideoTrans CLI with TTS provider `34` after `make doctor` passes. Start with a 1-2 minute clip before trying long videos.

```bash
cd pyvideotrans-main
.venv/bin/python cli.py --task vtv --name /absolute/path/input.mp4 \
  --recogn_type 0 --model_name tiny \
  --source_language_code en --target_language_code en \
  --tts_type 34 --voice_role default \
  --subtitle_type 1 --no-clear-cache --verbose
```

## Troubleshooting

- `make doctor` is the source of truth for missing local setup.
- Missing lines are treated as failure by default: `openvoice_allow_partial` defaults to `false`.
- OpenVoice timing is recorded in each manifest with duration ratios and timing status.
- Generated outputs, venvs, model caches, checkpoints, and local ffmpeg binaries are intentionally ignored.
