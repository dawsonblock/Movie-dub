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
make download-openvoice
make doctor
```

Checkpoints are not committed to git and are only downloaded when you explicitly run `make download-openvoice`.
The default checkpoint repo is `rsxdalv/OpenVoiceV2`. To use a personal mirror or bucket:

```bash
OPENVOICE_HF_REPO=DmanBlock/OpenVoiceV2-bucket make download-openvoice
```

You can pin a known-good checkpoint revision:

```bash
OPENVOICE_HF_REVISION=<commit> make download-openvoice
```

## Smoke Tests

Run these after install to confirm the runtime is ready:

```bash
make smoke-openvoice
make smoke-e2e
make test-pyvt
```

The OpenVoice provider is registered in pyVideoTrans as provider `34`, displayed as `OpenVoice V2(Local)`.

## Dub Your Own Video

After `make doctor` passes and the smoke tests are green, dub a personal clip.
Start with a 1-2 minute clip (clear speech, one or two speakers, little music)
before trying long videos.

```bash
make dub INPUT=~/Movies/test.mp4 OUTPUT=~/Movies/dubbed-test.mp4
```

Optional flags:

```bash
make dub INPUT=~/Movies/test.mp4 OUTPUT=~/Movies/dubbed-test.mp4 \
     SOURCE=en TARGET=en REFERENCE=voices/openvoice_default_reference.wav \
     BACKGROUND_VOLUME=0.15
```

Or call the script directly:

```bash
python scripts/run_personal_dub.py \
  --input ~/Movies/test.mp4 \
  --source-language en \
  --target-language en \
  --reference voices/openvoice_default_reference.wav \
  --output ~/Movies/dubbed-test.mp4 \
  --background-volume 0.15
```

`--background-volume` mixes the original audio (music, ambience) under the
dubbed speech at the given volume. `0.0` (default) replaces the whole audio
track with speech. `0.15` keeps background audio quiet under the dub.

The wrapper runs the full pyVideoTrans VTV pipeline with OpenVoice, then
rebuilds the dubbed audio track from the manifest and remuxes it into the
final MP4. Job artifacts (manifest, segment WAVs, dubbed audio, review,
remux command, report) are kept under
`pyvideotrans-main/tmp/personal_dub/<timestamp>/`.

## Regenerate a Segment

If a segment sounds wrong, regenerate it and rebuild the final video:

```bash
python scripts/regenerate_segment.py \
  --job-dir pyvideotrans-main/tmp/personal_dub/<timestamp> \
  --segment-id 3 \
  --text "Corrected shorter line." \
  --remux
```

`--remux` rebuilds `dubbed_audio.wav` from the updated manifest and remuxes
`final_dubbed.mp4`, so the change is reflected in the final video. This
works for both smoke jobs and personal dub jobs.

## Troubleshooting

- `make doctor` is the source of truth for missing local setup.
- Missing lines are treated as failure by default: `openvoice_allow_partial` defaults to `false`.
- OpenVoice timing is recorded in each manifest with duration ratios and timing status.
- Generated outputs, venvs, model caches, checkpoints, and local ffmpeg binaries are intentionally ignored.
