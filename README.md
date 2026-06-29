# Movie-dub

Personal Mac dubbing workflow using pyVideoTrans for video orchestration and OpenVoice V2 for cloned segment audio.

## Status

- Personal-use scaffold: yes
- One-click app: no
- Production-ready: no
- Commercial distribution: license review required because pyVideoTrans is GPLv3

## Licensing

This repository bundles multiple upstream projects with different licenses:

- **Wrapper scripts** (`scripts/`, `bridge/`, `Makefile`): MIT
- **pyVideoTrans** (`pyvideotrans-main/`): GPLv3
- **OpenVoice** (`OpenVoice-main/`): Apache 2.0

The root `LICENSE` file applies only to the original wrapper/orchestration
code. Because pyVideoTrans is GPLv3, redistribution of the full bundle must
comply with GPLv3 terms. See `LICENSES.md` and `UPSTREAMS.md` for details.

**Do not assume the whole project is MIT. It is not.**

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

The default install does **not** require Demucs. Demucs is only needed for
high-quality AI-based vocal separation. The default background extraction
uses an ffmpeg fallback (center-channel pan filter).

- Default doctor: passes without Demucs installed.
- `make doctor-demucs`: fails if Demucs is missing.
- `make doctor-strict`: fails on any optional warning.
- `make doctor-checkpoints`: checks only OpenVoice checkpoints.

Doctor flags:

```bash
python scripts/doctor.py                    # default: required checks only
python scripts/doctor.py --need-demucs      # require Demucs
python scripts/doctor.py --need-vocal-separation  # require Demucs
python scripts/doctor.py --strict           # require everything
python scripts/doctor.py --checkpoints-only # only check OpenVoice checkpoints
python scripts/doctor.py --no-openvoice     # skip OpenVoice (pyVideoTrans-only)
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
     BACKGROUND_VOLUME=0.15 VOICE_VOLUME=1.2
```

Or call the script directly:

```bash
python scripts/run_personal_dub.py \
  --input ~/Movies/test.mp4 \
  --source-language en \
  --target-language en \
  --reference voices/openvoice_default_reference.wav \
  --output ~/Movies/dubbed-test.mp4 \
  --background-volume 0.15 \
  --voice-volume 1.2
```

Audio quality flags:

| Flag | Default | What it does |
|------|---------|--------------|
| `--background-volume` | `0.0` | Original audio mix level (0.15 = quiet bg under dub) |
| `--voice-volume` | `1.0` | Gain on dubbed speech (1.5 = louder, 0.5 = quieter) |
| `--final-gain` | `1.0` | Gain applied after normalization (1.0 = no change) |
| `--no-normalize` | off | Skip peak normalization — use with `--final-gain` for manual level control |
| `--vocal-separation` | off | Remove center-channel vocals from background audio (karaoke-style) |
| `--ducking` | off | Lower background volume where dubbed speech is present |
| `--target-lufs` | off | EBU R128 LUFS normalization (-16 = web/streaming, -23 = broadcast) |
| `--background-timeout` | `300` | FFmpeg timeout for background audio extraction (seconds) |
| `--lufs-timeout` | `600` | FFmpeg timeout for LUFS normalization (seconds) |
| `--fail-if-background-mix-fails` | off | Fail hard if background mixing fails instead of silently producing speech-only output |

**Note on `--voice-volume`:** With default peak normalization, increasing
`--voice-volume` changes the voice-to-background balance but the final peak
is still normalized to 0.97. For absolute level control, use
`--no-normalize --voice-volume 1.2 --final-gain 0.9`.

If background mixing or LUFS normalization fails (e.g. timeout on a long
video), a warning is printed to stderr and the report includes a `warnings`
list. Use `--fail-if-background-mix-fails` to make background mix failures
hard errors instead of silent fallbacks.

For normal use, `--background-volume 0.15` is enough. For movies with music:

```bash
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 \
     BACKGROUND_VOLUME=0.2 VOCAL_SEPARATION=1 DUCKING=1 TARGET_LUFS=-16
```

This removes the original vocals from the background, ducks the music when
speech is playing, and normalizes the final mix to streaming loudness.

The wrapper runs the full pyVideoTrans VTV pipeline with OpenVoice, then
rebuilds the dubbed audio track from the manifest and remuxes it into the
final MP4. Job artifacts (manifest, segment WAVs, dubbed audio, review,
remux command, report, job.json) are kept under
`pyvideotrans-main/tmp/personal_dub/<job_id>/`.

### Job directory safety

The wrapper will **never** delete an existing job directory by default.
If you pass `--job-dir` pointing to an existing directory, the run will
fail with a clear error. To overwrite a job dir:

1. It must be under `pyvideotrans-main/tmp/personal_dub/`
2. It must contain a `.job-created-by-movie-dub` marker file
3. You must pass `--force-overwrite-job-dir`

External directories are never deleted automatically.

### Demucs fallback

If you request `--vocal-separation --vocal-separation-method demucs` but
Demucs is not installed or crashes during inference, the pipeline
automatically falls back to the ffmpeg center-channel pan filter. The
build audio report includes `demucs_used`, `demucs_error`, and
`ffmpeg_fallback_used` fields so you can verify which method was used.

### LUFS normalization and sample rate

LUFS normalization preserves the selected sample rate through the entire
pipeline. If you request `--sample-rate 48000`, the output will be 48000 Hz.

### Proof report

`make proof` writes a machine-readable `reports/proof_report.json` with
the result of every check (doctor, smoke tests, pyVideoTrans tests). This
file is written even on failure, so you can inspect which check failed.

## Regenerate a Segment

If a segment sounds wrong, regenerate it and rebuild the final video:

```bash
python scripts/regenerate_segment.py \
  --job-dir pyvideotrans-main/tmp/personal_dub/<job_id> \
  --segment-id 3 \
  --text "Corrected shorter line." \
  --remux
```

`--remux` rebuilds `dubbed_audio.wav` from the updated manifest, remuxes
`final_dubbed.mp4` in the job directory, and syncs the result to your
configured output path. A `regeneration_report.json` is written to the
job directory with hash verification of the old and new segment audio.

**Warning:** Regeneration updates the job output video and overwrites the
configured output MP4 path. Keep the job directory if you want future
segment repair. The canonical final video lives at
`<job_dir>/final_dubbed.mp4` — the user output path is a synced copy.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `make doctor` fails on Demucs | Demucs not installed | Demucs is optional; default doctor should pass. Use `make doctor-demucs` only if you need it. |
| `make doctor` fails on checkpoints | OpenVoice V2 checkpoints missing | Run `make download-openvoice` |
| `make doctor` fails on ffmpeg/ffprobe | ffmpeg not installed | Run `make setup-ffmpeg` |
| Job dir already exists error | `--job-dir` points to existing dir | Use `--force-overwrite-job-dir` (only works for marked dirs under `tmp/personal_dub/`) |
| Demucs fallback to ffmpeg | Demucs not installed or crashed | Install Demucs or accept ffmpeg fallback (check `build_audio_report.json`) |
| LUFS output has wrong sample rate | Old version used default 44100 | Fixed: LUFS now preserves the selected sample rate |
| Final MP4 missing audio stream | Remux produced bad output | Verification now catches this; check `verification` in report |
| Segment regeneration didn't update video | `--remux` not passed | Always pass `--remux` to rebuild audio + video |
| `proof_report.json` says fail | One of the proof checks failed | Read the `failed_check` field in the report |
| Config pollution after run | Old version restored missing keys as empty strings | Fixed: missing keys are now removed on restore |

## Mac Acceptance Checklist

Before considering the build ready, verify all of these on macOS:

```bash
# 1. Clean and set up from scratch
make clean
make setup-ffmpeg
make setup-pyvt
make setup-openvoice
make download-openvoice

# 2. Doctor passes (without Demucs)
make doctor

# 3. Proof passes and writes a report
make proof
cat reports/proof_report.json   # result must be "pass"

# 4. Dub a test clip
make dub INPUT=~/Movies/test-90s.mp4 OUTPUT=~/Movies/test-90s-dubbed.mp4 \
     BACKGROUND_VOLUME=0.15 VOICE_VOLUME=1.1 DUCKING=1 TARGET_LUFS=-16

# 5. Verify job artifacts exist
ls pyvideotrans-main/tmp/personal_dub/*/job.json
ls pyvideotrans-main/tmp/personal_dub/*/openvoice_manifest.json
ls pyvideotrans-main/tmp/personal_dub/*/review_segments.json
ls pyvideotrans-main/tmp/personal_dub/*/dubbed_audio.wav
ls pyvideotrans-main/tmp/personal_dub/*/final_dubbed.mp4

# 6. Verify final MP4 has video + audio
ffprobe -v error -show_streams pyvideotrans-main/tmp/personal_dub/*/final_dubbed.mp4

# 7. Regenerate a segment
python scripts/regenerate_segment.py \
  --job-dir <latest-job-dir> \
  --segment-id 1 \
  --text "replacement test line" \
  --remux

# 8. Check regeneration report
cat <latest-job-dir>/regeneration_report.json
```

- `make doctor` passes without Demucs installed
- `make doctor-demucs` fails if Demucs is missing
- `reports/proof_report.json` exists and says `pass`
- `job.json` exists in every job directory
- Final MP4 has both video and audio streams
- Duration is close to original (within 2s or 2%)
- Segment regeneration updates manifest, audio, and video
- No unsafe directory deletion is possible
- No global temp scan is required for artifact discovery

## Notes

- `make doctor` is the source of truth for missing local setup.
- Missing lines are treated as failure by default: `openvoice_allow_partial` defaults to `false`.
- OpenVoice timing is recorded in each manifest with duration ratios and timing status.
- Generated outputs, venvs, model caches, checkpoints, and local ffmpeg binaries are intentionally ignored.
