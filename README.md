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

Optional — Qwen3-TTS local engine (Apple Silicon, independent of OpenVoice):

```bash
make setup-qwen3
make doctor-qwen3
make smoke-qwen3
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
python scripts/doctor.py --qwen3-only       # only check Qwen3-local engine
python scripts/doctor.py --no-openvoice     # skip OpenVoice (pyVideoTrans-only)
python scripts/doctor.py --json             # machine-readable readiness JSON
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
make smoke-qwen3     # only if you ran make setup-qwen3
make test-pyvt
```

The OpenVoice provider is registered in pyVideoTrans as provider `34`, displayed as `OpenVoice V2(Local)`.

## Unit Tests

The `tests/` directory contains 238 pytest tests for the v0.12 modules (age
model, cache/resume, character profiles, review loop, benchmark, CLI wiring,
cache-wiring integration, voice-lock routing).
They are lightweight and need no model dependencies.

```bash
# One-time: install pytest + ruff
make setup-dev

# Run the suite (from the repo root)
make test

# Verbose output
make test-verbose

# Or from any directory:
bin/test
bin/test --verbose
```

If `make test` fails with "pytest is not installed", run `make setup-dev`
first. The `.python-version` file pins the repo to Python 3.12.0 via pyenv;
if you use a different version, make sure pytest is installed for it
(`python3 -m pip install pytest`).

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
     SOURCE_LANG=en TARGET_LANG=en REFERENCE=voices/openvoice_default_reference.wav \
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

### Speaker profiling (multi-character scenes)

For videos with multiple speakers, enable per-speaker profiling to
diarize speakers, extract a canonical reference voice per speaker, and
estimate apparent gender / age band / pitch per speaker:

```bash
# Requires HF_TOKEN for pyannote diarization
export HF_TOKEN=hf_xxxxx
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 \
     SPEAKER_PROFILING=1 DIARIZATION=pyannote \
     TTS_ENGINE=openvoice VERIFY_PITCH=1
```

#### Where the text comes from: subtitles vs. ASR

The split pipeline needs two things from the source: **what** was said
and **when**. There are two sources for that, and they are not
interchangeable:

- **Embedded subtitles** give text + timing, but never tell you **who**
  spoke. A subtitle cue has no speaker identity.
- **ASR (whisper)** transcribes the audio and gives text + timing, also
  without speaker identity.

Speaker identity (**who**) always comes from **audio diarization**
(`analyze_speakers.py`), which is then aligned to the text cues by
timestamp overlap. So the full pipeline is:

```
video
  -> find embedded subtitles          (ffprobe -select_streams s)
  -> extract subtitle cues + timing   (ffmpeg -map 0:s:k -> source.srt)
  -> translate cues                   (STS -> translated.srt)
  -> diarize speakers from audio      (pyannote -> speaker turns)
  -> align cue timestamps to speakers (overlap match)
  -> build per-speaker profiles       (gender/age/pitch + reference.wav)
  -> voice-clone TTS per cue          (reference -> English WAV)
  -> place WAVs back on the timeline
  -> mix/remux final video
```

By default the split pipeline runs whisper ASR to produce the source
SRT. Add `PREFER_EMBEDDED_SUBTITLES=1` to use an embedded subtitle
stream as the text+timing source instead, skipping ASR when one is
found:

```bash
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 \
     SPEAKER_PROFILING=1 PREFER_EMBEDDED_SUBTITLES=1 \
     TTS_ENGINE=openvoice
```

`scripts/extract_embedded_subtitles.py` handles stream discovery and
extraction:

- Lists subtitle streams with `ffprobe -select_streams s`.
- Prefers a text-based stream (`subrip`/`ass`/`mov_text`/`webvtt`)
  whose language tag matches `--source-language`, then any text-based
  stream.
- Extracts it to `job/subtitles/source.srt` via `ffmpeg -map 0:s:k
  -c:s srt`.
- If only image-based subtitles are present (PGS/VobSub), attempts OCR
  (`pgs-to-srt` / `vobsub2srt`). If the OCR tool is missing or fails,
  falls back to whisper ASR. Add `NO_OCR_IMAGE_SUBS=1` to skip OCR and
  go straight to ASR.
- If no usable subtitle stream is found at all, falls back to whisper
  ASR. Embedded subtitles are an optimization, never a hard requirement.

The chosen source is recorded in `job.json` as `subtitle_source`
(`embedded_text`, `embedded_ocr`, or `asr`).

This runs `scripts/analyze_speakers.py` before the dub, producing:

- `job/speakers/speaker_profiles.json` — per-speaker gender, age band,
  pitch profile (median/p10/p90 F0, voiced ratio), and reference audio path
- `job/speakers/SPEAKER_00/reference.wav` — canonical 3-8s reference clip
- `job/speakers/SPEAKER_01/reference.wav` — etc.
- `job/speakers/diarization.rttm` — raw pyannote output for audit

Each segment in the manifest is enriched with `speaker_id`,
`target_gender`, `target_age_band`, `target_pitch_median_hz`, and
`tts_engine`. The review file (`review_segments.json`) includes the same
fields per segment.

**Age estimation is a pitch-based heuristic, not a trained model.** It
classifies into child / teen / adult / senior bands based on F0 ranges
and voiced ratio. A real age regressor is flagged as future work.

You can also run speaker profiling standalone:

```bash
make analyze-speakers INPUT=~/Movies/test.mp4 OUTPUT=tmp/speakers
```

And reuse an existing profile for a re-dub:

```bash
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 \
     SPEAKER_PROFILE_JSON=tmp/speakers/speaker_profiles.json
```

### Post-generation pitch verification

With `--verify-pitch`, each generated segment's F0 is compared to the
source speaker profile. Segments are flagged:

- `needs_review` — pitch differs by >2 semitones, or voiced_ratio < 0.3
- `rewrite_shorter` — generated duration / target duration > 1.35
- `ok` — within thresholds

Results are written to `job/pitch_verification.json`. Use
`--fail-on-pitch-mismatch` to fail the job if any segment is flagged.

```bash
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 \
     SPEAKER_PROFILING=1 VERIFY_PITCH=1 FAIL_ON_PITCH_MISMATCH=1
```

### TTS engine selection

The wrapper supports three TTS engines via `--tts-engine`:

| Engine | Provider | Description |
|--------|----------|-------------|
| `openvoice` | 34 | OpenVoice V2 local voice cloning (default) |
| `qwen3-local` | 1 | Qwen3-TTS local built-in (Apple Silicon, MLX) |
| `omnivoice` | 2 | OmniVoice local API (no bridge yet — not supported in split mode) |

```bash
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 TTS_ENGINE=qwen3-local
```

A fallback engine can be configured with `--fallback-tts-engine`. If the
primary engine fails (nonzero exit), the wrapper retries with the
fallback. `make doctor` reports which engines are ready.

```bash
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 \
     TTS_ENGINE=qwen3-local FALLBACK_TTS_ENGINE=openvoice
```

#### Qwen3-local setup (independent of OpenVoice)

Qwen3-local runs on the same python that runs `run_personal_dub.py`
(`sys.executable`), NOT the OpenVoice or pyVideoTrans venvs. It is
intentionally independent of OpenVoice — you do **not** need OpenVoice
checkpoints installed to use `--tts-engine qwen3-local`.

```bash
make setup-qwen3      # installs mlx-audio + soundfile + librosa + model
make doctor-qwen3     # probes imports + model presence
make smoke-qwen3      # generates one real segment — the proof
```

The default model is `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`,
which includes the `speech_tokenizer/` subdir required by mlx-audio's
`post_load_hook`. **Do not** use the `aufklarer/...-MLX-4bit` repo —
it is the Base talker only and is missing the speech tokenizer, so
`model.generate()` fails with `ValueError: Speech tokenizer not loaded`.

`make smoke-qwen3` is the real proof: it loads the model, clones the
default reference voice, writes a 24 kHz WAV, and checks it is
non-empty and non-silent. The Qwen3 bridge API (`load_model` →
`model.generate(text, ref_audio, ref_text, ...)` → `results[0].audio`)
is validated by this smoke test on each Mac before use.

### Job directory safety

The wrapper will **never** delete an existing job directory by default.
If you pass `--job-dir` pointing to an existing directory, the run will
fail with a clear error. To overwrite a job dir:

1. It must be under `pyvideotrans-main/tmp/personal_dub/`
2. It must contain a `.job-created-by-movie-dub` marker file
3. You must pass `--force-overwrite-job-dir`

External directories are never deleted automatically.

### Demucs fallback and model download policy

The default vocal separation method is **ffmpeg** (center-channel pan filter).
Demucs (AI-based separation) is opt-in and requires an explicit flag:

```bash
# Default: ffmpeg vocal separation (fast, no model download)
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 VOCAL_SEPARATION=1

# Demucs: requires model download permission
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 \
     VOCAL_SEPARATION=1 VOCAL_SEPARATION_METHOD=demucs ALLOW_DEMUCS_DOWNLOAD=1
```

Demucs will **never** download its model at runtime unless you pass
`--allow-demucs-download` (or `ALLOW_DEMUCS_DOWNLOAD=1` in the Makefile).
If the model is not already cached and download is not allowed, the pipeline
falls back to ffmpeg automatically.

If Demucs is not installed or crashes during inference, the pipeline also
falls back to ffmpeg. The build audio report includes `demucs_used`,
`demucs_error`, and `ffmpeg_fallback_used` fields so you can verify which
method was used.

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

## v0.12: Quality, Caching, Review

The biggest improvements are not more models — they are pipeline quality,
caching, validation, and review tooling.

### Real age model (optional plugin)

Age estimation uses a real regressor when available and falls back to the
pitch heuristic otherwise. The plugin is
`griko/age_reg_ann_ecapa_librosa_combined` (SpeechBrain ECAPA-TDNN + ANN,
~6.93y MAE). It runs in its own dedicated venv (`.venv-age`) and the model
weights are downloaded into `models/` (gitignored) — never cloned into the
repo.

**Architecture:** the model runs only on clean per-speaker reference clips,
never on the whole movie, background audio, mixed music/effects, overlapping
dialogue, or TTS-generated audio.

```bash
# 1. Install the plugin + download the model (one-time)
make setup-age
# Optional: install git-xet first if Hugging Face requires it for large files
#   brew install git-xet && git xet install

# 2. Verify it loads
make doctor-age

# 3. Smoke-test on a reference WAV
make smoke-age

# 4. Check status from the bare interpreter
make age-model-status

# 5. Use it in a dub (auto = try model, fall back to heuristic)
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 SPEAKER_PROFILING=1 \
     AGE_MODEL=auto AGE_MODEL_PATH=models/age_reg_ann_ecapa_librosa_combined
```

`--age-model on` requires the model and fails if unavailable; `off` uses the
heuristic only; `auto` (default) tries the model and falls back silently.
`--age-model-path` points at the local model dir (preferred over a runtime
HuggingFace download). Each speaker profile records `age.source` (`model` or
`heuristic`), `age.method` (`griko/age_reg_ann_ecapa_librosa_combined` or
`pitch_heuristic`), and `age.note` (apparent vocal age caveat) so you know
which estimate was used. `make doctor` reports venv + model-dir readiness as
an optional check.

The standalone wrapper (`scripts/estimate_speaker_age.py`) runs a single WAV
through the model outside the pipeline:

```bash
source .venv-age/bin/activate
python scripts/estimate_speaker_age.py \
  --audio voices/male_reference.wav \
  --model-path models/age_reg_ann_ecapa_librosa_combined \
  --output tmp/age_test.json
```

### Character profiles

`character_profiles.json` upgrades speaker profiles into named, reviewable
characters with stable ids, locked voices, and per-segment membership.

```bash
# Generate alongside a dub
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 SPEAKER_PROFILING=1 CHARACTER_PROFILES=1

# Or build from an existing speaker_profiles.json
make character-profiles PROFILES=<job>/speakers/speaker_profiles.json \
     OUTPUT=<job>/character_profiles.json TTS_ENGINE=qwen3-local

# Rename, lock a voice, mark review status (idempotent — preserves edits)
make character-rename PROFILE_FILE=<job>/character_profiles.json CHAR_ID=CHAR_001 CHAR_NAME="Alice"
make character-lock PROFILE_FILE=<job>/character_profiles.json CHAR_ID=CHAR_001 REF=voices/alice.wav
make character-review PROFILE_FILE=<job>/character_profiles.json CHAR_ID=CHAR_001 STATUS=approved
```

Schema: `character_id`, `speaker_id`, `name`, `gender`, `age_band`,
`estimated_years`, `median_f0_hz`, `voice_reference`, `voice_locked`,
`segments[]`, `tts_engine`, `review_status`. Re-running the builder
preserves user-set names, locks, and review status by matching on
`speaker_id`.

A **locked voice** (`voice_locked: true` + `voice_reference`) is the routing
authority for TTS: it overrides the speaker profile's `reference_audio` when
building the TTS queue. Lock a character's voice to pin it to a specific
reference WAV across re-runs, even if speaker profiling re-extracts a
different reference clip.

### Cache and resume

Every stage is resumable. The split pipeline uses a **hash-based segment
cache**: each TTS segment's cache key is derived from the source audio hash,
subtitle hash, speaker profile hash, translated text, TTS engine + model,
reference voice hash, language, and rate/volume/pitch. If none of those
inputs changed since the last run of the same job, the cached WAV is reused
and the bridge is skipped. Changing one line's text invalidates only that
segment — not the whole movie.

```bash
# Resume the most recent job, skipping stages already marked pass
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 RESUME=1

# Start at a specific stage (aliases: subs/stt/sts/profile/tts/audio/remux/verify)
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 FROM_STAGE=tts

# Regenerate a single segment, then rebuild audio + remux
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 ONLY_SEGMENT=42

# Skip any segment whose output WAV already exists (incremental)
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 SKIP_EXISTING=1

# Disable the hash-based cache (force full TTS regeneration)
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 NO_CACHE=1
```

`--resume` reuses the most recent job dir (or `--job-dir`); `--from-stage`
skips earlier stages; `--only-segment` runs TTS for one segment then
rebuilds; `--skip-existing` drops segments with existing output WAVs and
merges regenerated results back into the manifest. The hash-based cache is
on by default; `--no-cache` disables it. Cache state lives at
`<job_dir>/cache/segment_cache.json`.

### Better reference clip selection

Reference clips are now quality-scored on voiced ratio, single-speaker
overlap, background noise, volume normality, and length (sweet spot 5–15s),
instead of just "longest run near the middle." Falls back to the legacy
heuristic when librosa is unavailable.

```bash
make dub INPUT=movie.mp4 OUTPUT=dubbed.mp4 SPEAKER_PROFILING=1 REFERENCE_CLIP_SCORING=on
```

### Segment review / regeneration loop

After a dub, the job directory contains `review_segments.json`,
`pitch_verification.json`, `speaker_profiles.json`, and
`failed_segments.json`. You can then regenerate individual segments,
re-assign speakers, swap reference clips, or shorten lines:

```bash
# List everything needing attention
make failed-segments JOB=<job_dir>

# Regenerate one segment with a different speaker's voice
make regenerate JOB=<job_dir> SEGMENT=12 REMUX=1 \
     TTS_ENGINE=qwen3-local CHANGE_SPEAKER=SPEAKER_01

# Regenerate with a custom reference clip
make regenerate JOB=<job_dir> SEGMENT=12 REMUX=1 CHANGE_REF=voices/alice.wav

# Regenerate with a shortened line (truncates to fit the segment duration)
make regenerate JOB=<job_dir> SEGMENT=12 REMUX=1 SHORTEN=1

# Re-assign a whole speaker's segments to a new speaker, then regenerate
make change-speaker JOB=<job_dir> FROM_SPEAKER=SPEAKER_00 TO_SPEAKER=SPEAKER_02
```

`regenerate_segment.py` now supports `--tts-engine qwen3-local` (not just
OpenVoice), `--change-speaker`, `--change-reference`, and `--shorten`.

### Benchmark harness

A repeatable benchmark runs the full pipeline on a clip and records a
metrics report. Supply your own 5–10 minute clip and judge every change
against it.

```bash
make benchmark INPUT=~/Movies/bench_clip.mp4 \
     OUTPUT_DIR=reports/benchmarks/run1 \
     TTS_ENGINE=qwen3-local SPEAKER_PROFILING=1 HF_TOKEN=$HF_TOKEN \
     VERIFY_PITCH=1 PREFER_EMBEDDED_SUBTITLES=1 TARGET_LUFS=-16 \
     AGE_MODEL=auto CHARACTER_PROFILES=1
```

Tracked metrics: total render time, per-stage times, TTS time per segment
(mean/p50/p90/max), segment success rate, timing drift, loudness mismatch,
failed segments, bad speaker assignments, bad age/gender guesses, speaker
F0 consistency, review-required count, final MP4 duration delta, and a
composite quality score (0–100). The report is written to
`<OUTPUT_DIR>/benchmark_report.json`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `make doctor` fails on Demucs | Demucs not installed | Demucs is optional; default doctor should pass. Use `make doctor-demucs` only if you need it. |
| `make doctor` fails on checkpoints | OpenVoice V2 checkpoints missing | Run `make download-openvoice` |
| `make doctor` fails on ffmpeg/ffprobe | ffmpeg not installed | Run `make setup-ffmpeg` |
| Job dir already exists error | `--job-dir` points to existing dir | Use `--force-overwrite-job-dir` (only works for marked dirs under `tmp/personal_dub/`) |
| Demucs fallback to ffmpeg | Demucs not installed, crashed, or model not cached | Install Demucs + `--allow-demucs-download`, or accept ffmpeg fallback (check `build_audio_report.json`) |
| Demucs model download at runtime | Default refuses download | Pass `--allow-demucs-download` explicitly |
| LUFS output has wrong sample rate | Old version used default 44100 | Fixed: LUFS now preserves the selected sample rate |
| Final MP4 missing audio stream | Remux produced bad output | Verification now catches this; check `verification` in report |
| Segment regeneration didn't update video | `--remux` not passed | Always pass `--remux` to rebuild audio + video |
| `proof_report.json` says fail | One of the proof checks failed | Read the `failed_check` field in the report |
| Review says "ok" for failed segment | Old version ignored manifest status | Fixed: review now propagates manifest `status` field |
| Bridge crashes on torch.load | Checkpoint is LFS pointer or corrupted | Bridge now validates checkpoints before loading; run `make download-openvoice` |
| `--allow-partial` not working | Old version didn't wire it to provider | Fixed: `openvoice_allow_partial` now set in config |
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
