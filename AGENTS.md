# Movie-dub — agent notes

Build/test commands and conventions learned while working in this repo.

## Verification commands

- **Unit tests:** `make test` or `python3 -m pytest tests/ -q` (238 tests, lightweight, no model deps)
- **Lint:** `ruff check scripts/` (must be clean; repo history shows lint fixes are expected)
- **Doctor:** `python3 scripts/doctor.py` → must print `READY: yes` with 0 required failures
- **E2E smoke:** `python3 scripts/smoke_e2e_short_clip.py` (needs OpenVoice checkpoints; ~minutes)
- **Age model status:** `python3 scripts/age_model.py` (exit 1 = unavailable, falls back to heuristic)
- **Syntax check a script:** `python3 -c "import ast; ast.parse(open('scripts/x.py').read())"`

## macOS / environment notes

- **Whisper MPS out-of-memory:** On Apple Silicon, `whisper-large-v3-turbo` can
  exhaust the MPS memory pool. Use `CPU_ONLY=1` on `make dub` / `make
  benchmark` to force Whisper to CPU (OpenVoice still uses MPS).
- **HF_TOKEN:** Speaker profiling (`SPEAKER_PROFILING=1`) requires the
  `pyannote/speaker-diarization-3.1` model, which is gated. Set a valid token
  and accept the model terms at https://hf.co/pyannote/speaker-diarization-3.1
  and https://hf.co/pyannote/segmentation-3.0.

## Test suite

Tests live in `tests/` and use pytest. `conftest.py` adds `scripts/` to
`sys.path`. No third-party deps beyond pytest itself (the age model is mocked).

- `test_age_model.py` — band mapping, source resolution, fallback, singleton
- `test_cache.py` — hash determinism, cache key sensitivity, SegmentCache
  round-trip + tamper detection, ResumePlan all modes, parse_from_stage aliases
- `test_cache_wiring.py` — SegmentCache integration into run_personal_dub.py:
  `_locked_voice_overrides`, voice-lock routing in `build_queue_tts_with_speakers`,
  `_compute_segment_cache_keys` determinism + input sensitivity, cache hit/miss flow
- `test_character_profiles.py` — build, idempotency (name/lock/review
  preservation), mutations (rename/lock/unlock/set-review), schema validation
- `test_review_loop.py` — failed_segments merging from 3 sources, dedup,
  speaker/reference changes, shorten_text edge cases
- `test_benchmark.py` — collect_metrics from fixture job dir, quality score,
  percentile helper
- `test_estimate_speaker_age.py` — age_band, mocked estimate_age, CLI
  validation, output writing
- `test_pipeline_flags.py` — argparse wiring (subprocess --help) for all
  pipeline scripts

## Architecture

- Two isolated venvs: `pyvideotrans-main/.venv` (pyannote/librosa/whisper) and
  `OpenVoice-main/.venv` (OpenVoice/MeloTTS). Never merge them.
- `scripts/run_personal_dub.py` is the main CLI (~1900 lines). Two pipelines:
  - **VTV pipeline** (default): pyVideoTrans `cli.py --task vtv`
  - **Split pipeline** (`--speaker-profiling`): STT → STS → analyze_speakers → direct bridge → build audio → remux
- Bridges live in `bridge/` (`openvoice_segment_tts.py`, `qwen3_segment_tts.py`).
  Both accept the same `queue_tts` JSON and emit the same manifest format.
- Job state: `scripts/job_state.py` is the single source of truth for job dirs
  (`pyvideotrans-main/tmp/personal_dub/<job_id>/`) and `job.json` stage status.
- Stage names (canonical order): `subtitle_extraction, transcription, translation,
  speaker_profiling, tts, audio_build, remux, verification`.

## v0.12 modules

- `scripts/age_model.py` — age regressor plugin + pitch heuristic fallback.
  Lazy-loaded singleton; safe to import from bare interpreter. Installed into
  `.venv-age` (created by `make setup-age`). Model weights live in
  `models/age_reg_ann_ecapa_librosa_combined` (gitignored, fetched via
  `hf download`, never `git clone`). Supports `model_path` (local dir
  preferred over HF repo id), `young_adult` band, and a `method` field.
  The Python package is `voice_age_regressor` (import
  `from voice_age_regressor import AgeRegressionPipeline`). When the main
  interpreter (e.g. pyVideoTrans venv) cannot import `voice_age_regressor`,
  `estimate_age` transparently falls back to a `.venv-age` subprocess
  (`scripts/age_model_subprocess.py`) so the full pipeline still gets the
  trained model. **Proven operational on macOS ARM64**: `make setup-age` +
  `make doctor-age` + `make smoke-age` all pass; male ref → 26.5y
  (young_adult, conf 0.78), female ref → 33.9y (young_adult).
- `scripts/estimate_speaker_age.py` — standalone CLI wrapper for one WAV.
  Run inside `.venv-age`.
- `scripts/setup_age_model.sh` — creates `.venv-age`, installs
  `voice-age-regression[full]`, downloads the model into `models/`.
- `scripts/character_profiles.py` — builds `character_profiles.json` from
  `speaker_profiles.json`. Idempotent (preserves names/locks/review_status).
- `scripts/cache.py` — hash-based segment cache + `ResumePlan` for
  `--resume/--from-stage/--only-segment/--skip-existing`. The `SegmentCache`
  is wired into `run_personal_dub.py`'s split-pipeline TTS stage: each
  segment's cache key is derived from source audio hash, subtitle hash,
  speaker profile hash, translated text, TTS engine/model, reference audio
  hash, language, and rate/volume/pitch. Cache hits skip the bridge; cache
  misses generate TTS and then record the output WAV. `--no-cache` disables
  this. Cache lives at `<job_dir>/cache/`.
- `scripts/review_loop.py` — `failed_segments.json` writer + speaker/reference
  change appliers + `shorten_text`.
- `scripts/benchmark.py` — repeatable benchmark harness wrapping
  `run_personal_dub.py`; writes `benchmark_report.json` with metrics + quality score.
- Character voice locks: `build_queue_tts_with_speakers()` accepts an optional
  `character_profiles` dict. A character with `voice_locked=True` and a
  `voice_reference` overrides the speaker profile's `reference_audio` during
  TTS queue building. This makes `character_profiles.json` the routing
  authority for locked voices, not just a review artifact.

## Conventions

- New scripts must be import-safe (no third-party imports at module top level
  when they may run under the bare system interpreter). Heavy imports go inside
  functions.
- `die(msg)` prints `<script>: FAIL\nReason: msg` to stderr and returns exit code.
- Makefile targets use tabs; verify with `make -n <target>` after editing.
- The dub Makefile target forwards flags via `$(if $(VAR),--flag $(VAR))`.
- Makefile variables MUST use project-specific names, never generic ones
  that collide with environment variables. Make imports env vars automatically,
  so a stray `TARGET` in the shell silently corrupts `--target-language`.
  Use `SOURCE_LANG`/`TARGET_LANG` (not `SOURCE`/`TARGET`), `ASR_MODEL`
  (not `MODEL`), `PROFILE_FILE` (not `FILE`), `FROM_SPEAKER`/`TO_SPEAKER`
  (not `FROM`/`TO`), `CHAR_NAME`/`RUN_NAME` (not `NAME`).
