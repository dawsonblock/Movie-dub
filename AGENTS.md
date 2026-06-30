# Movie-dub — agent notes

Build/test commands and conventions learned while working in this repo.

## Verification commands

- **Unit tests:** `make test` or `python3 -m pytest tests/ -q` (215 tests, ~2s, no model deps)
- **Lint:** `ruff check scripts/` (must be clean; repo history shows lint fixes are expected)
- **Doctor:** `python3 scripts/doctor.py` → must print `READY: yes` with 0 required failures
- **E2E smoke:** `python3 scripts/smoke_e2e_short_clip.py` (needs OpenVoice checkpoints; ~minutes)
- **Age model status:** `python3 scripts/age_model.py` (exit 1 = unavailable, falls back to heuristic)
- **Syntax check a script:** `python3 -c "import ast; ast.parse(open('scripts/x.py').read())"`

## Test suite

Tests live in `tests/` and use pytest. `conftest.py` adds `scripts/` to
`sys.path`. No third-party deps beyond pytest itself (the age model is mocked).

- `test_age_model.py` — band mapping, source resolution, fallback, singleton
- `test_cache.py` — hash determinism, cache key sensitivity, SegmentCache
  round-trip + tamper detection, ResumePlan all modes, parse_from_stage aliases
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
  Lazy-loaded singleton; safe to import from bare interpreter. Runs in a
  dedicated `.venv-age` (created by `make setup-age`). Model weights live in
  `models/age_reg_ann_ecapa_librosa_combined` (gitignored, fetched via
  `hf download`, never `git clone`). Supports `model_path` (local dir
  preferred over HF repo id), `young_adult` band, and a `method` field.
- `scripts/estimate_speaker_age.py` — standalone CLI wrapper for one WAV.
  Run inside `.venv-age`.
- `scripts/setup_age_model.sh` — creates `.venv-age`, installs
  `voice-age-regression[full]`, downloads the model into `models/`.
- `scripts/character_profiles.py` — builds `character_profiles.json` from
  `speaker_profiles.json`. Idempotent (preserves names/locks/review_status).
- `scripts/cache.py` — hash-based segment cache + `ResumePlan` for
  `--resume/--from-stage/--only-segment/--skip-existing`.
- `scripts/review_loop.py` — `failed_segments.json` writer + speaker/reference
  change appliers + `shorten_text`.
- `scripts/benchmark.py` — repeatable benchmark harness wrapping
  `run_personal_dub.py`; writes `benchmark_report.json` with metrics + quality score.

## Conventions

- New scripts must be import-safe (no third-party imports at module top level
  when they may run under the bare system interpreter). Heavy imports go inside
  functions.
- `die(msg)` prints `<script>: FAIL\nReason: msg` to stderr and returns exit code.
- Makefile targets use tabs; verify with `make -n <target>` after editing.
- The dub Makefile target forwards flags via `$(if $(VAR),--flag $(VAR))`.
