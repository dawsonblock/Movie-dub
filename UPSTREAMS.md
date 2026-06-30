# Upstream Projects

This repository bundles the following upstream projects. Each has its own
license, contributing guidelines, and maintainers. Please respect their
terms when using or redistributing this bundle.

## pyVideoTrans

- **Repository:** https://github.com/jianchang512/pyvideotrans
- **License:** GPLv3 (see `pyvideotrans-main/LICENSE`)
- **Purpose:** Video transcription, translation, subtitle generation,
  timing, mixing, and export orchestration.
- **Local path:** `pyvideotrans-main/`

## OpenVoice

- **Repository:** https://github.com/myshell-ai/OpenVoice
- **License:** Apache 2.0 (see `OpenVoice-main/LICENSE`)
- **Purpose:** Local voice cloning and tone color conversion using
  OpenVoice V2. Also includes MeloTTS for base speech synthesis.
- **Local path:** `OpenVoice-main/`

## OpenVoice V2 Checkpoints

- **Repository:** https://huggingface.co/rsxdalv/OpenVoiceV2
- **License:** Refer to the upstream model card.
- **Purpose:** Pre-trained OpenVoice V2 converter and base speaker
  embeddings.
- **Local path:** `OpenVoice-main/checkpoints_v2/` (not committed; downloaded
  via `make download-openvoice`)

## ffmpeg / ffprobe

- **Repository:** https://ffmpeg.org/
- **License:** LGPL or GPL depending on build configuration.
- **Purpose:** Audio/video encoding, decoding, and remuxing.
- **Local path:** `pyvideotrans-main/ffmpeg/` (bundled binary, macOS)

## How to update upstreams

To update a bundled upstream project, replace the contents of its local
directory with a fresh checkout from the upstream repository. Then run
`make doctor` and `make proof` to verify the bundle still works together.

## Local patches to upstream code

The following patches have been applied to the bundled pyVideoTrans tree.
They are minimal and safe — re-apply them after updating pyVideoTrans.

### `videotrans/tts/__init__.py`

No patches. We only read the provider index constants (`OPENVOICE_TTS=34`,
`QWEN3LOCAL_TTS=1`, `OMNIVOICE_TTS=2`) and the `SUPPORT_CLONE` list.

### `videotrans/process/prepare_audio.py`

No patches. We reuse `pyannote_speakers()` and `built_speakers()` from
`scripts/analyze_speakers.py` via direct import. The upstream functions
are called as-is.

### `videotrans/task/trans_create.py`

**Planned patch (not yet applied):** `_create_ref_from_vocal()` currently
cuts a fresh `clone-{i}.wav` from the vocal track for every clone segment.
When speaker profiling is active, `run_personal_dub.py` pre-places
canonical per-speaker reference WAVs. The patch will add a 3-line guard to
skip cutting if `ref_wav` already exists. This is not yet needed because
the current wiring uses `line_roles` + `openvoice_default_reference` rather
than the clone path. The patch becomes necessary when we switch to
per-segment reference routing through the clone role.

### `cli.py`

No patches. The cache folder uuid is deterministic (md5 of
`name-size-mtime`), so `run_personal_dub.py` can predict the SRT output
path without a CLI flag. If this changes upstream, a `--cache-dir` flag
will need to be added.

## pyannote speaker-diarization-3.1

- **Repository:** https://huggingface.co/pyannote/speaker-diarization-3.1
- **License:** Refer to the upstream model card. Requires accepting the
  license terms on HuggingFace and providing an access token.
- **Purpose:** Speaker diarization for the `--speaker-profiling` flow.
- **Used by:** `scripts/analyze_speakers.py` (via `--diarization pyannote`)
- **Token:** Set `HF_TOKEN` env var or pass `--hf-token`. The token is
  NOT committed to the repo.
