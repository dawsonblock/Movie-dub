#!/usr/bin/env python3
"""Hash-based caching + stage resume for the dub pipeline.

Two concerns are handled here:

1. **Segment TTS cache** — the expensive part. A generated segment WAV is
   keyed by a hash of everything that determines its content:

       source audio hash | subtitle hash | speaker profile hash |
       translated text hash | TTS engine + model hash | reference audio hash

   If one line changes, only that line is regenerated. The cache is stored
   per job under ``<job_dir>/cache/segment_cache.json`` and the WAVs under
   ``<job_dir>/cache/segments/``.

2. **Stage resume** — every pipeline stage is resumable. The job.json
   ``stages`` map records ``pending/running/pass/fail/skipped`` per stage.
   ``--resume`` reuses a job dir and skips stages already marked ``pass``.
   ``--from-stage`` starts at a given stage regardless of prior status.
   ``--skip-existing`` skips individual segments whose output WAV already
   exists and is valid. ``--only-segment`` runs a single segment.

This module imports no third-party packages and is safe to import from the
bare interpreter.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = "1.0"

# Canonical pipeline stage order. Used by --from-stage to know which stages
# to (re)run and which to skip.
PIPELINE_STAGES = (
    "subtitle_extraction",
    "transcription",
    "translation",
    "speaker_profiling",
    "tts",
    "audio_build",
    "remux",
    "verification",
)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def content_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()[:16]


def text_hash(text: str) -> str:
    """SHA-256 of a string, truncated to 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dict_hash(data: dict) -> str:
    """Stable hash of a JSON-serializable dict (sorted keys)."""
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def segment_cache_key(
    *,
    source_audio_hash: str,
    subtitle_hash: str,
    speaker_profile_hash: str,
    translated_text: str,
    tts_engine: str,
    tts_model: str,
    reference_audio_hash: str,
    language: str = "",
    voice_rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    tts_speed: float = 1.0,
) -> str:
    """Build a deterministic cache key for one TTS segment output.

    The key changes if ANY input that affects the generated audio changes:
    the source audio, the subtitle timing source, the speaker profile (which
    includes diarization), the translated text, the TTS engine + model, the
    reference voice, or the synthesis parameters.
    """
    parts = [
        "v2",
        source_audio_hash,
        subtitle_hash,
        speaker_profile_hash,
        text_hash(translated_text),
        tts_engine,
        tts_model,
        reference_audio_hash,
        language,
        voice_rate,
        volume,
        pitch,
        str(tts_speed),
    ]
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Segment cache index
# ---------------------------------------------------------------------------

class SegmentCache:
    """Per-job TTS segment cache.

    Stores a JSON index mapping cache keys -> {output_wav, hash, mtime, ts}.
    On a re-run, a segment whose key matches an existing entry and whose
    output WAV still exists + matches the recorded hash is skipped.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.segments_dir = cache_dir / "segments"
        self.index_path = cache_dir / "segment_cache.json"
        self._index: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("entries", {})
        except Exception:
            pass
        return {}

    def save(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": self._index,
        }
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, key: str) -> dict | None:
        """Return the cached entry if its output WAV exists + hash matches."""
        entry = self._index.get(key)
        if not entry:
            return None
        wav = Path(entry.get("output_wav", ""))
        if not wav.is_file():
            return None
        # Verify the file hasn't been tampered with since caching.
        if entry.get("hash") and content_hash(wav) != entry["hash"]:
            return None
        return entry

    def put(self, key: str, output_wav: Path) -> None:
        """Record a generated segment WAV in the cache."""
        wav_hash = content_hash(output_wav) if output_wav.is_file() else ""
        self._index[key] = {
            "output_wav": output_wav.as_posix(),
            "hash": wav_hash,
            "mtime": output_wav.stat().st_mtime if output_wav.is_file() else 0.0,
            "ts": time.time(),
        }

    def segments_path(self, key: str, segment_id: str) -> Path:
        """Return the canonical cached WAV path for a cache key."""
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(segment_id))
        return self.segments_dir / f"{key}_{safe_id}.wav"

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._index), "hit": 0, "miss": 0}


# ---------------------------------------------------------------------------
# Stage resume logic
# ---------------------------------------------------------------------------

class ResumePlan:
    """Decide which stages to run/skip for a resume.

    Parameters
    ----------
    stages
        The job.json ``stages`` dict (stage -> status).
    resume
        True if --resume: skip stages already ``pass``.
    from_stage
        Stage name to start at (--from-stage). Stages before it are skipped
        (assumed done) unless they never ran. None disables this.
    only_segment
        Segment id for --only-segment. When set, only the TTS stage runs and
        only for that segment.
    skip_existing
        True if --skip-existing: at the segment level, skip segments whose
        output WAV already exists.
    """

    def __init__(
        self,
        stages: dict[str, str],
        *,
        resume: bool = False,
        from_stage: str | None = None,
        only_segment: str | None = None,
        skip_existing: bool = False,
    ) -> None:
        self.stages = dict(stages)
        self.resume = resume
        self.from_stage = from_stage
        self.only_segment = only_segment
        self.skip_existing = skip_existing

    def should_run_stage(self, stage: str) -> bool:
        """True if ``stage`` should (re)run under this plan."""
        if self.only_segment:
            # Only the TTS stage runs (for one segment), plus downstream
            # audio_build + remux to materialize the change.
            return stage in ("tts", "audio_build", "remux", "verification")
        if self.from_stage:
            try:
                start_idx = PIPELINE_STAGES.index(self.from_stage)
            except ValueError:
                return True
            idx = PIPELINE_STAGES.index(stage) if stage in PIPELINE_STAGES else -1
            return idx >= start_idx
        if self.resume:
            # Skip stages already passing; re-run everything else.
            return self.stages.get(stage) != "pass"
        return True

    def stage_order(self) -> list[str]:
        """Return the stages to run, in canonical order."""
        return [s for s in PIPELINE_STAGES if self.should_run_stage(s)]

    def segment_should_run(self, segment_id: str, output_wav: Path | None) -> bool:
        """True if a single TTS segment should (re)run."""
        if self.only_segment and str(segment_id) != str(self.only_segment):
            return False
        if self.skip_existing and output_wav and output_wav.is_file():
            return False
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "resume": self.resume,
            "from_stage": self.from_stage,
            "only_segment": self.only_segment,
            "skip_existing": self.skip_existing,
            "stages_to_run": self.stage_order(),
            "stages_to_skip": [
                s for s in PIPELINE_STAGES if not self.should_run_stage(s)
            ],
        }


def parse_from_stage(value: str) -> str | None:
    """Validate a --from-stage value against canonical stage names.

    Accepts a few aliases for convenience.
    """
    if not value:
        return None
    aliases = {
        "subs": "subtitle_extraction",
        "subtitles": "subtitle_extraction",
        "stt": "transcription",
        "asr": "transcription",
        "transcribe": "transcription",
        "sts": "translation",
        "translate": "translation",
        "profile": "speaker_profiling",
        "speakers": "speaker_profiling",
        "tts": "tts",
        "audio": "audio_build",
        "build": "audio_build",
        "remux": "remux",
        "mux": "remux",
        "verify": "verification",
        "verification": "verification",
    }
    v = value.lower().strip()
    if v in PIPELINE_STAGES:
        return v
    if v in aliases:
        return aliases[v]
    raise ValueError(
        f"unknown stage '{value}'. Valid stages: "
        f"{', '.join(PIPELINE_STAGES)} (or aliases: "
        f"{', '.join(sorted(aliases))})"
    )


# ---------------------------------------------------------------------------
# CLI: inspect a cache index
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a segment cache index")
    parser.add_argument("--cache-dir", required=True, help="job cache/ dir")
    parser.add_argument("--summary", action="store_true", help="print summary only")
    args = parser.parse_args()
    cache = SegmentCache(Path(args.cache_dir))
    if args.summary:
        print(json.dumps(cache.stats(), indent=2))
        return 0
    print(json.dumps({"entries": cache._index}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
