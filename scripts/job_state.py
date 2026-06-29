#!/usr/bin/env python3
"""Shared job directory + job.json state management for Movie-dub.

Every dub job gets a unique, deterministic directory under
``pyvideotrans-main/tmp/personal_dub/<job_id>/`` with a marker file
``.job-created-by-movie-dub`` so that safe-deletion policy can be enforced.

This module is the single source of truth for:
  * job ID generation (``YYYYMMDD-HHMMSS-<short uuid>``)
  * safe job-dir creation / deletion (Blocker 3)
  * job.json read / write / stage updates (Blocker 12)
  * job-scoped path layout (Blocker 5)

Importing this module has no side effects and no third-party dependencies.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
PERSONAL_DUB_ROOT = PYVIDEOTRANS / "tmp" / "personal_dub"
JOB_MARKER = ".job-created-by-movie-dub"
JOB_STATE_FILE = "job.json"
JOB_SCHEMA_VERSION = "1.0"


class UnsafeJobDirError(RuntimeError):
    """Raised when a job-dir operation would be unsafe (Blocker 3)."""


def generate_job_id() -> str:
    """Return ``YYYYMMDD-HHMMSS-<short uuid>``."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    short = uuid.uuid4().hex[:6]
    return f"{stamp}-{short}"


def is_under(path: Path, root: Path) -> bool:
    """Return True if ``path`` is inside ``root`` (both resolved)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_delete_job_dir(
    job_dir: Path,
    force: bool = False,
    allow_external: bool = False,
) -> None:
    """Delete ``job_dir`` only if it is safe to do so (Blocker 3).

    Policy:
      * If ``job_dir`` does not exist, do nothing.
      * If ``force`` is False, raise ``UnsafeJobDirError``.
      * If ``job_dir`` is not under ``PERSONAL_DUB_ROOT`` and
        ``allow_external`` is False, raise ``UnsafeJobDirError``.
      * If the marker file ``.job-created-by-movie-dub`` is missing,
        raise ``UnsafeJobDirError``.
      * Otherwise, delete the directory tree.
    """
    if not job_dir.exists():
        return
    if not force:
        raise UnsafeJobDirError(
            f"Job dir already exists. Refusing to overwrite: {job_dir}"
        )
    if not is_under(job_dir, PERSONAL_DUB_ROOT) and not allow_external:
        raise UnsafeJobDirError(
            f"Refusing to delete external job dir (not under "
            f"{PERSONAL_DUB_ROOT}): {job_dir}"
        )
    marker = job_dir / JOB_MARKER
    if not marker.is_file():
        raise UnsafeJobDirError(
            f"Refusing to delete unmarked job dir (missing {JOB_MARKER}): "
            f"{job_dir}"
        )
    import shutil

    shutil.rmtree(job_dir)


def create_job_dir(
    job_dir: Path,
    force: bool = False,
    allow_external: bool = False,
) -> Path:
    """Create a fresh job directory with the marker file.

    If ``job_dir`` already exists, apply ``safe_delete_job_dir`` first.
    Returns the created ``job_dir``.
    """
    safe_delete_job_dir(job_dir, force=force, allow_external=allow_external)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / JOB_MARKER).write_text(
        f"created_by=movie-dub\ncreated_at={time.time()}\n",
        encoding="utf-8",
    )
    return job_dir


def default_job_dir(job_id: str | None = None) -> Path:
    """Return the default job directory path for a given or new job ID."""
    jid = job_id or generate_job_id()
    return PERSONAL_DUB_ROOT / jid


# --- Job-scoped path layout (Blocker 5) ---

class JobPaths:
    """Typed accessor for all job-scoped artifact paths (Blocker 5)."""

    def __init__(self, job_dir: Path) -> None:
        self.job_dir = job_dir
        self.input_dir = job_dir / "input"
        self.openvoice_dir = job_dir / "openvoice"
        self.segments_dir = job_dir / "openvoice" / "segments"
        self.audio_dir = job_dir / "audio"
        self.video_dir = job_dir / "video"
        self.subtitles_dir = job_dir / "subtitles"
        self.reports_dir = job_dir / "reports"
        self.logs_dir = job_dir / "logs"

    @property
    def source_video(self) -> Path:
        return self.input_dir / "source.mp4"

    @property
    def queue_file(self) -> Path:
        return self.openvoice_dir / "queue.json"

    @property
    def manifest_file(self) -> Path:
        return self.openvoice_dir / "manifest.json"

    @property
    def dubbed_audio(self) -> Path:
        return self.audio_dir / "dubbed_audio.wav"

    @property
    def build_audio_report(self) -> Path:
        return self.audio_dir / "build_audio_report.json"

    @property
    def final_video(self) -> Path:
        return self.video_dir / "final.mp4"

    @property
    def remux_command(self) -> Path:
        return self.video_dir / "remux_command.json"

    @property
    def source_srt(self) -> Path:
        return self.subtitles_dir / "source.srt"

    @property
    def translated_srt(self) -> Path:
        return self.subtitles_dir / "translated.srt"

    @property
    def run_report(self) -> Path:
        return self.reports_dir / "run_report.json"

    @property
    def proof_report(self) -> Path:
        return self.reports_dir / "proof_report.json"

    @property
    def job_json(self) -> Path:
        return self.job_dir / JOB_STATE_FILE

    @property
    def review_segments(self) -> Path:
        return self.job_dir / "review_segments.json"

    @property
    def generated_audio(self) -> Path:
        return self.openvoice_dir / "segments"

    @property
    def pyvt_log(self) -> Path:
        return self.logs_dir / "pyvideotrans.log"

    @property
    def openvoice_log(self) -> Path:
        return self.logs_dir / "openvoice_bridge.log"

    def mkdirs(self) -> None:
        """Create all job subdirectories."""
        for d in (
            self.input_dir,
            self.openvoice_dir,
            self.segments_dir,
            self.audio_dir,
            self.video_dir,
            self.subtitles_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# --- job.json single source of truth (Blocker 12) ---

def initial_job_state(
    job_id: str,
    input_video: str,
    output_video: str,
    reference_voice: str = "",
    source_language: str = "auto",
    target_language: str = "en",
    job_dir: str = "",
) -> dict[str, Any]:
    """Return the initial job.json structure."""
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "input_video": input_video,
        "output_video": output_video,
        "reference_voice": reference_voice,
        "language": {
            "source": source_language,
            "target": target_language,
        },
        "paths": {
            "job_dir": job_dir,
            "openvoice_queue": "",
            "openvoice_manifest": "",
            "dubbed_audio": "",
            "final_video": "",
            "review_segments": "",
        },
        "stages": {
            "transcription": "pending",
            "translation": "pending",
            "tts": "pending",
            "audio_build": "pending",
            "remux": "pending",
            "verification": "pending",
        },
        "result": None,
    }


def write_job_state(job_json_path: Path, state: dict[str, Any]) -> None:
    """Write the full job.json state."""
    job_json_path.parent.mkdir(parents=True, exist_ok=True)
    job_json_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_job_state(job_json_path: Path) -> dict[str, Any] | None:
    """Read job.json, returning None if it does not exist."""
    if not job_json_path.is_file():
        return None
    try:
        return json.loads(job_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def update_job_stage(
    job_json_path: Path,
    stage: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Update a single stage in job.json and return the new state.

    ``status`` should be one of: ``pending``, ``running``, ``pass``, ``fail``.
    """
    state = read_job_state(job_json_path)
    if state is None:
        return None
    stages = state.setdefault("stages", {})
    stages[stage] = status
    if extra:
        state.update(extra)
    if status == "fail":
        state["status"] = "failed"
    elif stage == "verification" and status == "pass":
        all_pass = all(
            v in ("pass", "skipped") for v in stages.values()
        )
        if all_pass:
            state["status"] = "pass"
            state["result"] = "pass"
    write_job_state(job_json_path, state)
    return state
