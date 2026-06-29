"""Tests for job-scoped artifacts and job.json (Blockers 5, 12)."""

import importlib.util
import json
import sys
from pathlib import Path


JOB_STATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "job_state.py"
SPEC = importlib.util.spec_from_file_location("job_state", JOB_STATE_PATH)
job_state = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = job_state
SPEC.loader.exec_module(job_state)


def test_job_paths_layout(tmp_path):
    """JobPaths should provide all required artifact paths (Blocker 5)."""
    paths = job_state.JobPaths(tmp_path / "test_job")
    assert paths.source_video.name == "source.mp4"
    assert paths.queue_file.name == "queue.json"
    assert paths.manifest_file.name == "manifest.json"
    assert paths.dubbed_audio.name == "dubbed_audio.wav"
    assert paths.build_audio_report.name == "build_audio_report.json"
    assert paths.final_video.name == "final.mp4"
    assert paths.remux_command.name == "remux_command.json"
    assert paths.run_report.name == "run_report.json"
    assert paths.proof_report.name == "proof_report.json"
    assert paths.job_json.name == "job.json"
    assert paths.review_segments.name == "review_segments.json"


def test_job_paths_mkdirs(tmp_path):
    """JobPaths.mkdirs should create all subdirectories (Blocker 5)."""
    paths = job_state.JobPaths(tmp_path / "test_job")
    paths.mkdirs()
    assert paths.input_dir.is_dir()
    assert paths.openvoice_dir.is_dir()
    assert paths.segments_dir.is_dir()
    assert paths.audio_dir.is_dir()
    assert paths.video_dir.is_dir()
    assert paths.subtitles_dir.is_dir()
    assert paths.reports_dir.is_dir()
    assert paths.logs_dir.is_dir()


def test_initial_job_state_structure():
    """initial_job_state should have the required schema (Blocker 12)."""
    state = job_state.initial_job_state(
        job_id="20260629-142211-a83f92",
        input_video="/input.mp4",
        output_video="/output.mp4",
        reference_voice="/ref.wav",
        source_language="en",
        target_language="es",
        job_dir="/job",
    )
    assert state["schema_version"] == "1.0"
    assert state["job_id"] == "20260629-142211-a83f92"
    assert state["status"] == "running"
    assert state["input_video"] == "/input.mp4"
    assert state["output_video"] == "/output.mp4"
    assert state["reference_voice"] == "/ref.wav"
    assert state["language"]["source"] == "en"
    assert state["language"]["target"] == "es"
    assert "paths" in state
    assert "stages" in state
    for stage in ("transcription", "translation", "tts", "audio_build", "remux", "verification"):
        assert state["stages"][stage] == "pending"
    assert state["result"] is None


def test_write_and_read_job_state(tmp_path):
    """write_job_state + read_job_state should round-trip (Blocker 12)."""
    job_json = tmp_path / "job.json"
    state = job_state.initial_job_state(
        job_id="test-123",
        input_video="/in.mp4",
        output_video="/out.mp4",
        job_dir=str(tmp_path),
    )
    job_state.write_job_state(job_json, state)
    read = job_state.read_job_state(job_json)
    assert read is not None
    assert read["job_id"] == "test-123"
    assert read["status"] == "running"


def test_update_job_stage(tmp_path):
    """update_job_stage should update a single stage (Blocker 12)."""
    job_json = tmp_path / "job.json"
    state = job_state.initial_job_state(
        job_id="test-456",
        input_video="/in.mp4",
        output_video="/out.mp4",
        job_dir=str(tmp_path),
    )
    job_state.write_job_state(job_json, state)
    updated = job_state.update_job_stage(job_json, "tts", "pass")
    assert updated is not None
    assert updated["stages"]["tts"] == "pass"
    assert updated["stages"]["transcription"] == "pending"


def test_update_job_stage_fail_marks_failed(tmp_path):
    """A failed stage should mark the job as failed (Blocker 12)."""
    job_json = tmp_path / "job.json"
    state = job_state.initial_job_state(
        job_id="test-fail",
        input_video="/in.mp4",
        output_video="/out.mp4",
        job_dir=str(tmp_path),
    )
    job_state.write_job_state(job_json, state)
    updated = job_state.update_job_stage(job_json, "audio_build", "fail")
    assert updated["status"] == "failed"


def test_read_job_state_missing_returns_none(tmp_path):
    """read_job_state should return None for missing file (Blocker 12)."""
    assert job_state.read_job_state(tmp_path / "nonexistent.json") is None
