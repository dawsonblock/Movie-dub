"""Tests for safe job-dir deletion policy (Blocker 3)."""

import importlib.util
import sys
from pathlib import Path


JOB_STATE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "job_state.py"
SPEC = importlib.util.spec_from_file_location("job_state", JOB_STATE_PATH)
job_state = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = job_state
SPEC.loader.exec_module(job_state)


def test_external_job_dir_not_deleted(tmp_path):
    """External job dir should not be deleted even with force (Blocker 3)."""
    external = tmp_path / "important_dir"
    external.mkdir()
    (external / "data.txt").write_text("important", encoding="utf-8")

    # Should refuse to delete
    try:
        job_state.safe_delete_job_dir(external, force=True)
        assert False, "should have raised UnsafeJobDirError"
    except job_state.UnsafeJobDirError:
        pass

    # Directory should still exist
    assert external.is_dir()
    assert (external / "data.txt").is_file()


def test_unmarked_job_dir_not_deleted(tmp_path):
    """Unmarked job dir under personal_dub root should not be deleted (Blocker 3)."""
    # Create a dir under PERSONAL_DUB_ROOT without the marker file
    job_dir = job_state.PERSONAL_DUB_ROOT / "test_unmarked"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "data.txt").write_text("data", encoding="utf-8")
    try:
        try:
            job_state.safe_delete_job_dir(job_dir, force=True)
            assert False, "should have raised UnsafeJobDirError"
        except job_state.UnsafeJobDirError:
            pass
        assert job_dir.is_dir()
    finally:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)


def test_no_force_refuses_existing_dir(tmp_path):
    """Existing job dir without --force should be refused (Blocker 3)."""
    job_dir = tmp_path / "existing_job"
    job_dir.mkdir()
    (job_dir / "data.txt").write_text("data", encoding="utf-8")

    try:
        job_state.safe_delete_job_dir(job_dir, force=False)
        assert False, "should have raised UnsafeJobDirError"
    except job_state.UnsafeJobDirError:
        pass

    assert job_dir.is_dir()


def test_marked_job_dir_deleted_with_force(tmp_path):
    """Marked job dir under personal_dub root should be deleted with force (Blocker 3)."""
    job_dir = job_state.PERSONAL_DUB_ROOT / "test_marked_delete"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / job_state.JOB_MARKER).write_text("marker", encoding="utf-8")
    (job_dir / "data.txt").write_text("data", encoding="utf-8")

    try:
        job_state.safe_delete_job_dir(job_dir, force=True)
        assert not job_dir.exists(), "marked job dir should be deleted"
    finally:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)


def test_create_job_dir_creates_marker(tmp_path):
    """create_job_dir should create the marker file (Blocker 3)."""
    job_dir = job_state.PERSONAL_DUB_ROOT / "test_create_marker"
    try:
        job_state.create_job_dir(job_dir, force=False, allow_external=False)
        assert job_dir.is_dir()
        assert (job_dir / job_state.JOB_MARKER).is_file()
    finally:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)


def test_nonexistent_dir_noop():
    """safe_delete on non-existent dir should be a no-op (Blocker 3)."""
    # Should not raise
    job_state.safe_delete_job_dir(Path("/nonexistent/path/xyz"), force=False)


def test_generate_job_id_format():
    """Job ID should match YYYYMMDD-HHMMSS-<6 hex chars> format."""
    jid = job_state.generate_job_id()
    parts = jid.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 6  # HHMMSS
    assert len(parts[2]) == 6  # 6 hex chars
    int(parts[0])  # should be numeric
    int(parts[1])  # should be numeric
