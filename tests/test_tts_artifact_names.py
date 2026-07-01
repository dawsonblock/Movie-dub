"""Tests that TTS artifacts use the new tts_* naming.

This verifies the constants inside run_personal_dub.py by checking the
job paths it writes to job.json.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DUB = ROOT / "scripts" / "run_personal_dub.py"


def test_job_json_paths_use_tts_names():
    """A dry-run or --help run should not create artifacts, but the path
    constants in run_personal_dub.py should use the tts_* names.
    We inspect the source code directly.
    """
    src = RUN_DUB.read_text(encoding="utf-8")
    assert "openvoice_manifest.json" not in src
    assert "openvoice_queue.json" not in src
    assert "openvoice_bridge.log" not in src
    assert "openvoice_work" not in src
    assert "openvoice_manifest_explicit.json" not in src
    assert "tts_manifest.json" in src
    assert "tts_queue.json" in src
    assert "tts_bridge.log" in src
    assert "tts_work" in src
    assert "tts_manifest_explicit.json" in src


def test_job_json_keys_use_tts_names():
    """The keys in job_state['paths'] must use tts_* names."""
    src = RUN_DUB.read_text(encoding="utf-8")
    assert '"tts_manifest"' in src
    assert '"tts_queue"' in src
    assert '"tts_logs"' in src
    assert '"openvoice_manifest"' not in src
    assert '"openvoice_queue"' not in src
    assert '"openvoice_logs"' not in src
