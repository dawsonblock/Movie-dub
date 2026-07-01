"""Tests that OpenVoice is no longer required anywhere in the pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DUB = ROOT / "scripts" / "run_personal_dub.py"


def test_run_personal_dub_help_rejects_openvoice():
    """--help should not list openvoice as a valid TTS engine choice."""
    result = subprocess.run(
        [sys.executable, RUN_DUB.as_posix(), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "openvoice" not in out
    assert "qwen3-local" in out
    assert "omnivoice" in out


def test_run_personal_dub_rejects_openvoice_engine():
    """Passing --tts-engine openvoice must fail with a clear removal message."""
    result = subprocess.run(
        [
            sys.executable, RUN_DUB.as_posix(),
            "--input", "/dev/null",
            "--output", "/dev/null",
            "--tts-engine", "openvoice",
        ],
        capture_output=True, text=True, timeout=30,
    )
    combined = result.stdout + result.stderr
    assert "OpenVoice has been removed" in combined, combined
    assert result.returncode != 0


def test_no_openvoice_bridge_file():
    """The OpenVoice bridge script must be deleted."""
    assert not (ROOT / "bridge" / "openvoice_segment_tts.py").exists()


def test_no_openvoice_checkpoints_dir():
    """The OpenVoice checkpoints directory must be deleted."""
    assert not (ROOT / "OpenVoice-main").exists()
