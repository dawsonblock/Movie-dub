"""Tests for the Qwen3-only doctor.py.

Verifies that doctor.py no longer treats OpenVoice as required and instead
reports Qwen3-local readiness.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCTOR = ROOT / "scripts" / "doctor.py"


def _doctor_help() -> str:
    result = subprocess.run(
        [sys.executable, DOCTOR.as_posix(), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_doctor_help_does_not_offer_openvoice_flags():
    out = _doctor_help()
    assert "--no-openvoice" not in out
    assert "--checkpoints-only" not in out


def test_doctor_help_offers_qwen3_flag():
    out = _doctor_help()
    assert "--qwen3-only" in out


def test_doctor_json_does_not_require_openvoice():
    """The default JSON report must have no required OpenVoice failures."""
    result = subprocess.run(
        [sys.executable, DOCTOR.as_posix(), "--json"],
        capture_output=True, text=True, timeout=120,
    )
    summary = json.loads(result.stdout)
    checks = summary.get("checks", [])
    required_failures = [c for c in checks if c.get("required") and not c.get("ok")]
    openvoice_required_failures = [
        c for c in required_failures
        if "openvoice" in c.get("name", "").lower()
        or "openvoice" in c.get("section", "").lower()
    ]
    assert openvoice_required_failures == [], (
        "doctor still reports OpenVoice as a required failure"
    )


def test_doctor_qwen3_only_reports_qwen3_imports():
    """--qwen3-only should mention the Qwen3 imports and model."""
    result = subprocess.run(
        [sys.executable, DOCTOR.as_posix(), "--qwen3-only"],
        capture_output=True, text=True, timeout=120,
    )
    out = result.stdout + result.stderr
    assert "mlx_audio" in out or "mlx-audio" in out
    assert "soundfile" in out
    assert "Qwen3" in out
