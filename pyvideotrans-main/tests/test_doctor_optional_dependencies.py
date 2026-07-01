"""Tests for doctor optional dependency handling (Blocker 1)."""

import importlib.util
import sys
from pathlib import Path


DOCTOR_PATH = Path(__file__).resolve().parents[2] / "scripts" / "doctor.py"
SPEC = importlib.util.spec_from_file_location("doctor", DOCTOR_PATH)
doctor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = doctor
SPEC.loader.exec_module(doctor)


def test_doctor_demucs_optional_by_default():
    """Default doctor should not fail on missing Demucs (Blocker 1)."""
    checks = doctor.collect_checks(need_demucs=False)
    demucs_checks = [c for c in checks if "demucs" in c.name]
    assert demucs_checks, "expected a demucs import check"
    for c in demucs_checks:
        assert not c.required, f"demucs check '{c.name}' should be optional by default"


def test_doctor_demucs_required_with_need_demucs():
    """--need-demucs should make Demucs required (Blocker 1)."""
    checks = doctor.collect_checks(need_demucs=True)
    demucs_checks = [c for c in checks if "demucs" in c.name]
    assert demucs_checks
    for c in demucs_checks:
        assert c.required, f"demucs check '{c.name}' should be required with --need-demucs"


def test_doctor_demucs_required_with_need_vocal_separation():
    """--need-vocal-separation should make Demucs required (Blocker 1)."""
    checks = doctor.collect_checks(need_vocal_separation=True)
    demucs_checks = [c for c in checks if "demucs" in c.name]
    assert demucs_checks
    for c in demucs_checks:
        assert c.required


def test_doctor_librosa_always_optional():
    """librosa should always be optional (Blocker 1)."""
    checks = doctor.collect_checks(need_demucs=True, strict=True)
    librosa_checks = [c for c in checks if "librosa" in c.name]
    assert librosa_checks
    for c in librosa_checks:
        assert not c.required, "librosa should always be optional"


def test_doctor_transformers_always_optional():
    """transformers should always be optional (Blocker 1)."""
    checks = doctor.collect_checks(need_demucs=True, strict=True)
    transformers_checks = [c for c in checks if "transformers" in c.name]
    assert transformers_checks
    for c in transformers_checks:
        assert not c.required, "transformers should always be optional"


def test_doctor_qwen3_required_by_default():
    """Default doctor must include required Qwen3-local checks."""
    checks = doctor.collect_checks()
    qwen3_checks = [c for c in checks if c.section == "Qwen3"]
    assert qwen3_checks, "expected Qwen3 checks"
    for c in qwen3_checks:
        assert c.required, f"Qwen3 check '{c.name}' should be required"
