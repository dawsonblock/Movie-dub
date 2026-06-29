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


def test_doctor_checkpoint_report_rejects_pointer(tmp_path):
    """Checkpoint report should flag LFS pointer files (Blocker 9)."""
    ckpt = tmp_path / "converter" / "checkpoint.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:aaa\nsize 131320490\n",
        encoding="utf-8",
    )
    report = doctor._checkpoint_report(tmp_path)
    cv = report["openvoice_checkpoints"]
    assert cv["status"] == "fail"
    assert len(cv["pointer_files"]) > 0


def test_doctor_checkpoint_report_missing_files(tmp_path):
    """Checkpoint report should flag missing files (Blocker 9)."""
    report = doctor._checkpoint_report(tmp_path)
    cv = report["openvoice_checkpoints"]
    assert cv["status"] == "fail"
    assert len(cv["missing"]) > 0


def test_doctor_checkpoint_report_pass(tmp_path):
    """Checkpoint report should pass with valid files (Blocker 9)."""
    converter = tmp_path / "converter"
    converter.mkdir()
    (converter / "config.json").write_text("{}", encoding="utf-8")
    ckpt = converter / "checkpoint.pth"
    ckpt.write_bytes(b"\x00" * (doctor.MIN_CONVERTER_CHECKPOINT_BYTES + 100))
    speaker_dir = tmp_path / "base_speakers" / "ses"
    speaker_dir.mkdir(parents=True)
    (speaker_dir / "en.pth").write_bytes(b"\x00" * 5000)
    report = doctor._checkpoint_report(tmp_path)
    cv = report["openvoice_checkpoints"]
    assert cv["status"] == "pass"
