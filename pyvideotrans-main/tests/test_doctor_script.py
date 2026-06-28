import importlib.util
import sys
from pathlib import Path


DOCTOR_PATH = Path(__file__).resolve().parents[2] / "scripts" / "doctor.py"
SPEC = importlib.util.spec_from_file_location("doctor", DOCTOR_PATH)
doctor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = doctor
SPEC.loader.exec_module(doctor)


def test_doctor_summary_json_shape():
    checks = [
        doctor.Check("System", "python", True, "ok"),
        doctor.Check("Checkpoints", "hf CLI", False, "missing", required=False),
    ]

    summary = doctor.summarize(checks)

    assert summary["ready"] is True
    assert summary["required_failures"] == 0
    assert summary["optional_warnings"] == 1
    assert summary["checks"][0]["section"] == "System"


def test_doctor_summary_strict_fails_on_optional_warnings():
    checks = [doctor.Check("Checkpoints", "hf CLI", False, "missing", required=False)]

    summary = doctor.summarize(checks, strict=True)

    assert summary["ready"] is False
    assert summary["required_failures"] == 0
    assert summary["optional_warnings"] == 1


def test_doctor_rejects_lfs_pointer_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "size 131320490\n",
        encoding="utf-8",
    )

    result = doctor.converter_checkpoint_real_file_check(checkpoint)

    assert result.ok is False
    assert "pointer" in result.detail


def test_doctor_rejects_suspiciously_small_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"PK\x03\x04small")

    result = doctor.converter_checkpoint_real_file_check(checkpoint)

    assert result.ok is False
    assert "suspiciously small" in result.detail
