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


def test_doctor_qwen3_required_checks_present():
    """Default doctor must include required Qwen3-local checks."""
    checks = doctor.collect_checks()
    qwen3_checks = [c for c in checks if c.section == "Qwen3"]
    assert qwen3_checks, "expected Qwen3 checks"
    for c in qwen3_checks:
        assert c.required, f"Qwen3 check '{c.name}' should be required"
