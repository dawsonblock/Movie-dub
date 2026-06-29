"""Tests for machine-readable proof report (Blocker 7)."""

import importlib.util
import json
import sys
from pathlib import Path


PROOF_PATH = Path(__file__).resolve().parents[2] / "scripts" / "write_proof_report.py"
SPEC = importlib.util.spec_from_file_location("write_proof_report", PROOF_PATH)
write_proof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = write_proof
SPEC.loader.exec_module(write_proof)


def test_proof_report_pass_structure(tmp_path):
    """Proof report on pass should have all required fields (Blocker 7)."""
    checks = {
        "doctor": "pass",
        "openvoice_smoke": "pass",
        "audio_builder_smoke": "pass",
        "remux_smoke": "pass",
        "pyvideotrans_tests": "pass",
    }
    report = write_proof.build_proof_report(
        checks=checks,
        result="pass",
        artifacts={"test_video": "/test.mp4"},
    )
    assert report["schema_version"] == "1.0"
    assert report["result"] == "pass"
    assert "timestamp" in report
    assert "platform" in report
    assert report["platform"]["system"]
    assert report["checks"] == checks
    assert report["artifacts"]["test_video"] == "/test.mp4"
    assert "failed_check" not in report


def test_proof_report_fail_includes_failed_check(tmp_path):
    """Proof report on fail should include failed_check and message (Blocker 7)."""
    report = write_proof.build_proof_report(
        checks={"doctor": "pass", "openvoice_smoke": "fail"},
        result="fail",
        failed_check="openvoice_smoke",
        message="smoke test failed",
    )
    assert report["result"] == "fail"
    assert report["failed_check"] == "openvoice_smoke"
    assert report["message"] == "smoke test failed"


def test_proof_report_written_to_file(tmp_path):
    """write_proof_report should write JSON to the specified path (Blocker 7)."""
    output = tmp_path / "proof_report.json"
    report = write_proof.build_proof_report(
        checks={"doctor": "pass"},
        result="pass",
    )
    write_proof.write_proof_report(report, output)
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["result"] == "pass"
    assert data["checks"]["doctor"] == "pass"


def test_proof_report_written_even_on_failure(tmp_path):
    """Proof report should be written even when result is fail (Blocker 7)."""
    output = tmp_path / "proof_report.json"
    report = write_proof.build_proof_report(
        checks={"doctor": "fail"},
        result="fail",
        failed_check="doctor",
        message="doctor failed",
    )
    write_proof.write_proof_report(report, output)
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["result"] == "fail"
    assert data["failed_check"] == "doctor"
