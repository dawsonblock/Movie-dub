#!/usr/bin/env python3
"""Write a machine-readable proof_report.json after running proof checks.

This script collects the results of all proof checks (doctor, smoke tests,
pyVideoTrans tests, etc.) and writes a structured JSON report to
``reports/proof_report.json``. It is designed to be called by the Makefile
``proof`` target.

Usage:
    python scripts/write_proof_report.py --result pass
    python scripts/write_proof_report.py --result fail --failed-check qwen3_model
    python scripts/write_proof_report.py --result fail --failed-check doctor --message "..."
    python scripts/write_proof_report.py --checks-json /tmp/checks.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
PROOF_REPORT_PATH = REPORTS_DIR / "proof_report.json"
SCHEMA_VERSION = "1.0"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def build_proof_report(
    checks: dict[str, str],
    result: str,
    artifacts: dict[str, str] | None = None,
    failed_check: str = "",
    message: str = "",
) -> dict[str, Any]:
    """Build the proof report dictionary."""
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": _python_version(),
        },
        "checks": checks,
        "artifacts": artifacts or {},
        "result": result,
    }
    if failed_check:
        report["failed_check"] = failed_check
    if message:
        report["message"] = message
    return report


def write_proof_report(report: dict[str, Any], path: Path = PROOF_REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a machine-readable proof_report.json"
    )
    parser.add_argument("--result", choices=["pass", "fail"], default="pass")
    parser.add_argument("--failed-check", default="",
                        help="name of the check that failed")
    parser.add_argument("--message", default="",
                        help="failure message")
    parser.add_argument("--checks-json", default="",
                        help="path to a JSON file with check results")
    parser.add_argument("--artifacts-json", default="",
                        help="path to a JSON file with artifact paths")
    parser.add_argument("--output", default=str(PROOF_REPORT_PATH),
                        help="output path for the proof report")
    args = parser.parse_args()

    checks: dict[str, str] = {}
    if args.checks_json:
        try:
            checks = json.loads(Path(args.checks_json).read_text(encoding="utf-8"))
        except Exception:
            pass

    artifacts: dict[str, str] = {}
    if args.artifacts_json:
        try:
            artifacts = json.loads(Path(args.artifacts_json).read_text(encoding="utf-8"))
        except Exception:
            pass

    report = build_proof_report(
        checks=checks,
        result=args.result,
        artifacts=artifacts,
        failed_check=args.failed_check,
        message=args.message,
    )
    write_proof_report(report, Path(args.output))
    print(f"Proof report written: {args.output}")
    print(f"Result: {args.result}")
    return 0 if args.result == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
