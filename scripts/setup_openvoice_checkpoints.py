#!/usr/bin/env python3
"""Compatibility wrapper for the shell checkpoint downloader."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    script = ROOT / "scripts" / "download_openvoice_v2_checkpoints.sh"
    result = subprocess.run(["bash", script.as_posix()])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
