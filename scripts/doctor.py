#!/usr/bin/env python3
"""Readiness checks for the local pyVideoTrans + OpenVoice Mac workflow."""

from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
OPENVOICE = ROOT / "OpenVoice-main"
PYTHON310_CANDIDATES = [
    Path(os.environ["PYTHON_BIN"]) if "PYTHON_BIN" in os.environ else None,
    Path(shutil.which("python3.10")) if shutil.which("python3.10") else None,
    Path("/opt/homebrew/opt/python@3.10/bin/python3.10"),
    Path("/usr/local/opt/python@3.10/bin/python3.10"),
]


@dataclass
class Check:
    section: str
    name: str
    ok: bool
    detail: str
    required: bool = True


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 45) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output.splitlines()[-1] if output else "ok"


def check(section: str, name: str, ok: bool, detail: str, required: bool = True) -> Check:
    return Check(section, name, ok, detail, required)


def file_check(section: str, name: str, path: Path, required: bool = True) -> Check:
    return check(section, name, path.is_file(), path.as_posix(), required)


def dir_check(section: str, name: str, path: Path, required: bool = True) -> Check:
    return check(section, name, path.is_dir(), path.as_posix(), required)


def import_check(section: str, python: Path, module: str, cwd: Path | None = None) -> Check:
    if not python.is_file():
        return check(section, f"import {module}", False, f"missing python: {python}")
    ok, detail = run([python.as_posix(), "-c", f"import {module}; print('ok')"], cwd=cwd)
    return check(section, f"import {module}", ok, detail)


def python_version_check(section: str, python: Path, name: str) -> Check:
    if not python.is_file():
        return check(section, name, False, f"missing python: {python}")
    ok, detail = run(
        [
            python.as_posix(),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
        ]
    )
    if not ok:
        return check(section, name, False, detail)
    major_minor = ".".join(detail.split(".")[:2])
    return check(section, name, major_minor == "3.10", detail)


def binary_check(section: str, name: str, binary: str, local_path: Path | None = None) -> Check:
    found = shutil.which(binary)
    if not found and local_path and local_path.is_file():
        found = local_path.as_posix()
    if not found:
        suffix = f" or {local_path}" if local_path else ""
        return check(section, name, False, f"not found on PATH{suffix}")
    ok, detail = run([found, "-version"] if name in {"ffmpeg", "ffprobe"} else [found, "--version"])
    return check(section, name, ok, f"{found} ({detail})")


def python310_binary_check() -> Check:
    for candidate in PYTHON310_CANDIDATES:
        if candidate and candidate.is_file():
            ok, detail = run([candidate.as_posix(), "--version"])
            return check("System", "python3.10", ok, f"{candidate} ({detail})")
    return check("System", "python3.10", False, "not found via PYTHON_BIN, PATH, or Homebrew fallback")


def bridge_compile_check() -> Check:
    bridge = ROOT / "bridge" / "openvoice_segment_tts.py"
    if not bridge.is_file():
        return check("Bridge", "bridge compiles", False, bridge.as_posix())
    try:
        py_compile.compile(bridge.as_posix(), doraise=True)
    except Exception as exc:
        return check("Bridge", "bridge compiles", False, str(exc))
    return check("Bridge", "bridge compiles", True, bridge.as_posix())


def base_speaker_check(checkpoint_dir: Path) -> Check:
    speaker_dir = checkpoint_dir / "base_speakers" / "ses"
    speakers = sorted(path.name for path in speaker_dir.glob("*.pth")) if speaker_dir.is_dir() else []
    return check(
        "Models",
        "OpenVoice base speaker embeddings",
        bool(speakers),
        ", ".join(speakers) or speaker_dir.as_posix(),
    )


def collect_checks() -> list[Check]:
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"
    ov_python = OPENVOICE / ".venv" / "bin" / "python"
    checkpoint_dir = OPENVOICE / "checkpoints_v2"

    return [
        check("System", "macOS", sys.platform == "darwin", sys.platform),
        python310_binary_check(),
        binary_check("System", "ffmpeg", "ffmpeg", PYVIDEOTRANS / "ffmpeg" / "ffmpeg"),
        binary_check("System", "ffprobe", "ffprobe", PYVIDEOTRANS / "ffmpeg" / "ffprobe"),
        dir_check("pyVideoTrans", "repo", PYVIDEOTRANS),
        dir_check("pyVideoTrans", "venv", PYVIDEOTRANS / ".venv"),
        file_check("pyVideoTrans", "python", py_python),
        python_version_check("pyVideoTrans", py_python, "Python 3.10"),
        import_check("pyVideoTrans", py_python, "PySide6", cwd=PYVIDEOTRANS),
        import_check("pyVideoTrans", py_python, "videotrans", cwd=PYVIDEOTRANS),
        file_check("pyVideoTrans", "cli.py", PYVIDEOTRANS / "cli.py"),
        dir_check("OpenVoice", "repo", OPENVOICE),
        dir_check("OpenVoice", "venv", OPENVOICE / ".venv"),
        file_check("OpenVoice", "python", ov_python),
        python_version_check("OpenVoice", ov_python, "Python 3.10"),
        import_check("OpenVoice", ov_python, "torch", cwd=OPENVOICE),
        import_check("OpenVoice", ov_python, "openvoice", cwd=OPENVOICE),
        import_check("OpenVoice", ov_python, "melo", cwd=OPENVOICE),
        import_check("OpenVoice", ov_python, "soundfile", cwd=OPENVOICE),
        import_check("OpenVoice", ov_python, "unidic", cwd=OPENVOICE),
        file_check("Models", "converter config", checkpoint_dir / "converter" / "config.json"),
        file_check("Models", "converter checkpoint", checkpoint_dir / "converter" / "checkpoint.pth"),
        base_speaker_check(checkpoint_dir),
        file_check("Bridge", "bridge script", ROOT / "bridge" / "openvoice_segment_tts.py"),
        bridge_compile_check(),
        file_check("Bridge", "OpenVoice smoke script", ROOT / "scripts" / "smoke_openvoice_bridge.py"),
        file_check("Assets", "test MP4", ROOT / "assets" / "pyvideotrans_test_clip.mp4"),
        file_check("Assets", "default reference WAV", ROOT / "voices" / "openvoice_default_reference.wav"),
    ]


def print_table(checks: list[Check]) -> None:
    name_width = max(len(f"{item.section}: {item.name}") for item in checks)
    for item in checks:
        status = "PASS" if item.ok else ("WARN" if not item.required else "FAIL")
        label = f"{item.section}: {item.name}"
        print(f"[{status}] {label:<{name_width}}  {item.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local pyVideoTrans + OpenVoice readiness")
    parser.add_argument("--strict", action="store_true", help="return nonzero for optional warnings too")
    args = parser.parse_args()
    checks = collect_checks()
    print_table(checks)
    required_failures = [item for item in checks if item.required and not item.ok]
    optional_warnings = [item for item in checks if not item.required and not item.ok]
    ready = not required_failures and (not args.strict or not optional_warnings)
    print()
    print(f"READY: {'yes' if ready else 'no'}")
    print(f"Required failures: {len(required_failures)}")
    print(f"Optional warnings: {len(optional_warnings)}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
