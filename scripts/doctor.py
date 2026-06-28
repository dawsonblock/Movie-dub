#!/usr/bin/env python3
"""Local readiness checks for the pyVideoTrans + OpenVoice workspace."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
OPENVOICE = ROOT / "OpenVoice-main"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> tuple[bool, str]:
    try:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output.splitlines()[-1] if output else "ok"


def import_check(python: Path, module: str, cwd: Path | None = None) -> Check:
    if not python.is_file():
        return Check(f"import {module}", False, f"missing python: {python}")
    ok, detail = run([python.as_posix(), "-c", f"import {module}; print('ok')"], cwd=cwd)
    return Check(f"import {module}", ok, detail)


def path_check(name: str, path: Path, required: bool = True) -> Check:
    return Check(name, path.exists(), path.as_posix(), required=required)


def file_check(name: str, path: Path, required: bool = True) -> Check:
    return Check(name, path.is_file(), path.as_posix(), required=required)


def dir_check(name: str, path: Path, required: bool = True) -> Check:
    return Check(name, path.is_dir(), path.as_posix(), required=required)


def python_version_check(python: Path, name: str) -> Check:
    if not python.is_file():
        return Check(name, False, f"missing python: {python}")
    ok, detail = run(
        [
            python.as_posix(),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
        ]
    )
    if not ok:
        return Check(name, False, detail)
    major_minor = ".".join(detail.split(".")[:2])
    return Check(name, major_minor == "3.10", detail)


def ffmpeg_check() -> Check:
    local_ffmpeg = PYVIDEOTRANS / "ffmpeg" / "ffmpeg"
    ffmpeg = shutil.which("ffmpeg") or (local_ffmpeg.as_posix() if local_ffmpeg.is_file() else None)
    if not ffmpeg:
        return Check("ffmpeg", False, "not found on PATH or pyvideotrans-main/ffmpeg")
    ok, detail = run([ffmpeg, "-version"])
    version = detail if ok else "failed to run"
    return Check("ffmpeg", ok, f"{ffmpeg} ({version})")


def ffprobe_check() -> Check:
    local_ffprobe = PYVIDEOTRANS / "ffmpeg" / "ffprobe"
    ffprobe = shutil.which("ffprobe") or (local_ffprobe.as_posix() if local_ffprobe.is_file() else None)
    if not ffprobe:
        return Check("ffprobe", False, "not found on PATH or pyvideotrans-main/ffmpeg")
    ok, detail = run([ffprobe, "-version"])
    version = detail if ok else "failed to run"
    return Check("ffprobe", ok, f"{ffprobe} ({version})")


def base_speaker_check(checkpoint_dir: Path) -> Check:
    speaker_dir = checkpoint_dir / "base_speakers" / "ses"
    speakers = sorted(path.name for path in speaker_dir.glob("*.pth")) if speaker_dir.is_dir() else []
    return Check("OpenVoice base speaker embeddings", bool(speakers), ", ".join(speakers) or speaker_dir.as_posix())


def collect_checks() -> list[Check]:
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"
    ov_python = OPENVOICE / ".venv" / "bin" / "python"
    checkpoint_dir = OPENVOICE / "checkpoints_v2"

    checks = [
        path_check("workspace root", ROOT),
        ffmpeg_check(),
        ffprobe_check(),
        dir_check("pyVideoTrans venv", PYVIDEOTRANS / ".venv"),
        python_version_check(py_python, "pyVideoTrans Python"),
        import_check(py_python, "PySide6", cwd=PYVIDEOTRANS),
        dir_check("OpenVoice venv", OPENVOICE / ".venv"),
        python_version_check(ov_python, "OpenVoice Python"),
        import_check(ov_python, "openvoice", cwd=OPENVOICE),
        import_check(ov_python, "melo", cwd=OPENVOICE),
        import_check(ov_python, "soundfile", cwd=OPENVOICE),
        import_check(ov_python, "torch", cwd=OPENVOICE),
        file_check("OpenVoice converter config", checkpoint_dir / "converter" / "config.json"),
        file_check("OpenVoice converter checkpoint", checkpoint_dir / "converter" / "checkpoint.pth"),
        base_speaker_check(checkpoint_dir),
        file_check("OpenVoice bridge", ROOT / "bridge" / "openvoice_segment_tts.py"),
        file_check("default reference WAV", ROOT / "voices" / "openvoice_default_reference.wav"),
        file_check("smoke test clip", ROOT / "assets" / "pyvideotrans_test_clip.mp4"),
    ]
    return checks


def print_table(checks: list[Check]) -> None:
    width = max(len(check.name) for check in checks)
    for check in checks:
        status = "PASS" if check.ok else ("WARN" if not check.required else "FAIL")
        print(f"{status:4}  {check.name:<{width}}  {check.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local pyVideoTrans + OpenVoice readiness")
    parser.add_argument("--strict", action="store_true", help="return nonzero for optional warnings too")
    args = parser.parse_args()
    checks = collect_checks()
    print_table(checks)
    failures = [check for check in checks if not check.ok and (check.required or args.strict)]
    if failures:
        print(f"\n{len(failures)} required check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
