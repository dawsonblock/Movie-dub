#!/usr/bin/env python3
"""Readiness checks for the local pyVideoTrans + OpenVoice Mac workflow."""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
OPENVOICE = ROOT / "OpenVoice-main"
DEFAULT_HF_REPO = "rsxdalv/OpenVoiceV2"
MIN_CONVERTER_CHECKPOINT_BYTES = 100_000_000
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"
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


def run(cmd: list[str], cwd: Path | None = None,
        timeout: int = 45) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout
        )
    except Exception as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output.splitlines()[-1] if output else "ok"


def check(section: str, name: str, ok: bool, detail: str,
          required: bool = True) -> Check:
    return Check(section, name, ok, detail, required)


def file_check(section: str, name: str, path: Path,
               required: bool = True) -> Check:
    return check(section, name, path.is_file(), path.as_posix(), required)


def dir_check(section: str, name: str, path: Path,
              required: bool = True) -> Check:
    return check(section, name, path.is_dir(), path.as_posix(), required)


def import_check(section: str, python: Path, module: str,
                 cwd: Path | None = None, required: bool = True) -> Check:
    if not python.is_file():
        return check(section, f"import {module}", False,
                     f"missing python: {python}", required=required)
    ok, detail = run(
        [python.as_posix(), "-c", f"import {module}; print('ok')"], cwd=cwd
    )
    return check(section, f"import {module}", ok, detail, required=required)


def python_version_check(section: str, python: Path, name: str) -> Check:
    if not python.is_file():
        return check(section, name, False, f"missing python: {python}")
    ok, detail = run(
        [
            python.as_posix(),
            "-c",
            "import sys; print(f'{sys.version_info.major}."
            "{sys.version_info.minor}.{sys.version_info.micro}')",
        ]
    )
    if not ok:
        return check(section, name, False, detail)
    major_minor = ".".join(detail.split(".")[:2])
    return check(section, name, major_minor == "3.10", detail)


def binary_check(section: str, name: str, binary: str,
                 local_path: Path | None = None) -> Check:
    found = shutil.which(binary)
    if not found and local_path and local_path.is_file():
        found = local_path.as_posix()
    if not found:
        suffix = f" or {local_path}" if local_path else ""
        return check(section, name, False, f"not found on PATH{suffix}")
    ver_flag = "-version" if name in {"ffmpeg", "ffprobe"} else "--version"
    ok, detail = run([found, ver_flag])
    return check(section, name, ok, f"{found} ({detail})")


def command_check(section: str, name: str, cmd: list[str], required: bool = True) -> Check:
    found = shutil.which(cmd[0])
    if not found:
        return check(section, name, False, f"{cmd[0]} not found on PATH", required)
    ok, detail = run(cmd)
    return check(section, name, ok, detail, required)


def git_xet_check(required: bool = False) -> Check:
    if shutil.which("git-xet"):
        return check("Checkpoints", "git-xet", True, shutil.which("git-xet") or "git-xet", required)
    ok, detail = run(["git", "xet", "--help"], timeout=15)
    return check("Checkpoints", "git-xet", ok, detail if ok else "git xet not installed", required)


def uvx_check() -> Check:
    found = shutil.which("uvx")
    if not found:
        return check("Checkpoints", "uvx", False, "uvx not found on PATH", required=False)
    return check("Checkpoints", "uvx", True, found, required=False)


def hf_cli_check(required: bool) -> Check:
    return command_check("Checkpoints", "hf CLI", ["hf", "--help"], required=required)


def download_tools_status() -> dict:
    """Report which HuggingFace download tools are available."""
    uvx_ok = bool(shutil.which("uvx"))
    hf_ok = bool(shutil.which("hf"))
    git_xet_ok = bool(shutil.which("git-xet")) or run(["git", "xet", "--help"], timeout=15)[0]
    return {
        "uvx": uvx_ok,
        "hf": hf_ok,
        "git_xet": git_xet_ok,
        "can_download_hf": uvx_ok or hf_ok,
    }


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


def converter_checkpoint_real_file_check(checkpoint_path: Path) -> Check:
    if not checkpoint_path.is_file():
        return check("Models", "converter checkpoint real file", False, f"missing: {checkpoint_path}")
    try:
        size = checkpoint_path.stat().st_size
        prefix = checkpoint_path.read_bytes()[:200]
    except OSError as exc:
        return check("Models", "converter checkpoint real file", False, str(exc))
    if prefix.startswith(LFS_POINTER_PREFIX) or LFS_POINTER_PREFIX in prefix:
        return check(
            "Models",
            "converter checkpoint real file",
            False,
            f"Git LFS/Xet pointer stub, not model data: {checkpoint_path}",
        )
    if size < MIN_CONVERTER_CHECKPOINT_BYTES:
        return check(
            "Models",
            "converter checkpoint real file",
            False,
            f"suspiciously small: {size} bytes; expected about 131 MB",
        )
    return check("Models", "converter checkpoint real file", True, f"{size} bytes")


def checkpoint_dir() -> Path:
    raw = os.environ.get("OPENVOICE_CHECKPOINT_DIR")
    return Path(raw).expanduser().resolve() if raw else OPENVOICE / "checkpoints_v2"


def checkpoints_ready(checkpoint_path: Path) -> bool:
    speaker_dir = checkpoint_path / "base_speakers" / "ses"
    return (
        (checkpoint_path / "converter" / "config.json").is_file()
        and (checkpoint_path / "converter" / "checkpoint.pth").is_file()
        and converter_checkpoint_real_file_check(checkpoint_path / "converter" / "checkpoint.pth").ok
        and speaker_dir.is_dir()
        and any(speaker_dir.glob("*.pth"))
    )


def collect_checks(
    need_openvoice: bool = True,
    need_demucs: bool = False,
    need_vocal_separation: bool = False,
    strict: bool = False,
) -> list[Check]:
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"
    ov_python = OPENVOICE / ".venv" / "bin" / "python"
    ov_checkpoint_dir = checkpoint_dir()
    checkpoint_tools_required = not checkpoints_ready(ov_checkpoint_dir)

    # --- Required group (default install must pass) ---
    checks: list[Check] = [
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
    ]

    # --- OpenVoice group (required by default, can be skipped with --no-openvoice) ---
    if need_openvoice:
        checks.extend([
            dir_check("OpenVoice", "repo", OPENVOICE),
            dir_check("OpenVoice", "venv", OPENVOICE / ".venv"),
            file_check("OpenVoice", "python", ov_python),
            python_version_check("OpenVoice", ov_python, "Python 3.10"),
            import_check("OpenVoice", ov_python, "torch", cwd=OPENVOICE),
            import_check("OpenVoice", ov_python, "openvoice", cwd=OPENVOICE),
            import_check("OpenVoice", ov_python, "melo", cwd=OPENVOICE),
            import_check("OpenVoice", ov_python, "soundfile", cwd=OPENVOICE),
            import_check("OpenVoice", ov_python, "unidic", cwd=OPENVOICE),
        ])

    # --- Checkpoint download tools (required only if checkpoints missing) ---
    checks.extend([
        uvx_check(),
        hf_cli_check(required=checkpoint_tools_required and not shutil.which("uvx")),
        git_xet_check(required=False),
        check(
            "Checkpoints",
            "Hugging Face downloader",
            bool(shutil.which("uvx") or shutil.which("hf")),
            "uvx available" if shutil.which("uvx") else ("hf available" if shutil.which("hf") else "none"),
            required=checkpoint_tools_required,
        ),
        check(
            "Checkpoints",
            "Hugging Face repo",
            bool(os.environ.get("OPENVOICE_HF_REPO", DEFAULT_HF_REPO).strip()),
            os.environ.get("OPENVOICE_HF_REPO", DEFAULT_HF_REPO),
            required=checkpoint_tools_required,
        ),
    ])

    # --- OpenVoice checkpoints (required by default) ---
    if need_openvoice:
        checks.extend([
            file_check("Models", "converter config", ov_checkpoint_dir / "converter" / "config.json"),
            file_check("Models", "converter checkpoint", ov_checkpoint_dir / "converter" / "checkpoint.pth"),
            converter_checkpoint_real_file_check(ov_checkpoint_dir / "converter" / "checkpoint.pth"),
            base_speaker_check(ov_checkpoint_dir),
        ])

    # --- Bridge + assets ---
    checks.extend([
        file_check("Bridge", "bridge script", ROOT / "bridge" / "openvoice_segment_tts.py"),
        bridge_compile_check(),
        file_check("Bridge", "OpenVoice smoke script", ROOT / "scripts" / "smoke_openvoice_bridge.py"),
        file_check("Assets", "test MP4", ROOT / "assets" / "pyvideotrans_test_clip.mp4"),
        file_check("Assets", "default reference WAV", ROOT / "voices" / "openvoice_default_reference.wav"),
        # Optional: male/female reference voices for gender-based dubbing
        file_check("Assets", "male reference WAV", ROOT / "voices" / "male_reference.wav", required=False),
        file_check("Assets", "female reference WAV", ROOT / "voices" / "female_reference.wav", required=False),
    ])

    # --- Optional audio-quality dependencies (Blocker 1) ---
    # Demucs is only required when --need-demucs or --need-vocal-separation is passed.
    # librosa is used for gender detection (optional).
    # transformers is used for whisper-large-v3-turbo (optional).
    demucs_required = need_demucs or need_vocal_separation or strict
    checks.append(
        import_check("Audio", py_python, "demucs", cwd=PYVIDEOTRANS,
                      required=demucs_required)
    )
    checks.append(
        import_check("Audio", py_python, "librosa", cwd=PYVIDEOTRANS, required=False)
    )
    checks.append(
        import_check("Audio", py_python, "transformers", cwd=PYVIDEOTRANS, required=False)
    )

    return checks


def summarize(checks: list[Check], strict: bool = False) -> dict:
    required_failures = [item for item in checks if item.required and not item.ok]
    optional_warnings = [item for item in checks if not item.required and not item.ok]
    ready = not required_failures and (not strict or not optional_warnings)
    return {
        "ready": ready,
        "required_failures": len(required_failures),
        "optional_warnings": len(optional_warnings),
        "download_tools": download_tools_status(),
        "checks": [asdict(item) for item in checks],
    }


def checkpoint_failures(checks: list[Check]) -> bool:
    return any(item.section == "Models" and not item.ok for item in checks)


def print_checkpoint_instructions() -> None:
    target = checkpoint_dir()
    print()
    print("Checkpoints missing. Run:")
    print("  make download-openvoice")
    print("Expected layout:")
    print(f"  {target / 'converter' / 'config.json'}")
    print(f"  {target / 'converter' / 'checkpoint.pth'}")
    print(f"  {target / 'base_speakers' / 'ses' / '*.pth'}")
    print("Install download tooling if needed:")
    print("  brew install uv")
    print("  uvx hf download rsxdalv/OpenVoiceV2 --local-dir OpenVoice-main --include 'checkpoints_v2/*'")
    print("Or install the Hugging Face CLI from huggingface_hub:")
    print("  python3 -m pip install -U huggingface_hub")
    print()
    print("WARNING: only use trusted OpenVoice checkpoints. torch.load can execute")
    print("unsafe pickle data, so never load checkpoints from untrusted sources.")


def print_table(checks: list[Check]) -> None:
    name_width = max(len(f"{item.section}: {item.name}") for item in checks)
    for item in checks:
        status = "PASS" if item.ok else ("WARN" if not item.required else "FAIL")
        label = f"{item.section}: {item.name}"
        print(f"[{status}] {label:<{name_width}}  {item.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local pyVideoTrans + OpenVoice readiness")
    parser.add_argument("--strict", action="store_true",
                        help="return nonzero for optional warnings too")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable readiness JSON")
    parser.add_argument("--need-openvoice", action="store_true", default=True,
                        help="require OpenVoice venv + checkpoints (default: on)")
    parser.add_argument("--no-openvoice", dest="need_openvoice",
                        action="store_false",
                        help="skip OpenVoice checks (pyVideoTrans-only setup)")
    parser.add_argument("--need-demucs", action="store_true",
                        help="require Demucs for AI-based vocal separation")
    parser.add_argument("--need-vocal-separation", action="store_true",
                        help="require Demucs + vocal-separation dependencies")
    parser.add_argument("--checkpoints-only", action="store_true",
                        help="only check OpenVoice checkpoints (Blocker 9)")
    args = parser.parse_args()

    if args.checkpoints_only:
        return _checkpoints_only_main()

    checks = collect_checks(
        need_openvoice=args.need_openvoice,
        need_demucs=args.need_demucs,
        need_vocal_separation=args.need_vocal_separation,
        strict=args.strict,
    )
    summary = summarize(checks, strict=args.strict)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["ready"] else 1

    print_table(checks)
    if checkpoint_failures(checks):
        print_checkpoint_instructions()
    print()
    print(f"READY: {'yes' if summary['ready'] else 'no'}")
    print(f"Required failures: {summary['required_failures']}")
    print(f"Optional warnings: {summary['optional_warnings']}")
    return 0 if summary["ready"] else 1


def _checkpoint_report(checkpoint_path: Path) -> dict:
    """Build a detailed checkpoint report (Blocker 9)."""
    converter_dir = checkpoint_path / "converter"
    speaker_dir = checkpoint_path / "base_speakers" / "ses"
    checked: list[str] = []
    missing: list[str] = []
    too_small: list[str] = []
    pointer_files: list[str] = []

    config_path = converter_dir / "config.json"
    ckpt_path = converter_dir / "checkpoint.pth"
    checked.append(config_path.as_posix())
    if not config_path.is_file():
        missing.append(config_path.as_posix())
    checked.append(ckpt_path.as_posix())
    if not ckpt_path.is_file():
        missing.append(ckpt_path.as_posix())
    else:
        size = ckpt_path.stat().st_size
        prefix = ckpt_path.read_bytes()[:200]
        if prefix.startswith(LFS_POINTER_PREFIX) or LFS_POINTER_PREFIX in prefix:
            pointer_files.append(f"{ckpt_path.as_posix()} (LFS pointer stub)")
        elif size < MIN_CONVERTER_CHECKPOINT_BYTES:
            too_small.append(f"{ckpt_path.as_posix()} ({size} bytes)")

    speakers = sorted(speaker_dir.glob("*.pth")) if speaker_dir.is_dir() else []
    for sp in speakers:
        checked.append(sp.as_posix())
        size = sp.stat().st_size
        prefix = sp.read_bytes()[:200]
        if prefix.startswith(LFS_POINTER_PREFIX) or LFS_POINTER_PREFIX in prefix:
            pointer_files.append(f"{sp.as_posix()} (LFS pointer stub)")
        elif size < 1000:
            too_small.append(f"{sp.as_posix()} ({size} bytes)")
    if not speakers:
        missing.append(speaker_dir.as_posix())

    status = "pass" if not (missing or too_small or pointer_files) else "fail"
    return {
        "openvoice_checkpoints": {
            "status": status,
            "missing": missing,
            "too_small": too_small,
            "pointer_files": pointer_files,
            "checked_files": checked,
        }
    }


def _checkpoints_only_main() -> int:
    """Run only OpenVoice checkpoint validation (Blocker 9)."""
    ckpt_dir = checkpoint_dir()
    report = _checkpoint_report(ckpt_dir)
    cv = report["openvoice_checkpoints"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if cv["status"] == "pass":
        print("PASS: all OpenVoice checkpoints are present and non-pointer files")
        return 0
    print("FAIL:")
    for item in cv["missing"]:
        print(f"  - missing: {item}")
    for item in cv["pointer_files"]:
        print(f"  - pointer file: {item}")
    for item in cv["too_small"]:
        print(f"  - too small: {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
