#!/usr/bin/env python3
"""Readiness checks for the local pyVideoTrans + Qwen3-local Mac workflow."""

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


def python310_binary_check() -> Check:
    for candidate in PYTHON310_CANDIDATES:
        if candidate and candidate.is_file():
            ok, detail = run([candidate.as_posix(), "--version"])
            return check("System", "python3.10", ok, f"{candidate} ({detail})")
    return check("System", "python3.10", False, "not found via PYTHON_BIN, PATH, or Homebrew fallback")


def bridge_compile_check() -> Check:
    bridge = ROOT / "bridge" / "qwen3_segment_tts.py"
    if not bridge.is_file():
        return check("Bridge", "bridge compiles", False, bridge.as_posix())
    try:
        py_compile.compile(bridge.as_posix(), doraise=True)
    except Exception as exc:
        return check("Bridge", "bridge compiles", False, str(exc))
    return check("Bridge", "bridge compiles", True, bridge.as_posix())


def reference_wav_check() -> Check:
    voices_dir = ROOT / "voices"
    candidates = [
        voices_dir / "reference.wav",
        voices_dir / "male_reference.wav",
    ]
    for path in candidates:
        if path.is_file():
            return check("Assets", "reference WAV", True, path.as_posix())
    if voices_dir.is_dir():
        wavs = sorted(voices_dir.glob("*.wav"))
        if wavs:
            return check("Assets", "reference WAV", True, wavs[0].as_posix())
    return check("Assets", "reference WAV", False, voices_dir.as_posix())


# Qwen3-local runs on the wrapper python (sys.executable), NOT the
# pyVideoTrans venv. doctor.py itself is run with python3, so
# sys.executable here is the same interpreter the Makefile uses for
# run_personal_dub.py and the Qwen3 bridge.
QWEN3_MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base-bf16"
QWEN3_REQUIRED_IMPORTS = ("mlx_audio", "soundfile", "librosa")


def _qwen3_imports_ok() -> tuple[bool, list[str]]:
    missing: list[str] = []
    for mod in QWEN3_REQUIRED_IMPORTS:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    return (not missing, missing)


def _qwen3_model_present() -> bool:
    return (QWEN3_MODEL_DIR / "model.safetensors").is_file()


def _qwen3_ready() -> bool:
    imports_ok, _ = _qwen3_imports_ok()
    return imports_ok and _qwen3_model_present()


def _qwen3_detail() -> str:
    if _qwen3_ready():
        return "ready"
    imports_ok, missing = _qwen3_imports_ok()
    if not imports_ok:
        return (
            f"missing imports on {sys.executable}: "
            f"{', '.join(missing)} (run: make setup-qwen3)"
        )
    if not _qwen3_model_present():
        return (
            f"model not found at {QWEN3_MODEL_DIR} "
            f"(run: make setup-qwen3)"
        )
    return "not configured (run: make setup-qwen3)"


def collect_checks(
    need_demucs: bool = False,
    need_vocal_separation: bool = False,
    strict: bool = False,
) -> list[Check]:
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"

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
        import_check("pyVideoTrans", py_python, "videotrans", cwd=PYVIDEOTRANS),
    ]

    # --- Bridge + assets ---
    checks.extend([
        file_check("Bridge", "bridge script", ROOT / "bridge" / "qwen3_segment_tts.py"),
        bridge_compile_check(),
        file_check("Assets", "test MP4", ROOT / "assets" / "pyvideotrans_test_clip.mp4"),
        reference_wav_check(),
    ])

    # --- Qwen3-local (required TTS engine) ---
    qwen3_imports_ok, qwen3_missing = _qwen3_imports_ok()
    checks.extend([
        check(
            "Qwen3",
            "imports",
            qwen3_imports_ok,
            "ok" if qwen3_imports_ok else f"missing: {', '.join(qwen3_missing)}",
        ),
        check(
            "Qwen3",
            "model present",
            _qwen3_model_present(),
            QWEN3_MODEL_DIR.as_posix(),
        ),
    ])

    # --- Optional audio-quality dependencies ---
    # Demucs is only required when --need-demucs or --need-vocal-separation is passed.
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

    # --- Speaker profiling dependencies ---
    # pyannote.audio + torchaudio are required for --speaker-profiling with
    # --speaker-diarization pyannote. librosa is shared with gender detection.
    # HF_TOKEN is required for pyannote model access (warn, not fail).
    checks.append(
        import_check("Speaker", py_python, "pyannote.audio", cwd=PYVIDEOTRANS,
                      required=False)
    )
    checks.append(
        import_check("Speaker", py_python, "torchaudio", cwd=PYVIDEOTRANS,
                      required=False)
    )
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    checks.append(
        check("Speaker", "HF_TOKEN env", bool(hf_token),
              "set" if hf_token else "not set (required for pyannote diarization)",
              required=False)
    )

    # --- TTS Engines (informational) ---
    # Qwen3-local is the required default engine and is already checked above.
    # OmniVoice is server-based; the bridge script is bundled, but the user
    # must provide a running OmniVoice/Gradio endpoint.
    checks.append(
        check("TTS Engines", "qwen3-local (provider 1)",
              _qwen3_ready(),
              _qwen3_detail(),
              required=False)
    )
    omnivoice_bridge = ROOT / "bridge" / "omnivoice_segment_tts.py"
    try:
        import gradio_client  # noqa: F401
        import requests  # noqa: F401
        omnivoice_deps = True
    except Exception:
        omnivoice_deps = False
    checks.append(
        check("TTS Engines", "omnivoice (provider 2)",
              omnivoice_bridge.is_file() and omnivoice_deps,
              "bridge present + client deps available" if omnivoice_bridge.is_file() and omnivoice_deps
              else ("bridge present; install gradio_client + requests" if omnivoice_bridge.is_file()
                    else "bridge missing"),
              required=False)
    )

    # --- Age model plugin (optional) ---
    # Optional: griko/age_reg_ann_ecapa_librosa_combined for real age
    # regression. Falls back to the pitch heuristic when unavailable.
    # Checks three things: .venv-age exists, local model dir exists, and the
    # regressor loads. The plugin runs in its own venv, so the doctor (which
    # runs under the bare system interpreter) only checks for the presence of
    # the venv + model dir + a marker file written by setup_age_model.sh.
    age_venv = ROOT / ".venv-age"
    age_model_dir = ROOT / "models" / "age_reg_ann_ecapa_librosa_combined"
    age_model_available = age_venv.is_dir() and age_model_dir.is_dir()
    if age_model_available:
        age_model_detail = (
            f"ready (venv={age_venv.name}, model={age_model_dir.name})"
        )
    else:
        parts = []
        if not age_venv.is_dir():
            parts.append(".venv-age missing")
        if not age_model_dir.is_dir():
            parts.append("model dir missing")
        age_model_detail = (
            f"{'; '.join(parts)} — run `make setup-age` "
            "(pitch heuristic fallback otherwise)"
        )
    checks.append(
        check("Speaker", "age regression venv + model", age_model_available,
              age_model_detail, required=False)
    )
    # Also check the in-process plugin (only loads if voice_age_regressor is
    # importable from the current interpreter, which it usually isn't outside
    # .venv-age). This is informational; the venv+model check above is the
    # real readiness gate.
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from age_model import is_available as _age_available  # noqa: E402
        from age_model import last_load_error as _age_err  # noqa: E402
        in_proc = _age_available()
        in_proc_detail = (
            "loaded in current interpreter" if in_proc
            else f"not importable here (use .venv-age): {_age_err()[:60] or 'n/a'}"
        )
    except Exception as exc:
        in_proc = False
        in_proc_detail = f"plugin check error: {exc}"
    checks.append(
        check("Speaker", "age regression plugin (in-process)", in_proc,
              in_proc_detail, required=False)
    )

    # Demucs cache presence: warn if not cached so users know
    # Demucs will fall back to ffmpeg without --allow-demucs-download.
    demucs_cached = False
    try:
        import torch as _torch
        hub_dir = Path(_torch.hub.get_dir())
        for d in [hub_dir / "checkpoints",
                  Path.home() / "Library" / "Caches" / "torch" / "hub" / "checkpoints",
                  Path.home() / ".cache" / "torch" / "hub" / "checkpoints"]:
            if d.is_dir():
                if any("htdemucs" in p.name.lower() and p.stat().st_size > 1_000_000
                       for p in d.iterdir()):
                    demucs_cached = True
                    break
    except Exception:
        pass
    checks.append(
        check("Audio", "Demucs htdemucs cached", demucs_cached,
              "cached" if demucs_cached else "not cached (will fall back to ffmpeg)",
              required=demucs_required)
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
        "checks": [asdict(item) for item in checks],
    }


def print_table(checks: list[Check]) -> None:
    name_width = max(len(f"{item.section}: {item.name}") for item in checks)
    for item in checks:
        status = "PASS" if item.ok else ("WARN" if not item.required else "FAIL")
        label = f"{item.section}: {item.name}"
        print(f"[{status}] {label:<{name_width}}  {item.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local pyVideoTrans + Qwen3 readiness")
    parser.add_argument("--strict", action="store_true",
                        help="return nonzero for optional warnings too")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable readiness JSON")
    parser.add_argument("--need-demucs", action="store_true",
                        help="require Demucs for AI-based vocal separation")
    parser.add_argument("--need-vocal-separation", action="store_true",
                        help="require Demucs + vocal-separation dependencies")
    parser.add_argument("--qwen3-only", action="store_true",
                        help="only check Qwen3-local engine readiness")
    args = parser.parse_args()

    if args.qwen3_only:
        return _qwen3_only_main()

    checks = collect_checks(
        need_demucs=args.need_demucs,
        need_vocal_separation=args.need_vocal_separation,
        strict=args.strict,
    )
    summary = summarize(checks, strict=args.strict)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["ready"] else 1

    print_table(checks)
    print()
    print(f"READY: {'yes' if summary['ready'] else 'no'}")
    print(f"Required failures: {summary['required_failures']}")
    print(f"Optional warnings: {summary['optional_warnings']}")
    return 0 if summary["ready"] else 1


def _qwen3_only_main() -> int:
    """Run only Qwen3-local engine readiness checks."""
    imports_ok, missing = _qwen3_imports_ok()
    model_ok = _qwen3_model_present()
    print("Qwen3-local engine readiness")
    print(f"  python:        {sys.executable}")
    for mod in QWEN3_REQUIRED_IMPORTS:
        ok = mod not in missing
        print(f"  import {mod:<11} {'PASS' if ok else 'FAIL'}")
    print(f"  model present  {'PASS' if model_ok else 'FAIL'} "
          f"({QWEN3_MODEL_DIR})")
    ready = imports_ok and model_ok
    print("READY: " + ("yes" if ready else "no"))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
