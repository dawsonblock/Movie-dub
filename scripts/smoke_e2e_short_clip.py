#!/usr/bin/env python3
"""Run the included short MP4 through pyVideoTrans with OpenVoice TTS.

This smoke test uses the same deterministic job-scoped architecture as
run_personal_dub.py: explicit OpenVoice queue/manifest/log paths via
pyVideoTrans config, no global mtime scanning, and full config
restoration.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Shared helpers for review_segments.json + remux_command.json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dub_job_helpers import (
    extract_manifest_from_output as _extract_manifest_from_output,
    write_remux_command as _write_remux_command,
    write_review_file as _write_review_file,
)
from job_state import create_job_dir


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
INPUT_VIDEO = ROOT / "assets" / "pyvideotrans_test_clip.mp4"
JOB_DIR = PYVIDEOTRANS / "tmp" / "e2e_short_clip"
OUTPUT_DIR = JOB_DIR / "output"
REPORT_PATH = JOB_DIR / "report.json"
REVIEW_PATH = JOB_DIR / "review_segments.json"
REMUX_PATH = JOB_DIR / "remux_command.json"
DUBBED_AUDIO_PATH = JOB_DIR / "dubbed_audio.wav"
FINAL_VIDEO_PATH = JOB_DIR / "final_dubbed.mp4"
STABLE_MANIFEST_PATH = JOB_DIR / "openvoice_manifest.json"
GENERATED_AUDIO_DIR = JOB_DIR / "generated_audio"
SUBTITLES_DIR = JOB_DIR / "subtitles"
BUILD_AUDIO_SCRIPT = ROOT / "scripts" / "build_dubbed_audio_from_manifest.py"
REMUX_SCRIPT = ROOT / "scripts" / "remux_dubbed_video.py"
PYVT_CONFIG = PYVIDEOTRANS / "videotrans" / "cfg.json"
PRESERVE_KEY = "openvoice_preserve_dir"

# Job-scoped explicit paths (same as run_personal_dub.py)
OPENVOICE_QUEUE = JOB_DIR / "openvoice_queue.json"
OPENVOICE_MANIFEST_EXPLICIT = JOB_DIR / "openvoice_manifest_explicit.json"
OPENVOICE_LOGS = JOB_DIR / "openvoice_bridge.log"
OPENVOICE_WORK = JOB_DIR / "openvoice_work"


def local_ffprobe() -> str:
    bundled = PYVIDEOTRANS / "ffmpeg" / "ffprobe"
    return bundled.as_posix() if bundled.is_file() else "ffprobe"


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            local_ffprobe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            path.as_posix(),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return float(result.stdout.strip())


def newest_mp4(output_dir: Path, start_time: float) -> Path | None:
    candidates = [
        path
        for path in output_dir.rglob("*.mp4")
        if path.is_file() and path.stat().st_mtime >= start_time
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def write_report(report: dict) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_remux_command() -> None:
    _write_remux_command(
        REMUX_PATH, INPUT_VIDEO, DUBBED_AUDIO_PATH, FINAL_VIDEO_PATH, STABLE_MANIFEST_PATH
    )


def run_build_dubbed_audio() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        BUILD_AUDIO_SCRIPT.as_posix(),
        "--manifest",
        STABLE_MANIFEST_PATH.as_posix(),
        "--input-video",
        INPUT_VIDEO.as_posix(),
        "--output-audio",
        DUBBED_AUDIO_PATH.as_posix(),
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def run_remux_dubbed_video() -> tuple[bool, str]:
    cmd = [
        sys.executable,
        REMUX_SCRIPT.as_posix(),
        "--input-video",
        INPUT_VIDEO.as_posix(),
        "--dubbed-audio",
        DUBBED_AUDIO_PATH.as_posix(),
        "--output-video",
        FINAL_VIDEO_PATH.as_posix(),
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def configure_pyvt_config() -> dict[str, dict]:
    """Set job-scoped explicit paths in pyVideoTrans config.

    Returns a dict of {key: {"existed": bool, "value": any}} for restoration.
    """
    PYVT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if PYVT_CONFIG.is_file():
        try:
            cfg = json.loads(PYVT_CONFIG.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    else:
        cfg = {}

    cfg_original: dict[str, dict] = {}
    for key in (
        PRESERVE_KEY,
        "openvoice_queue_file",
        "openvoice_manifest_file",
        "openvoice_logs_file",
        "openvoice_work_dir",
    ):
        cfg_original[key] = {
            "existed": key in cfg,
            "value": cfg.get(key),
        }
    GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OPENVOICE_WORK.mkdir(parents=True, exist_ok=True)
    cfg[PRESERVE_KEY] = GENERATED_AUDIO_DIR.as_posix()
    cfg["openvoice_queue_file"] = OPENVOICE_QUEUE.as_posix()
    cfg["openvoice_manifest_file"] = OPENVOICE_MANIFEST_EXPLICIT.as_posix()
    cfg["openvoice_logs_file"] = OPENVOICE_LOGS.as_posix()
    cfg["openvoice_work_dir"] = OPENVOICE_WORK.as_posix()
    PYVT_CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg_original


def restore_pyvt_config(cfg_original: dict[str, dict]) -> None:
    """Restore pyVideoTrans config to its original state."""
    if not PYVT_CONFIG.is_file():
        return
    try:
        current = json.loads(PYVT_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        current = {}
    for key, state in cfg_original.items():
        if state["existed"]:
            current[key] = state["value"]
        else:
            current.pop(key, None)
    PYVT_CONFIG.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def write_review_file(queue_file: Path | None, manifest_file: Path | None, srt_file: Path | None) -> None:
    _write_review_file(
        REVIEW_PATH, queue_file, manifest_file, srt_file, JOB_DIR, GENERATED_AUDIO_DIR
    )


def main() -> int:
    if not INPUT_VIDEO.is_file():
        print(f"Missing test video: {INPUT_VIDEO}", file=sys.stderr)
        return 1
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"
    if not py_python.is_file():
        print(f"Missing pyVideoTrans venv python: {py_python}", file=sys.stderr)
        return 1

    if JOB_DIR.exists():
        create_job_dir(JOB_DIR, force=True, allow_external=True)
    else:
        create_job_dir(JOB_DIR, force=False, allow_external=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cfg_original = configure_pyvt_config()
    cmd = [
        py_python.as_posix(),
        "cli.py",
        "--task",
        "vtv",
        "--name",
        INPUT_VIDEO.as_posix(),
        "--recogn_type",
        "0",
        "--model_name",
        "tiny",
        "--source_language_code",
        "en",
        "--target_language_code",
        "en",
        "--tts_type",
        "34",
        "--voice_role",
        "default",
        "--subtitle_type",
        "1",
        "--output-dir",
        OUTPUT_DIR.as_posix(),
        "--no-clear-cache",
        "--verbose",
    ]
    try:
        result = subprocess.run(cmd, cwd=PYVIDEOTRANS, text=True, capture_output=True)
    finally:
        restore_pyvt_config(cfg_original)
    final_video = newest_mp4(OUTPUT_DIR, started) or OUTPUT_DIR / INPUT_VIDEO.name

    # --- Locate manifest (explicit path, no global mtime scan) ---
    manifest: Path | None = None
    if OPENVOICE_MANIFEST_EXPLICIT.is_file():
        shutil.copy2(OPENVOICE_MANIFEST_EXPLICIT, STABLE_MANIFEST_PATH)
        manifest = STABLE_MANIFEST_PATH
    if not manifest:
        extracted_manifest = _extract_manifest_from_output(f"{result.stdout}\n{result.stderr}")
        if extracted_manifest:
            STABLE_MANIFEST_PATH.write_text(
                json.dumps(extracted_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest = STABLE_MANIFEST_PATH

    # --- Locate queue (explicit path, no global mtime scan) ---
    queue_file = OPENVOICE_QUEUE if OPENVOICE_QUEUE.is_file() else None

    # --- Locate SRT (copy newly created SRTs into job dir) ---
    srt_file: Path | None = None
    job_srt = SUBTITLES_DIR / "translated.srt"
    new_srts = [
        p for p in (PYVIDEOTRANS / "tmp").rglob("*.srt")
        if p.stat().st_mtime >= started
    ]
    if new_srts:
        chosen = max(new_srts, key=lambda p: p.stat().st_mtime)
        shutil.copy2(chosen, job_srt)
        srt_file = job_srt

    report = {
        "status": "fail",
        "input_video": INPUT_VIDEO.as_posix(),
        "output_video": final_video.as_posix(),
        "final_dubbed_video": FINAL_VIDEO_PATH.as_posix(),
        "dubbed_audio": DUBBED_AUDIO_PATH.as_posix(),
        "segments_total": 0,
        "segments_ok": 0,
        "segments_failed": 0,
        "duration_seconds": 0.0,
        "final_duration_seconds": 0.0,
        "tts_backend": "openvoice",
        "device": "",
        "manifest": manifest.as_posix() if manifest else "",
        "review_segments": REVIEW_PATH.as_posix(),
        "remux_command": REMUX_PATH.as_posix(),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if manifest and manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        results = data.get("results", [])
        report.update(
            {
                "segments_total": len(results),
                "segments_ok": data.get("ok", 0),
                "segments_failed": data.get("error", 0),
                "device": data.get("device", ""),
            }
        )

    try:
        if result.returncode != 0:
            raise RuntimeError(f"pyVideoTrans exited with code {result.returncode}")
        if not final_video.is_file():
            raise RuntimeError(f"final MP4 missing: {final_video}")
        duration = ffprobe_duration(final_video)
        if duration <= 0:
            raise RuntimeError(f"invalid output duration: {duration}")
        if not manifest:
            raise RuntimeError("OpenVoice manifest was not found")
        if report["segments_ok"] < 1:
            raise RuntimeError("OpenVoice manifest has no successful segment")
        write_review_file(queue_file, manifest, srt_file)
        write_remux_command()
        # Build the real dubbed audio track from the manifest.
        audio_ok, audio_log = run_build_dubbed_audio()
        if not audio_ok or not DUBBED_AUDIO_PATH.is_file():
            raise RuntimeError(f"dubbed audio build failed: {audio_log[-1500:]}")
        # Remux the dubbed audio into a final MP4.
        remux_ok, remux_log = run_remux_dubbed_video()
        if not remux_ok or not FINAL_VIDEO_PATH.is_file():
            raise RuntimeError(f"remux failed: {remux_log[-1500:]}")
        final_duration = ffprobe_duration(FINAL_VIDEO_PATH)
        if final_duration <= 0:
            raise RuntimeError(f"invalid final dubbed duration: {final_duration}")
        report["status"] = "pass"
        report["duration_seconds"] = duration
        report["final_duration_seconds"] = final_duration
        write_report(report)
        print("E2E short clip smoke: PASS")
        print(f"pyVideoTrans output: {final_video}")
        print(f"Final dubbed video: {FINAL_VIDEO_PATH}")
        print(f"Dubbed audio: {DUBBED_AUDIO_PATH}")
        print(f"Report: {REPORT_PATH}")
        print(f"Manifest: {manifest}")
        print(f"Duration: {duration:.3f}s  Final: {final_duration:.3f}s")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        write_report(report)
        print("E2E short clip smoke: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print(f"Report: {REPORT_PATH}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
