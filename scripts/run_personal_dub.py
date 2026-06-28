#!/usr/bin/env python3
"""Dub a personal video end-to-end with pyVideoTrans + OpenVoice V2.

This is the user-facing CLI promised in Phase 10. It runs the full
pyVideoTrans VTV pipeline with the OpenVoice provider, then rebuilds
the dubbed audio track from the OpenVoice manifest and remuxes it into
a final dubbed MP4 at the requested output path.

Workflow:
  1. Validate inputs (video, reference WAV, venvs, checkpoints).
  2. Point OpenVoice segment preservation at a job generated_audio dir.
  3. Run pyVideoTrans cli.py --task vtv --tts_type 34.
  4. Locate the OpenVoice manifest + preserved segment WAVs.
  5. Build dubbed_audio.wav from the manifest.
  6. Remux into the final output MP4.
  7. Write report.json.

Example:
  python scripts/run_personal_dub.py \
    --input ~/Movies/test.mp4 \
    --source-language en \
    --target-language en \
    --reference voices/openvoice_default_reference.wav \
    --output ~/Movies/dubbed-test.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
OPENVOICE = ROOT / "OpenVoice-main"
DEFAULT_REFERENCE = ROOT / "voices" / "openvoice_default_reference.wav"
BUILD_AUDIO_SCRIPT = ROOT / "scripts" / "build_dubbed_audio_from_manifest.py"
REMUX_SCRIPT = ROOT / "scripts" / "remux_dubbed_video.py"
PYVT_CONFIG = PYVIDEOTRANS / "videotrans" / "cfg.json"
PRESERVE_KEY = "openvoice_preserve_dir"
OPENVOICE_PROVIDER = "34"  # OpenVoice V2(Local)


def die(msg: str, code: int = 1) -> int:
    print(f"run_personal_dub: FAIL\nReason: {msg}", file=sys.stderr)
    return code


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def local_ffprobe() -> str:
    bundled = PYVIDEOTRANS / "ffmpeg" / "ffprobe"
    return bundled.as_posix() if bundled.is_file() else "ffprobe"


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [local_ffprobe(), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path.as_posix()],
        text=True, capture_output=True, check=True,
    )
    return float(result.stdout.strip())


def set_preserve_dir(preserve_dir: Path) -> str | None:
    if not PYVT_CONFIG.is_file():
        return None
    try:
        cfg = read_json(PYVT_CONFIG)
    except Exception:
        cfg = {}
    previous = cfg.get(PRESERVE_KEY, "")
    preserve_dir.mkdir(parents=True, exist_ok=True)
    cfg[PRESERVE_KEY] = preserve_dir.as_posix()
    write_json(PYVT_CONFIG, cfg)
    return previous


def restore_preserve_dir(previous: str | None) -> None:
    if previous is None or not PYVT_CONFIG.is_file():
        return
    try:
        cfg = read_json(PYVT_CONFIG)
    except Exception:
        cfg = {}
    cfg[PRESERVE_KEY] = previous
    write_json(PYVT_CONFIG, cfg)


def latest_artifact(root: Path, start_time: float, pattern: str) -> Path | None:
    candidates = [p for p in root.glob(pattern) if p.stat().st_mtime >= start_time]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def latest_manifest(start_time: float) -> Path | None:
    return latest_artifact(PYVIDEOTRANS / "tmp", start_time, "**/openvoice-manifest-*.json")


def extract_manifest_from_output(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and {"ok", "error", "results"}.issubset(candidate):
            return candidate
    return None


def run_subprocess(cmd: list[str], cwd: Path, label: str) -> subprocess.CompletedProcess:
    print(f"[{label}] running: {' '.join(cmd[:2])} ...")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Dub a personal video with pyVideoTrans + OpenVoice V2")
    parser.add_argument("--input", required=True, help="absolute path to the input video (MP4)")
    parser.add_argument("--source-language", default="en", help="source language code (e.g. en, auto)")
    parser.add_argument("--target-language", default="en", help="target language code (e.g. en, es, zh-cn)")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE.as_posix(),
                        help="reference voice WAV for OpenVoice cloning")
    parser.add_argument("--output", required=True, help="absolute path for the final dubbed MP4")
    parser.add_argument("--model", default="tiny", help="ASR model (tiny/base/small/medium/large-v3)")
    parser.add_argument("--subtitle-type", default="1",
                        help="0=None 1=Hard 2=Soft 3=HardDual 4=SoftDual")
    parser.add_argument("--job-dir", default="", help="working directory (default: tmp/personal_dub/<timestamp>)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="allow the final remux even if some segments failed")
    args = parser.parse_args()

    input_video = Path(args.input).expanduser().resolve()
    output_video = Path(args.output).expanduser().resolve()
    reference = Path(args.reference).expanduser().resolve()
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"

    # --- Validate inputs ---
    if not input_video.is_file():
        return die(f"input video missing: {input_video}")
    if not reference.is_file():
        return die(f"reference WAV missing: {reference}")
    if not py_python.is_file():
        return die(f"pyVideoTrans venv python missing: {py_python}")
    if not (OPENVOICE / ".venv" / "bin" / "python").is_file():
        return die(f"OpenVoice venv python missing: {OPENVOICE / '.venv' / 'bin' / 'python'}")
    if not (OPENVOICE / "checkpoints_v2" / "converter" / "checkpoint.pth").is_file():
        return die("OpenVoice checkpoints missing. Run: make download-openvoice")
    if not BUILD_AUDIO_SCRIPT.is_file() or not REMUX_SCRIPT.is_file():
        return die("build/remux scripts missing from scripts/")

    # --- Set up job dir ---
    stamp = str(int(time.time()))
    job_dir = Path(args.job_dir).expanduser().resolve() if args.job_dir else (
        PYVIDEOTRANS / "tmp" / "personal_dub" / stamp
    )
    if job_dir.exists():
        shutil.rmtree(job_dir)
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_audio = job_dir / "generated_audio"
    stable_manifest = job_dir / "openvoice_manifest.json"
    dubbed_audio = job_dir / "dubbed_audio.wav"
    report_path = job_dir / "report.json"

    print(f"Input:    {input_video}")
    print(f"Output:   {output_video}")
    print(f"Reference: {reference}")
    print(f"Job dir:  {job_dir}")

    # --- Configure OpenVoice settings in pyVideoTrans ---
    # Set the reference voice + preserve dir so segment WAVs survive for rebuild.
    cfg_previous: dict = {}
    if PYVT_CONFIG.is_file():
        try:
            cfg = read_json(PYVT_CONFIG)
        except Exception:
            cfg = {}
        cfg_previous = dict(cfg)
        cfg["openvoice_default_reference"] = reference.as_posix()
        cfg[PRESERVE_KEY] = generated_audio.as_posix()
        write_json(PYVT_CONFIG, cfg)

    # --- Run pyVideoTrans VTV with OpenVoice provider ---
    started = time.time()
    cmd = [
        py_python.as_posix(), "cli.py",
        "--task", "vtv",
        "--name", input_video.as_posix(),
        "--recogn_type", "0",
        "--model_name", args.model,
        "--source_language_code", args.source_language,
        "--target_language_code", args.target_language,
        "--tts_type", OPENVOICE_PROVIDER,
        "--voice_role", "default",
        "--subtitle_type", args.subtitle_type,
        "--output-dir", output_dir.as_posix(),
        "--no-clear-cache",
        "--verbose",
    ]
    result = run_subprocess(cmd, PYVIDEOTRANS, "pyVideoTrans")

    # Restore the preserve_dir setting so we don't leak it into other runs.
    if PYVT_CONFIG.is_file():
        write_json(PYVT_CONFIG, cfg_previous)

    # --- Locate the OpenVoice manifest ---
    manifest = latest_manifest(started)
    if not manifest:
        extracted = extract_manifest_from_output(f"{result.stdout}\n{result.stderr}")
        if extracted:
            write_json(stable_manifest, extracted)
            manifest = stable_manifest
    elif manifest.is_file() and manifest != stable_manifest:
        shutil.copy2(manifest, stable_manifest)
        manifest = stable_manifest

    report = {
        "status": "fail",
        "input_video": input_video.as_posix(),
        "output_video": output_video.as_posix(),
        "job_dir": job_dir.as_posix(),
        "manifest": manifest.as_posix() if manifest else "",
        "dubbed_audio": dubbed_audio.as_posix(),
        "segments_total": 0,
        "segments_ok": 0,
        "segments_failed": 0,
        "final_duration_seconds": 0.0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if manifest and manifest.is_file():
        data = read_json(manifest)
        report.update({
            "segments_total": len(data.get("results", data.get("segments", []))),
            "segments_ok": data.get("ok", 0),
            "segments_failed": data.get("error", 0),
        })

    try:
        if result.returncode != 0:
            raise RuntimeError(f"pyVideoTrans exited with code {result.returncode}")
        if not manifest or not manifest.is_file():
            raise RuntimeError("OpenVoice manifest was not found")
        if report["segments_ok"] < 1:
            raise RuntimeError("OpenVoice manifest has no successful segment")

        # --- Build dubbed audio from manifest ---
        build_cmd = [
            sys.executable, BUILD_AUDIO_SCRIPT.as_posix(),
            "--manifest", stable_manifest.as_posix(),
            "--input-video", input_video.as_posix(),
            "--output-audio", dubbed_audio.as_posix(),
        ]
        if args.allow_partial:
            build_cmd.append("--allow-partial")
        build_result = run_subprocess(build_cmd, ROOT, "build-dubbed-audio")
        if build_result.returncode != 0 or not dubbed_audio.is_file():
            raise RuntimeError(f"dubbed audio build failed: {build_result.stderr[-1500:]}")

        # --- Remux into the final output MP4 ---
        remux_cmd = [
            sys.executable, REMUX_SCRIPT.as_posix(),
            "--input-video", input_video.as_posix(),
            "--dubbed-audio", dubbed_audio.as_posix(),
            "--output-video", output_video.as_posix(),
        ]
        remux_result = run_subprocess(remux_cmd, ROOT, "remux")
        if remux_result.returncode != 0 or not output_video.is_file():
            raise RuntimeError(f"remux failed: {remux_result.stderr[-1500:]}")

        final_duration = ffprobe_duration(output_video)
        report["status"] = "pass"
        report["final_duration_seconds"] = final_duration
        write_json(report_path, report)
        print()
        print("run_personal_dub: PASS")
        print(f"Final dubbed video: {output_video}")
        print(f"Duration: {final_duration:.3f}s")
        print(f"Segments: {report['segments_ok']} ok / {report['segments_total']} total")
        print(f"Report: {report_path}")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        write_json(report_path, report)
        return die(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
