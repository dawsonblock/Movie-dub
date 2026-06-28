#!/usr/bin/env python3
"""Run the included short MP4 through pyVideoTrans with OpenVoice TTS."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
INPUT_VIDEO = ROOT / "assets" / "pyvideotrans_test_clip.mp4"
JOB_DIR = PYVIDEOTRANS / "tmp" / "e2e_short_clip"
OUTPUT_DIR = JOB_DIR / "output"
REPORT_PATH = JOB_DIR / "report.json"
REVIEW_PATH = JOB_DIR / "review_segments.json"
REMUX_PATH = JOB_DIR / "remux_command.json"


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


def latest_artifact(start_time: float, pattern: str) -> Path | None:
    candidates = [
        path
        for path in (PYVIDEOTRANS / "tmp").glob(pattern)
        if path.stat().st_mtime >= start_time
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def latest_manifest(start_time: float) -> Path | None:
    return latest_artifact(start_time, "**/openvoice-manifest-*.json")


def latest_queue(start_time: float) -> Path | None:
    return latest_artifact(start_time, "**/openvoice-queue-*.json")


def newest_mp4(output_dir: Path, start_time: float) -> Path | None:
    candidates = [
        path
        for path in output_dir.rglob("*.mp4")
        if path.is_file() and path.stat().st_mtime >= start_time
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


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


def parse_srt(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    def parse_timecode(value: str) -> float:
        match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value.strip())
        if not match:
            return 0.0
        hours, minutes, seconds, millis = [int(part) for part in match.groups()]
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    blocks = path.read_text(encoding="utf-8", errors="ignore").strip().split("\n\n")
    items = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[1].split("-->", 1)]
        text = " ".join(lines[2:])
        items.append({"text": text, "start": parse_timecode(start_raw), "end": parse_timecode(end_raw)})
    return items


def write_report(report: dict) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_remux_command(final_video: Path) -> None:
    relative_video = final_video.relative_to(JOB_DIR) if final_video.is_relative_to(JOB_DIR) else final_video
    command = ["/bin/cp", relative_video.as_posix(), "final_dubbed.mp4"]
    REMUX_PATH.write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")


def write_review_file(queue_file: Path | None, manifest_file: Path | None, srt_file: Path) -> None:
    if not manifest_file or not manifest_file.is_file():
        return
    if queue_file and queue_file.is_file():
        queue = json.loads(queue_file.read_text(encoding="utf-8"))
    else:
        srt_items = parse_srt(srt_file)
        manifest_preview = json.loads(manifest_file.read_text(encoding="utf-8"))
        queue = []
        for index, result in enumerate(manifest_preview.get("results", [])):
            text = srt_items[index]["text"] if index < len(srt_items) else ""
            start = srt_items[index]["start"] if index < len(srt_items) else 0.0
            end = srt_items[index]["end"] if index < len(srt_items) else result.get("target_duration", 0.0)
            queue.append(
                {
                    "id": result.get("id", index + 1),
                    "line": result.get("id", index + 1),
                    "text": text,
                    "start": start,
                    "end": end,
                    "filename": result.get("output_audio", ""),
                }
            )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_by_id = {str(item.get("id", item.get("index"))): item for item in manifest.get("results", [])}
    review = []
    for index, item in enumerate(queue):
        segment_id = item.get("id", item.get("line", index + 1))
        result = manifest_by_id.get(str(segment_id), {})
        timing = result.get("timing_status", "unknown")
        status = "needs_review" if timing in {"rewrite_shorter", "padding_or_slowdown_candidate"} else "ok"
        if "start_time" in item or "end_time" in item:
            start = float(item.get("start_time", 0.0)) / 1000.0
            end = float(item.get("end_time", 0.0)) / 1000.0
        else:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        review.append(
            {
                "id": segment_id,
                "start": start,
                "end": end,
                "source_text": item.get("source_text", ""),
                "translated_text": item.get("text", item.get("target_text", "")),
                "edited_text": "",
                "output_audio": (JOB_DIR / "generated_audio" / f"{segment_id}.wav").as_posix(),
                "status": status,
                "reason": timing,
                "role": item.get("role", "clone"),
                "ref_wav": item.get("ref_wav") or item.get("voice_reference", ""),
                "language": manifest.get("language", "EN"),
                "device": manifest.get("device", "auto"),
            }
        )
        original_audio = Path(str(result.get("output_audio", "")))
        if original_audio.is_file():
            target_audio = Path(review[-1]["output_audio"])
            target_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_audio, target_audio)
    write_json = json.dumps(review, ensure_ascii=False, indent=2)
    REVIEW_PATH.write_text(write_json, encoding="utf-8")


def main() -> int:
    if not INPUT_VIDEO.is_file():
        print(f"Missing test video: {INPUT_VIDEO}", file=sys.stderr)
        return 1
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"
    if not py_python.is_file():
        print(f"Missing pyVideoTrans venv python: {py_python}", file=sys.stderr)
        return 1

    if JOB_DIR.exists():
        shutil.rmtree(JOB_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
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
    result = subprocess.run(cmd, cwd=PYVIDEOTRANS, text=True, capture_output=True)
    final_video = newest_mp4(OUTPUT_DIR, started) or OUTPUT_DIR / INPUT_VIDEO.name
    manifest = latest_manifest(started)
    queue_file = latest_queue(started)
    stable_manifest = JOB_DIR / "openvoice_manifest.json"
    if not manifest:
        extracted_manifest = extract_manifest_from_output(f"{result.stdout}\n{result.stderr}")
        if extracted_manifest:
            write_json = json.dumps(extracted_manifest, ensure_ascii=False, indent=2)
            stable_manifest.write_text(write_json, encoding="utf-8")
            manifest = stable_manifest
    report = {
        "status": "fail",
        "input_video": INPUT_VIDEO.as_posix(),
        "output_video": final_video.as_posix(),
        "segments_total": 0,
        "segments_ok": 0,
        "segments_failed": 0,
        "duration_seconds": 0.0,
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
        write_review_file(queue_file, manifest, OUTPUT_DIR / "en.srt")
        write_remux_command(final_video)
        report["status"] = "pass"
        report["duration_seconds"] = duration
        write_report(report)
        print("E2E short clip smoke: PASS")
        print(f"Output video: {final_video}")
        print(f"Report: {REPORT_PATH}")
        print(f"Manifest: {manifest}")
        print(f"Duration: {duration:.3f}s")
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
