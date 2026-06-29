#!/usr/bin/env python3
"""Shared helpers for writing review_segments.json and remux_command.json.

Used by both smoke_e2e_short_clip.py and run_personal_dub.py so that every
dub job — smoke or personal — produces the artifacts needed for
regenerate_segment.py --remux to rebuild the final video.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_AUDIO_SCRIPT = ROOT / "scripts" / "build_dubbed_audio_from_manifest.py"
REMUX_SCRIPT = ROOT / "scripts" / "remux_dubbed_video.py"


def parse_srt(path: Path | None) -> list[dict]:
    """Parse an SRT subtitle file into a list of {text, start, end} dicts."""
    if not path or not Path(path).is_file():
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
        items.append({
            "text": text,
            "start": parse_timecode(start_raw),
            "end": parse_timecode(end_raw),
        })
    return items


def extract_manifest_from_output(text: str) -> dict | None:
    """Scan stdout/stderr text for an OpenVoice manifest JSON object."""
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


def write_remux_command(
    remux_path: Path,
    input_video: Path,
    dubbed_audio: Path,
    output_video: Path,
    manifest_path: Path,
    audio_options: dict | None = None,
) -> None:
    """Write remux_command.json with real build-audio + remux commands.

    This dict format is consumed by regenerate_segment.py --remux to replay
    the audio rebuild and video remux after a segment is regenerated.

    If audio_options is provided, those settings are included in the
    build_audio_command so regeneration uses the same audio-quality
    settings as the initial dub.
    """
    build_cmd = [
        sys.executable,
        BUILD_AUDIO_SCRIPT.as_posix(),
        "--manifest",
        manifest_path.as_posix(),
        "--input-video",
        input_video.as_posix(),
        "--output-audio",
        dubbed_audio.as_posix(),
    ]
    # Include --report-json so regeneration also produces a build report
    report_json_path = dubbed_audio.parent / "build_audio_report.json"
    build_cmd.extend(["--report-json", report_json_path.as_posix()])
    if audio_options:
        if audio_options.get("background_volume", 0) > 0:
            build_cmd.extend(
                ["--background-volume", str(audio_options["background_volume"])]
            )
        if audio_options.get("voice_volume", 1.0) != 1.0:
            build_cmd.extend(
                ["--voice-volume", str(audio_options["voice_volume"])]
            )
        if audio_options.get("final_gain", 1.0) != 1.0:
            build_cmd.extend(
                ["--final-gain", str(audio_options["final_gain"])]
            )
        if audio_options.get("no_normalize"):
            build_cmd.append("--no-normalize")
        if audio_options.get("vocal_separation"):
            build_cmd.append("--vocal-separation")
            method = audio_options.get("vocal_separation_method", "ffmpeg")
            build_cmd.extend(["--vocal-separation-method", method])
        if audio_options.get("ducking"):
            build_cmd.append("--ducking")
        if audio_options.get("target_lufs") is not None:
            build_cmd.extend(
                ["--target-lufs", str(audio_options["target_lufs"])]
            )
        if audio_options.get("background_timeout", 300) != 300:
            build_cmd.extend(
                ["--background-timeout",
                 str(audio_options["background_timeout"])]
            )
        if audio_options.get("lufs_timeout", 600) != 600:
            build_cmd.extend(
                ["--lufs-timeout", str(audio_options["lufs_timeout"])]
            )
        if audio_options.get("demucs_timeout", 600) != 600:
            build_cmd.extend(
                ["--demucs-timeout", str(audio_options["demucs_timeout"])]
            )
        if audio_options.get("fail_if_background_mix_fails"):
            build_cmd.append("--fail-if-background-mix-fails")

    command = {
        "command": [
            sys.executable,
            REMUX_SCRIPT.as_posix(),
            "--input-video",
            input_video.as_posix(),
            "--dubbed-audio",
            dubbed_audio.as_posix(),
            "--output-video",
            output_video.as_posix(),
        ],
        "build_audio_command": build_cmd,
        "audio_options": audio_options or {},
    }
    remux_path.parent.mkdir(parents=True, exist_ok=True)
    remux_path.write_text(
        json.dumps(command, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_review_file(
    review_path: Path,
    queue_file: Path | None,
    manifest_file: Path | None,
    srt_file: Path | None,
    job_dir: Path,
    generated_audio_dir: Path,
) -> None:
    """Write review_segments.json from the manifest + queue/SRT data.

    Each review entry includes id, start, end, source/translated/edited text,
    output_audio path, status, reason, role, ref_wav, language, and device.
    Segment WAVs are copied from preserved_audio (or output_audio) into the
    generated_audio dir so regeneration can find them.
    """
    if not manifest_file or not manifest_file.is_file():
        return

    if queue_file and queue_file.is_file():
        queue = json.loads(queue_file.read_text(encoding="utf-8"))
    else:
        srt_items = parse_srt(srt_file)
        manifest_preview = json.loads(manifest_file.read_text(encoding="utf-8"))
        queue = []
        for index, result in enumerate(manifest_preview.get("results", [])):
            # Prefer SRT text, then fall back to the manifest's own text field
            # so review entries are never empty when the manifest has usable text.
            srt_text = srt_items[index]["text"] if index < len(srt_items) else ""
            text = (
                srt_text
                or result.get("text")
                or result.get("target_text")
                or ""
            )
            start = (
                srt_items[index]["start"]
                if index < len(srt_items)
                else result.get("start", 0.0)
            )
            end = (
                srt_items[index]["end"]
                if index < len(srt_items)
                else result.get("end", result.get("target_duration", 0.0))
            )
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
    manifest_by_id = {
        str(item.get("id", item.get("index"))): item
        for item in manifest.get("results", [])
    }
    review = []
    for index, item in enumerate(queue):
        segment_id = item.get("id", item.get("line", index + 1))
        result = manifest_by_id.get(str(segment_id), {})
        timing = result.get("timing_status", "unknown")
        status = "needs_review" if timing in {
            "rewrite_shorter", "padding_or_slowdown_candidate"
        } else "ok"
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
                "output_audio": (generated_audio_dir / f"{segment_id}.wav").as_posix(),
                "status": status,
                "reason": timing,
                "role": item.get("role", "clone"),
                "ref_wav": item.get("ref_wav") or item.get("voice_reference", ""),
                "language": manifest.get("language", "EN"),
                "device": manifest.get("device", "auto"),
            }
        )
        original_audio = Path(str(result.get("preserved_audio") or result.get("output_audio", "")))
        if original_audio.is_file():
            target_audio = Path(review[-1]["output_audio"])
            target_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_audio, target_audio)
        elif result.get("preserved_audio"):
            review[-1]["output_audio"] = result.get("preserved_audio")

    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
