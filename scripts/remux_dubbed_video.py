#!/usr/bin/env python3
"""Mux a rebuilt dubbed audio track into the original video.

Takes the original video and the dubbed_audio.wav produced by
build_dubbed_audio_from_manifest.py and writes final_dubbed.mp4 with the
original video stream copied and the new audio encoded as AAC.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def local_ffmpeg() -> str:
    """Find ffmpeg: prefer bundled, then PATH, then common Homebrew locations."""
    candidates = [
        Path(__file__).resolve().parents[1] / "pyvideotrans-main" / "ffmpeg" / "ffmpeg",
        Path(shutil.which("ffmpeg") or ""),
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    raise RuntimeError("ffmpeg not found")


def local_ffprobe() -> str:
    """Find ffprobe: prefer bundled, then PATH, then common Homebrew locations."""
    candidates = [
        Path(__file__).resolve().parents[1] / "pyvideotrans-main" / "ffmpeg" / "ffprobe",
        Path(shutil.which("ffprobe") or ""),
        Path("/opt/homebrew/bin/ffprobe"),
        Path("/usr/local/bin/ffprobe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    raise RuntimeError("ffprobe not found")


def ffprobe_streams(path: Path) -> list[dict]:
    result = subprocess.run(
        [
            local_ffprobe(),
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name",
            "-of",
            "json",
            path.as_posix(),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout).get("streams", [])


def ffprobe_full(path: Path) -> dict:
    """Run ffprobe -show_streams -show_format and return parsed JSON."""
    result = subprocess.run(
        [
            local_ffprobe(),
            "-v", "error",
            "-show_streams", "-show_format",
            "-of", "json",
            path.as_posix(),
        ],
        text=True, capture_output=True, check=True, timeout=30,
    )
    return json.loads(result.stdout)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            local_ffprobe(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nk=1:nw=1",
            path.as_posix(),
        ],
        text=True, capture_output=True, check=True,
    )
    return float(result.stdout.strip())


def verify_final_video(
    output_video: Path, source_duration: float
) -> dict:
    """Verify a final MP4 with ffprobe (Blocker 11).

    Checks: video stream exists, audio stream exists, duration exists,
    duration is close to source, audio/video codecs exist, file size > 1KB.
    Duration mismatch fails if larger than max(2s, 2% of source duration).
    """
    try:
        info = ffprobe_full(output_video)
    except Exception as exc:
        return {"verified": False, "error": str(exc)}

    streams = info.get("streams", [])
    fmt = info.get("format", {})
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video_codec = next(
        (s.get("codec_name") for s in streams if s.get("codec_type") == "video"),
        None,
    )
    audio_codec = next(
        (s.get("codec_name") for s in streams if s.get("codec_type") == "audio"),
        None,
    )
    duration_str = fmt.get("duration", "0")
    try:
        output_duration = float(duration_str)
    except (ValueError, TypeError):
        output_duration = 0.0

    errors: list[str] = []
    if not has_video:
        errors.append("missing video stream")
    if not has_audio:
        errors.append("missing audio stream")
    if not video_codec:
        errors.append("missing video codec")
    if not audio_codec:
        errors.append("missing audio codec")
    if output_duration <= 0:
        errors.append("invalid duration")
    else:
        delta = abs(output_duration - source_duration)
        tolerance = max(2.0, 0.02 * source_duration)
        if delta > tolerance:
            errors.append(
                f"duration mismatch: {delta:.1f}s exceeds tolerance "
                f"{tolerance:.1f}s (source={source_duration:.1f}, "
                f"output={output_duration:.1f})"
            )
    file_size = output_video.stat().st_size if output_video.is_file() else 0
    if file_size < 1024:
        errors.append(f"file too small: {file_size} bytes")

    return {
        "verified": len(errors) == 0,
        "source_duration": source_duration,
        "output_duration": output_duration,
        "duration_delta": abs(output_duration - source_duration),
        "has_video": has_video,
        "has_audio": has_audio,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "file_size": file_size,
        "errors": errors,
    }


def remux(input_video: Path, dubbed_audio: Path, output_video: Path) -> dict:
    if not input_video.is_file():
        raise FileNotFoundError(f"input video missing: {input_video}")
    if not dubbed_audio.is_file():
        raise FileNotFoundError(f"dubbed audio missing: {dubbed_audio}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        local_ffmpeg(),
        "-y",
        "-i",
        input_video.as_posix(),
        "-i",
        dubbed_audio.as_posix(),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_video.as_posix(),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg remux failed with code {result.returncode}: {result.stderr[-2000:]}"
        )
    if not output_video.is_file():
        raise RuntimeError(f"ffmpeg produced no output: {output_video}")

    streams = ffprobe_streams(output_video)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_video or not has_audio:
        raise RuntimeError(
            f"output missing streams: video={has_video} audio={has_audio} streams={streams}"
        )

    # Full verification (Blocker 11)
    source_duration = ffprobe_duration(input_video)
    verification = verify_final_video(output_video, source_duration)
    if not verification["verified"]:
        raise RuntimeError(
            f"final video verification failed: {verification['errors']}"
        )

    return {
        "status": "ok",
        "input_video": input_video.as_posix(),
        "dubbed_audio": dubbed_audio.as_posix(),
        "output_video": output_video.as_posix(),
        "has_video": has_video,
        "has_audio": has_audio,
        "streams": streams,
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Remux dubbed audio into the original video")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--dubbed-audio", required=True)
    parser.add_argument("--output-video", required=True)
    args = parser.parse_args()

    input_video = Path(args.input_video).expanduser().resolve()
    dubbed_audio = Path(args.dubbed_audio).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()

    try:
        report = remux(input_video, dubbed_audio, output_video)
        print("Remux dubbed video: PASS")
        print(f"Output: {output_video}")
        print(f"Video stream: {report['has_video']}  Audio stream: {report['has_audio']}")
        v = report.get("verification", {})
        if v:
            print(f"Duration: {v.get('output_duration', 0):.3f}s "
                  f"(delta {v.get('duration_delta', 0):.3f}s)")
            print(f"Codecs: video={v.get('video_codec')} audio={v.get('audio_codec')}")
        return 0
    except Exception as exc:
        print("Remux dubbed video: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
