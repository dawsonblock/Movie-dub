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

    return {
        "status": "ok",
        "input_video": input_video.as_posix(),
        "dubbed_audio": dubbed_audio.as_posix(),
        "output_video": output_video.as_posix(),
        "has_video": has_video,
        "has_audio": has_audio,
        "streams": streams,
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
        return 0
    except Exception as exc:
        print("Remux dubbed video: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
