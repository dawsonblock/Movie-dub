#!/usr/bin/env python3
"""Detect speaker gender from audio using fundamental frequency (F0) analysis.

Male voices typically have F0 in 85-180 Hz range.
Female voices typically have F0 in 165-255 Hz range.

Uses librosa's pyin algorithm for robust F0 estimation.

Output: prints "male", "female", or "unknown" and exits 0 on success.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
import subprocess


def extract_audio_sample(
    video_path: Path, duration: float = 30.0, sample_rate: int = 16000
) -> Path | None:
    """Extract up to `duration` seconds of audio from video as a WAV file."""
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
            if Path(p).is_file():
                ffmpeg = p
                break
    if not ffmpeg:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = Path(f.name)
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-i", video_path.as_posix(),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-t", str(duration),
        "-f", "wav",
        tmp.as_posix(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        return None
    return tmp


def detect_gender_from_audio(audio_path: Path) -> tuple[str, float]:
    """Analyze F0 of audio and return (gender, median_f0_hz).

    Returns ("male", f0) or ("female", f0) or ("unknown", 0.0).
    """
    try:
        import numpy as np
        import librosa
    except ImportError:
        return ("unknown", 0.0)

    try:
        y, sr = librosa.load(audio_path.as_posix(), sr=16000, mono=True)
    except (FileNotFoundError, ValueError, RuntimeError):
        return ("unknown", 0.0)

    if len(y) < sr * 0.5:
        return ("unknown", 0.0)

    # Use pyin for robust F0 estimation
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=65, fmax=400, sr=sr, frame_length=2048
        )
    except (ValueError, RuntimeError):
        return ("unknown", 0.0)

    # Filter to voiced frames only
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 10:
        return ("unknown", 0.0)

    median_f0 = float(np.median(voiced_f0))
    mean_f0 = float(np.mean(voiced_f0))

    # Use both median and mean for robustness
    avg_f0 = (median_f0 + mean_f0) / 2.0

    # Threshold: ~165 Hz is the boundary between male and female ranges
    if avg_f0 < 165.0:
        return ("male", median_f0)
    else:
        return ("female", median_f0)


def detect_gender(video_path: Path) -> tuple[str, float]:
    """Extract audio from video and detect gender.

    Returns (gender, median_f0_hz).
    """
    audio = extract_audio_sample(video_path)
    if audio is None:
        return ("unknown", 0.0)
    try:
        return detect_gender_from_audio(audio)
    finally:
        audio.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect speaker gender from video/audio")
    parser.add_argument("--input", required=True, help="path to video or audio file")
    parser.add_argument("--json", action="store_true", help="output as JSON")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    gender, f0 = detect_gender(input_path)

    if args.json:
        import json
        print(json.dumps({"gender": gender, "median_f0_hz": round(f0, 1)}))
    else:
        print(f"gender: {gender}")
        print(f"median_f0: {f0:.1f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
