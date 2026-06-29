"""Tests for the gender detection script."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENDER_SCRIPT = ROOT / "scripts" / "detect_gender.py"
TEST_VIDEO = ROOT / "assets" / "pyvideotrans_test_clip.mp4"

SPEC = importlib.util.spec_from_file_location("detect_gender", GENDER_SCRIPT)
detect_gender = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detect_gender)


def test_detect_gender_returns_tuple():
    """detect_gender should return (gender_str, f0_float)."""
    if not TEST_VIDEO.is_file():
        pytest.skip("test video not found")
    gender, f0 = detect_gender.detect_gender(TEST_VIDEO)
    assert gender in ("male", "female", "unknown")
    assert isinstance(f0, float)
    assert f0 >= 0.0


def test_detect_gender_from_audio_with_silence(tmp_path):
    """A silent audio file should return unknown gender."""
    import struct
    import wave
    silent_wav = tmp_path / "silence.wav"
    sr = 16000
    duration = 2.0
    n = int(duration * sr)
    frames = b"\x00\x00" * n
    with wave.open(silent_wav.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(frames)
    gender, f0 = detect_gender.detect_gender_from_audio(silent_wav)
    assert gender == "unknown"
    assert f0 == 0.0


def test_detect_gender_from_nonexistent_file():
    """Non-existent file should return unknown."""
    gender, f0 = detect_gender.detect_gender_from_audio(Path("/nonexistent/audio.wav"))
    assert gender == "unknown"
    assert f0 == 0.0


def test_detect_gender_from_short_audio(tmp_path):
    """Very short audio (< 0.5s) should return unknown."""
    import struct
    import wave
    short_wav = tmp_path / "short.wav"
    sr = 16000
    n = int(0.1 * sr)  # 100ms - too short
    frames = b"\x00\x00" * n
    with wave.open(short_wav.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(frames)
    gender, f0 = detect_gender.detect_gender_from_audio(short_wav)
    assert gender == "unknown"
