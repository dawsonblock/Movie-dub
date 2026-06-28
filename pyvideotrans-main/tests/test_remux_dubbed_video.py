import importlib.util
import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
REMUX_PATH = ROOT / "scripts" / "remux_dubbed_video.py"
TEST_VIDEO = ROOT / "assets" / "pyvideotrans_test_clip.mp4"

SPEC = importlib.util.spec_from_file_location("remux_dubbed_video", REMUX_PATH)
remux_dubbed_video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remux_dubbed_video)


def _have_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None or (ROOT / "pyvideotrans-main" / "ffmpeg" / "ffprobe").is_file()


def _write_silent_wav(path: Path, duration: float, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration * sr)
    frames = struct.pack("<" + "h" * n, *([0] * n))
    with wave.open(path.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(frames)


pytestmark = pytest.mark.skipif(not _have_ffprobe() or not TEST_VIDEO.is_file(), reason="ffprobe/test clip unavailable")


def test_remux_provides_video_and_audio_streams(tmp_path):
    dubbed_audio = tmp_path / "dubbed_audio.wav"
    _write_silent_wav(dubbed_audio, 8.4)
    output_video = tmp_path / "final_dubbed.mp4"

    report = remux_dubbed_video.remux(TEST_VIDEO, dubbed_audio, output_video)

    assert report["status"] == "ok"
    assert report["has_video"] is True
    assert report["has_audio"] is True
    assert output_video.is_file()
    # Independently confirm streams via ffprobe.
    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            output_video.as_posix(),
        ],
        text=True, capture_output=True, check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    codec_types = {s.get("codec_type") for s in streams}
    assert "video" in codec_types
    assert "audio" in codec_types


def test_remux_missing_input_video_raises(tmp_path):
    dubbed_audio = tmp_path / "dubbed_audio.wav"
    _write_silent_wav(dubbed_audio, 1.0)
    with pytest.raises(FileNotFoundError, match="input video"):
        remux_dubbed_video.remux(tmp_path / "nope.mp4", dubbed_audio, tmp_path / "out.mp4")


def test_remux_missing_dubbed_audio_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="dubbed audio"):
        remux_dubbed_video.remux(TEST_VIDEO, tmp_path / "nope.wav", tmp_path / "out.mp4")
