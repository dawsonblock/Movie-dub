"""Tests for remux verification (Blocker 11)."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


REMUX_PATH = Path(__file__).resolve().parents[2] / "scripts" / "remux_dubbed_video.py"
SPEC = importlib.util.spec_from_file_location("remux", REMUX_PATH)
remux = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = remux
SPEC.loader.exec_module(remux)


def test_verify_rejects_missing_audio_stream():
    """Verification should fail for a video with no audio stream (Blocker 11)."""
    fake_info = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
        ],
        "format": {"duration": "91.4"},
    }
    with patch.object(remux, "ffprobe_full", return_value=fake_info), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value = MagicMock(st_size=1000000)
        result = remux.verify_final_video(Path("/fake.mp4"), 91.4)

    assert result["verified"] is False
    assert "missing audio stream" in result["errors"]


def test_verify_rejects_missing_video_stream():
    """Verification should fail for a video with no video stream (Blocker 11)."""
    fake_info = {
        "streams": [
            {"index": 0, "codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "91.4"},
    }
    with patch.object(remux, "ffprobe_full", return_value=fake_info), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value = MagicMock(st_size=1000000)
        result = remux.verify_final_video(Path("/fake.mp4"), 91.4)

    assert result["verified"] is False
    assert "missing video stream" in result["errors"]


def test_verify_rejects_duration_mismatch():
    """Verification should fail for large duration mismatch (Blocker 11)."""
    fake_info = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "50.0"},  # source is 91.4, delta is 41.4
    }
    with patch.object(remux, "ffprobe_full", return_value=fake_info), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value = MagicMock(st_size=1000000)
        result = remux.verify_final_video(Path("/fake.mp4"), 91.4)

    assert result["verified"] is False
    assert any("duration mismatch" in e for e in result["errors"])


def test_verify_passes_valid_video():
    """Verification should pass for a valid video with matching duration (Blocker 11)."""
    fake_info = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "91.1"},  # delta 0.3, within tolerance
    }
    with patch.object(remux, "ffprobe_full", return_value=fake_info), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value = MagicMock(st_size=1000000)
        result = remux.verify_final_video(Path("/fake.mp4"), 91.4)

    assert result["verified"] is True
    assert result["has_video"] is True
    assert result["has_audio"] is True
    assert result["video_codec"] == "h264"
    assert result["audio_codec"] == "aac"


def test_verify_rejects_small_file():
    """Verification should fail for a file smaller than 1KB (Blocker 11)."""
    fake_info = {
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "91.1"},
    }
    with patch.object(remux, "ffprobe_full", return_value=fake_info), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value = MagicMock(st_size=100)  # < 1024
        result = remux.verify_final_video(Path("/fake.mp4"), 91.4)

    assert result["verified"] is False
    assert any("too small" in e for e in result["errors"])
