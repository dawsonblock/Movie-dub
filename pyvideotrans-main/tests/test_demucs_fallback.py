"""Tests for Demucs fallback robustness (Blocker 2)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


BUILD_AUDIO_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_dubbed_audio_from_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_audio", BUILD_AUDIO_PATH)
build_audio = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_audio
SPEC.loader.exec_module(build_audio)


def test_mix_background_returns_dict_with_required_fields():
    """_mix_background_audio should return a dict with required report fields (Blocker 2)."""
    # Create a minimal fake canvas and input
    # We'll mock the ffmpeg subprocess so no real ffmpeg is needed
    canvas = [0.0] * 100
    input_video = Path("/fake/video.mp4")

    with patch.object(build_audio, "local_ffmpeg", return_value="/fake/ffmpeg"), \
         patch.object(build_audio, "load_audio_samples", return_value=([0.1] * 100, 44100)), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = build_audio._mix_background_audio(
            canvas, input_video, 44100, 100, 0.15,
            vocal_separation=False,
        )

    assert isinstance(result, dict)
    assert "background_mix_applied" in result
    assert "demucs_requested" in result
    assert "demucs_used" in result
    assert "demucs_error" in result
    assert "ffmpeg_fallback_used" in result
    assert "warnings" in result


def test_demucs_failure_falls_back_to_ffmpeg():
    """Demucs crash should fall back to ffmpeg, not kill the job (Blocker 2)."""
    canvas = [0.0] * 100
    input_video = Path("/fake/video.mp4")

    # Track the bg_path so we can make is_file() return True after ffmpeg runs
    bg_path_holder = []

    def fake_run(cmd, **kwargs):
        # Simulate ffmpeg creating the output file
        if len(cmd) > 1 and cmd[-1].endswith(".wav"):
            Path(cmd[-1]).write_bytes(b"\x00" * 100)
        return MagicMock(returncode=0)

    with patch.object(build_audio, "local_ffmpeg", return_value="/fake/ffmpeg"), \
         patch.object(build_audio, "_demucs_vocal_separation", side_effect=RuntimeError("Demucs crashed")), \
         patch.object(build_audio, "load_audio_samples", return_value=([0.1] * 100, 44100)), \
         patch("subprocess.run", side_effect=fake_run):
        result = build_audio._mix_background_audio(
            canvas, input_video, 44100, 100, 0.15,
            vocal_separation=True,
            vocal_separation_method="demucs",
        )

    assert result["demucs_requested"] is True
    assert result["demucs_used"] is False
    assert "Demucs crashed" in result["demucs_error"]
    assert result["ffmpeg_fallback_used"] is True
    assert result["background_mix_applied"] is True
    assert len(result["warnings"]) >= 1
    assert result["warnings"][0]["stage"] == "demucs"


def test_demucs_success_no_fallback():
    """When Demucs succeeds, ffmpeg fallback should not be used (Blocker 2)."""
    canvas = [0.0] * 100
    input_video = Path("/fake/video.mp4")
    fake_no_vocals = Path("/fake/no_vocals.wav")

    with patch.object(build_audio, "local_ffmpeg", return_value="/fake/ffmpeg"), \
         patch.object(build_audio, "_demucs_vocal_separation", return_value=fake_no_vocals), \
         patch.object(build_audio, "load_audio_samples", return_value=([0.1] * 100, 44100)), \
         patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.is_file", return_value=True):
        mock_run.return_value = MagicMock(returncode=0)
        result = build_audio._mix_background_audio(
            canvas, input_video, 44100, 100, 0.15,
            vocal_separation=True,
            vocal_separation_method="demucs",
        )

    assert result["demucs_requested"] is True
    assert result["demucs_used"] is True
    assert result["ffmpeg_fallback_used"] is False
