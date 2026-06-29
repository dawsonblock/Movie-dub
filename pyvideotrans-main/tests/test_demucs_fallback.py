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


def test_demucs_model_not_cached_refuses_download(tmp_path):
    """If Demucs model is not cached and allow_download=False, return None (Blocker 2)."""
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"\x00" * 100)
    output_dir = tmp_path / "demucs_out"
    output_dir.mkdir()

    # Mock ffmpeg to create the raw audio file
    def fake_run(cmd, **kwargs):
        if len(cmd) > 1 and cmd[-1].endswith(".wav"):
            Path(cmd[-1]).write_bytes(b"\x00" * 100)
        return MagicMock(returncode=0)

    with patch.object(build_audio, "local_ffmpeg", return_value="/fake/ffmpeg"), \
         patch("subprocess.run", side_effect=fake_run), \
         patch.dict("sys.modules", {
             "demucs": MagicMock(),
             "torch": MagicMock(),
             "torchaudio": MagicMock(),
             "demucs.pretrained": MagicMock(),
             "demucs.apply": MagicMock(),
         }):
        # Simulate get_model raising an error (model not cached)
        import sys as _sys
        demucs_pretrained = _sys.modules["demucs.pretrained"]
        demucs_pretrained.get_model = MagicMock(side_effect=RuntimeError("Model not cached"))
        result = build_audio._demucs_vocal_separation(
            input_video, output_dir, timeout=10, allow_download=False,
        )

    assert result is None, "should return None when model not cached and download refused"


def test_demucs_model_not_cached_allows_download_when_flag_set(tmp_path):
    """If allow_download=True, should attempt to load the model (Blocker 2)."""
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"\x00" * 100)
    output_dir = tmp_path / "demucs_out"
    output_dir.mkdir()

    def fake_run(cmd, **kwargs):
        if len(cmd) > 1 and cmd[-1].endswith(".wav"):
            Path(cmd[-1]).write_bytes(b"\x00" * 100)
        return MagicMock(returncode=0)

    with patch.object(build_audio, "local_ffmpeg", return_value="/fake/ffmpeg"), \
         patch("subprocess.run", side_effect=fake_run), \
         patch.dict("sys.modules", {
             "demucs": MagicMock(),
             "torch": MagicMock(),
             "torchaudio": MagicMock(),
             "demucs.pretrained": MagicMock(),
             "demucs.apply": MagicMock(),
         }):
        import sys as _sys
        demucs_pretrained = _sys.modules["demucs.pretrained"]
        # get_model raises — with allow_download=True it should re-raise
        demucs_pretrained.get_model = MagicMock(side_effect=RuntimeError("Download failed"))
        try:
            build_audio._demucs_vocal_separation(
                input_video, output_dir, timeout=10, allow_download=True,
            )
            assert False, "should have raised"
        except RuntimeError as exc:
            assert "Download failed" in str(exc)
