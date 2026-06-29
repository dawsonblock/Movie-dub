"""Tests for LUFS normalization sample-rate preservation (Blocker 6)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


BUILD_AUDIO_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_dubbed_audio_from_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_audio", BUILD_AUDIO_PATH)
build_audio = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_audio
SPEC.loader.exec_module(build_audio)


def test_lufs_uses_selected_sample_rate(tmp_path):
    """_apply_lufs_normalization should use the passed sample_rate, not default (Blocker 6)."""
    input_wav = tmp_path / "input.wav"
    input_wav.write_bytes(b"\x00" * 100)

    captured_cmd = []

    def fake_run(cmd, **kwargs):
        captured_cmd.append(cmd)
        # Simulate ffmpeg writing the output file
        # The cmd[-1] is the output path
        Path(cmd[-1]).write_bytes(b"\x00" * 100)
        return MagicMock(returncode=0)

    with patch.object(build_audio, "local_ffmpeg", return_value="/fake/ffmpeg"), \
         patch("subprocess.run", side_effect=fake_run), \
         patch("shutil.move"):
        result = build_audio._apply_lufs_normalization(
            input_wav, target_lufs=-16.0, timeout=10, sample_rate=48000,
        )

    assert result is True
    # Check that the ffmpeg command used 48000, not 44100
    cmd = captured_cmd[0]
    ar_idx = cmd.index("-ar")
    assert cmd[ar_idx + 1] == "48000", f"expected 48000, got {cmd[ar_idx + 1]}"


def test_lufs_defaults_to_44100():
    """_apply_lufs_normalization default sample_rate should still be 44100 (Blocker 6)."""
    import inspect
    sig = inspect.signature(build_audio._apply_lufs_normalization)
    assert sig.parameters["sample_rate"].default == build_audio.DEFAULT_SAMPLE_RATE
