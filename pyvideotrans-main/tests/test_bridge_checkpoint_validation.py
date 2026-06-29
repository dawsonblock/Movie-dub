"""Tests for bridge checkpoint validation (Blocker 5)."""

import importlib.util
import sys
from pathlib import Path


BRIDGE_PATH = Path(__file__).resolve().parents[2] / "bridge" / "openvoice_segment_tts.py"
SPEC = importlib.util.spec_from_file_location("bridge", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


def test_bridge_rejects_lfs_pointer(tmp_path):
    """Bridge should reject Git LFS pointer checkpoint files (Blocker 5)."""
    converter_dir = tmp_path / "converter"
    converter_dir.mkdir()
    (converter_dir / "config.json").write_text("{}", encoding="utf-8")
    ckpt = converter_dir / "checkpoint.pth"
    ckpt.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:aaa\nsize 131320490\n",
        encoding="utf-8",
    )
    try:
        bridge.validate_checkpoint_dir(tmp_path.as_posix())
        assert False, "should have raised"
    except FileNotFoundError as exc:
        assert "Git LFS pointer" in str(exc)


def test_bridge_rejects_xet_pointer(tmp_path):
    """Bridge should reject Xet pointer checkpoint files (Blocker 5)."""
    converter_dir = tmp_path / "converter"
    converter_dir.mkdir()
    (converter_dir / "config.json").write_text("{}", encoding="utf-8")
    ckpt = converter_dir / "checkpoint.pth"
    ckpt.write_bytes(b"https://huggingface.co/xet/..." + b"\x00" * 200)
    try:
        bridge.validate_checkpoint_dir(tmp_path.as_posix())
        assert False, "should have raised"
    except FileNotFoundError as exc:
        assert "Xet pointer" in str(exc)


def test_bridge_rejects_tiny_checkpoint(tmp_path):
    """Bridge should reject suspiciously tiny checkpoint files (Blocker 5)."""
    converter_dir = tmp_path / "converter"
    converter_dir.mkdir()
    (converter_dir / "config.json").write_text("{}", encoding="utf-8")
    ckpt = converter_dir / "checkpoint.pth"
    ckpt.write_bytes(b"\x00" * 100)  # way too small
    try:
        bridge.validate_checkpoint_dir(tmp_path.as_posix())
        assert False, "should have raised"
    except FileNotFoundError as exc:
        assert "too small" in str(exc)


def test_bridge_rejects_missing_config(tmp_path):
    """Bridge should reject missing config.json (Blocker 5)."""
    converter_dir = tmp_path / "converter"
    converter_dir.mkdir()
    ckpt = converter_dir / "checkpoint.pth"
    ckpt.write_bytes(b"\x00" * 2000)
    try:
        bridge.validate_checkpoint_dir(tmp_path.as_posix())
        assert False, "should have raised"
    except FileNotFoundError as exc:
        assert "config" in str(exc).lower()


def test_bridge_rejects_invalid_json_config(tmp_path):
    """Bridge should reject invalid JSON config (Blocker 5)."""
    converter_dir = tmp_path / "converter"
    converter_dir.mkdir()
    (converter_dir / "config.json").write_text("not json {{{", encoding="utf-8")
    ckpt = converter_dir / "checkpoint.pth"
    ckpt.write_bytes(b"\x00" * 1_100_000)  # large enough to pass size check
    try:
        bridge.validate_checkpoint_dir(tmp_path.as_posix())
        assert False, "should have raised"
    except FileNotFoundError as exc:
        assert "not valid JSON" in str(exc)


def test_bridge_rejects_lfs_pointer_speaker_embedding(tmp_path):
    """Bridge should reject LFS pointer in speaker embeddings (Blocker 5)."""
    converter_dir = tmp_path / "converter"
    converter_dir.mkdir()
    (converter_dir / "config.json").write_text("{}", encoding="utf-8")
    ckpt = converter_dir / "checkpoint.pth"
    ckpt.write_bytes(b"\x00" * 1_100_000)  # large enough to pass size check
    ses_dir = tmp_path / "base_speakers" / "ses"
    ses_dir.mkdir(parents=True)
    (ses_dir / "en.pth").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:bbb\nsize 5000\n",
        encoding="utf-8",
    )
    try:
        bridge.validate_checkpoint_dir(tmp_path.as_posix())
        assert False, "should have raised"
    except FileNotFoundError as exc:
        assert "Git LFS pointer" in str(exc)


def test_bridge_accepts_valid_checkpoints(tmp_path):
    """Bridge should accept valid checkpoint files (Blocker 5)."""
    converter_dir = tmp_path / "converter"
    converter_dir.mkdir()
    (converter_dir / "config.json").write_text('{"model":"test"}', encoding="utf-8")
    ckpt = converter_dir / "checkpoint.pth"
    ckpt.write_bytes(b"\x00" * 1_100_000)  # large enough to pass size check
    ses_dir = tmp_path / "base_speakers" / "ses"
    ses_dir.mkdir(parents=True)
    (ses_dir / "en.pth").write_bytes(b"\x00" * 5000)
    result = bridge.validate_checkpoint_dir(tmp_path.as_posix())
    assert result == converter_dir
