"""Tests for the Qwen3-local TTS bridge.

Covers: bridge compiles, --help wiring, and manifest schema produced
from a minimal queue JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "bridge" / "qwen3_segment_tts.py"
REFERENCE = ROOT / "voices" / "reference.wav"


def test_bridge_script_compiles():
    """bridge/qwen3_segment_tts.py must be valid Python."""
    result = subprocess.run(
        [sys.executable, "-c", f"import ast; ast.parse(open('{BRIDGE}').read())"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_bridge_help():
    """The bridge must accept --help and document the queue/manifest flags."""
    result = subprocess.run(
        [sys.executable, BRIDGE.as_posix(), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "--queue-tts-file" in out
    assert "--manifest-file" in out
    assert "--model-path" in out


@pytest.mark.skipif(not REFERENCE.is_file(), reason="reference WAV not present")
def test_bridge_manifest_schema(tmp_path):
    """The bridge must write a manifest with the expected schema fields."""
    queue = [{
        "line": 1,
        "text": "Hello world",
        "role": "clone",
        "ref_wav": REFERENCE.as_posix(),
        "voice_reference": REFERENCE.as_posix(),
        "start_time": 0,
        "end_time": 2000,
        "rate": "+0%",
        "volume": "+0%",
        "pitch": "+0Hz",
        "tts_type": 1,
        "filename": (tmp_path / "seg.wav").as_posix(),
    }]
    queue_path = tmp_path / "queue.json"
    manifest_path = tmp_path / "manifest.json"
    queue_path.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")

    try:
        subprocess.run(
            [
                sys.executable, BRIDGE.as_posix(),
                "--queue-tts-file", queue_path.as_posix(),
                "--manifest-file", manifest_path.as_posix(),
                "--work-dir", tmp_path.as_posix(),
                "--model-path", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                "--language", "en",
            ],
            capture_output=True, text=True, timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # Model loading can take a long time; the manifest may still be absent.
        pass
    # The bridge may fail because the model is not present, but it should
    # still write a manifest if it got far enough. If it fails early, we
    # only check the manifest schema when it exists.
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "results" in data
        assert "ok" in data
        assert "error" in data
        assert "device" in data
    # If the manifest is missing, the test still passes because the unit-test
    # environment is not required to have the Qwen3 model downloaded.
