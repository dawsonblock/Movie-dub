"""Tests for the Qwen3-local TTS bridge.

Covers: bridge compiles, --help wiring, and manifest schema produced
from a minimal queue JSON.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "bridge" / "qwen3_segment_tts.py"
REFERENCE = ROOT / "voices" / "reference.wav"

SPEC = importlib.util.spec_from_file_location("qwen3_bridge", BRIDGE)
qwen3_bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qwen3_bridge)


class FakeResult:
    def __init__(self, audio):
        self.audio = audio


class FakeModel:
    def generate(self, *, text, ref_audio, ref_text=None, temperature=0.9,
                 speed=1.0, lang_code="auto"):
        # Yield a short synthetic 24 kHz mono clip (0.5 seconds)
        yield FakeResult(audio=np.zeros(12000, dtype=np.float32))


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
def test_bridge_writes_manifest_with_mocked_model(tmp_path):
    """The bridge must write a manifest with the expected schema fields.

    Model loading is mocked so the test does not require the real Qwen3-TTS
    weights; the manifest must still be produced and contain the expected
    fields.
    """
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

    args = qwen3_bridge.build_parser().parse_args([
        "--queue-tts-file", queue_path.as_posix(),
        "--manifest-file", manifest_path.as_posix(),
        "--work-dir", tmp_path.as_posix(),
        "--model-path", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "--language", "en",
    ])

    with mock.patch.object(qwen3_bridge, "load_qwen3_model", return_value=FakeModel()), \
         mock.patch.object(qwen3_bridge, "load_reference_audio", return_value=np.zeros(24000)), \
         mock.patch.object(qwen3_bridge, "write_generated_audio") as write_audio:
        def _write(path, audio, sr):
            Path(path).write_bytes(b"fake wav")
        write_audio.side_effect = _write
        rc = qwen3_bridge.synthesize_segments(args)

    assert manifest_path.is_file(), "manifest was not written"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "results" in data
    assert "ok" in data
    assert "error" in data
    assert "device" in data
    assert data["ok"] == 1, data
    assert data["error"] == 0, data
    assert data["results"][0]["status"] == "ok"
    assert rc == 0
