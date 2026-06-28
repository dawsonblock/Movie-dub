import importlib.util
import json
import struct
import wave
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "build_dubbed_audio_from_manifest.py"
TEST_VIDEO = ROOT / "assets" / "pyvideotrans_test_clip.mp4"

SPEC = importlib.util.spec_from_file_location("build_dubbed_audio_from_manifest", BUILDER_PATH)
build_dubbed_audio_from_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_dubbed_audio_from_manifest)


def _write_tone_wav(path: Path, duration: float, sr: int = 44100, freq: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration * sr)
    frames = b"".join(
        struct.pack("<h", int(8000 * ((i % sr) / sr * freq % 1)))
        for i in range(n)
    )
    with wave.open(path.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(frames)


def test_build_dubbed_audio_places_segments_on_canvas(tmp_path):
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.2)
    _write_tone_wav(gen / "000002.wav", 0.8)
    manifest = {
        "status": "ok",
        "ok": 2,
        "error": 0,
        "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.5, "end": 1.7, "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "ok", "start": 2.0, "end": 2.8, "output_audio": (gen / "000002.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "openvoice_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio
    )

    assert report["status"] == "ok"
    assert report["segments_placed"] == 2
    assert output_audio.is_file()
    with wave.open(output_audio.as_posix(), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    duration = frames / float(rate)
    # Output duration should roughly match the input video (~8.4s).
    assert 7.0 < duration < 10.0


def test_build_dubbed_audio_rejects_failed_segment_without_allow_partial(tmp_path):
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "status": "partial",
        "ok": 1,
        "error": 1,
        "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0, "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "error", "start": 1.5, "end": 2.5, "output_audio": (gen / "000002.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "openvoice_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    with pytest.raises(RuntimeError, match="status 'error'"):
        build_dubbed_audio_from_manifest.build_dubbed_audio(
            manifest_path, TEST_VIDEO, output_audio
        )


def test_build_dubbed_audio_allow_partial_skips_failed(tmp_path):
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "status": "partial",
        "ok": 1,
        "error": 1,
        "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0, "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "error", "start": 1.5, "end": 2.5, "output_audio": (gen / "000002.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "openvoice_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio, allow_partial=True
    )

    assert report["segments_placed"] == 1
    assert report["segments_failed"] == 1
    assert output_audio.is_file()


def test_build_dubbed_audio_supports_legacy_results_key(tmp_path):
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1,
        "error": 0,
        "results": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0, "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "openvoice_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio
    )

    assert report["segments_placed"] == 1
    assert output_audio.is_file()


def test_skipped_empty_text_fails_without_allow_partial(tmp_path):
    """skipped_empty_text segments must fail unless --allow-partial is used."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 1,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0, "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "skipped_empty_text", "start": 1.5, "end": 2.5, "output_audio": ""},
        ],
    }
    manifest_path = tmp_path / "openvoice_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    with pytest.raises(RuntimeError, match="skipped_empty_text"):
        build_dubbed_audio_from_manifest.build_dubbed_audio(
            manifest_path, TEST_VIDEO, output_audio
        )


def test_skipped_empty_text_passes_with_allow_partial(tmp_path):
    """With --allow-partial, skipped_empty_text segments are skipped silently."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 1,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0, "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "skipped_empty_text", "start": 1.5, "end": 2.5, "output_audio": ""},
        ],
    }
    manifest_path = tmp_path / "openvoice_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio, allow_partial=True
    )

    assert report["segments_placed"] == 1
    assert report["segments_skipped"] == 1
    assert output_audio.is_file()


def test_skipped_status_fails_without_allow_partial(tmp_path):
    """skipped status (not skipped_empty_text) must also fail without --allow-partial."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 1,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0, "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "skipped", "start": 1.5, "end": 2.5, "output_audio": ""},
        ],
    }
    manifest_path = tmp_path / "openvoice_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    with pytest.raises(RuntimeError, match="status 'skipped'"):
        build_dubbed_audio_from_manifest.build_dubbed_audio(
            manifest_path, TEST_VIDEO, output_audio
        )
