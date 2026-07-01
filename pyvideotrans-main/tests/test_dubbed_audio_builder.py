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
    manifest_path = tmp_path / "tts_manifest.json"
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
    manifest_path = tmp_path / "tts_manifest.json"
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
    manifest_path = tmp_path / "tts_manifest.json"
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
    manifest_path = tmp_path / "tts_manifest.json"
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
    manifest_path = tmp_path / "tts_manifest.json"
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
    manifest_path = tmp_path / "tts_manifest.json"
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
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    with pytest.raises(RuntimeError, match="status 'skipped'"):
        build_dubbed_audio_from_manifest.build_dubbed_audio(
            manifest_path, TEST_VIDEO, output_audio
        )


def _read_wav_peak(path: Path) -> float:
    """Read a WAV file and return the max absolute sample value."""
    with wave.open(path.as_posix(), "rb") as wav:
        n = wav.getnframes()
        sw = wav.getsampwidth()
        data = wav.readframes(n)
    if sw == 2:
        import array
        arr = array.array("h", data)
        return max(abs(v) for v in arr) / 32768.0
    return 0.0


def test_voice_volume_increases_peak(tmp_path):
    """--voice-volume > 1.0 should produce a louder output than 1.0."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    normal_out = tmp_path / "normal.wav"
    loud_out = tmp_path / "loud.wav"

    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, normal_out, voice_volume=1.0
    )
    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, loud_out, voice_volume=2.0
    )

    # Both are normalized to 0.97 peak, so we need to test with --no-normalize
    # to see the voice_volume effect
    normal_raw = tmp_path / "normal_raw.wav"
    loud_raw = tmp_path / "loud_raw.wav"
    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, normal_raw,
        voice_volume=1.0, no_normalize=True
    )
    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, loud_raw,
        voice_volume=2.0, no_normalize=True
    )

    normal_peak = _read_wav_peak(normal_raw)
    loud_peak = _read_wav_peak(loud_raw)
    assert loud_peak > normal_peak, f"loud ({loud_peak}) should > normal ({normal_peak})"


def test_no_normalize_keeps_lower_peak(tmp_path):
    """With --no-normalize, the peak should be lower than with normalization."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    norm_out = tmp_path / "normalized.wav"
    raw_out = tmp_path / "raw.wav"

    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, norm_out
    )
    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, raw_out, no_normalize=True
    )

    norm_peak = _read_wav_peak(norm_out)
    raw_peak = _read_wav_peak(raw_out)
    # Normalized output should be at target peak (~0.97)
    assert norm_peak > 0.9, f"normalized peak {norm_peak} should be near 0.97"
    # Raw output should be lower (not normalized to 0.97)
    assert raw_peak < norm_peak, f"raw ({raw_peak}) should < normalized ({norm_peak})"


def test_final_gain_affects_output_level(tmp_path):
    """--final-gain should scale the output level."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Use no_normalize so final_gain is the only gain control
    low_out = tmp_path / "low.wav"
    high_out = tmp_path / "high.wav"

    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, low_out,
        no_normalize=True, final_gain=0.5
    )
    build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, high_out,
        no_normalize=True, final_gain=1.0
    )

    low_peak = _read_wav_peak(low_out)
    high_peak = _read_wav_peak(high_out)
    assert high_peak > low_peak, f"gain=1.0 ({high_peak}) should > gain=0.5 ({low_peak})"


def test_report_includes_audio_quality_fields(tmp_path):
    """The build report should include voice_volume, final_gain, no_normalize."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        voice_volume=1.5, final_gain=0.8, no_normalize=True
    )

    assert report["voice_volume"] == 1.5
    assert report["final_gain"] == 0.8
    assert report["no_normalize"] is True


def test_report_includes_advanced_audio_fields(tmp_path):
    """The build report should include vocal_separation, ducking, lufs fields."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        vocal_separation=True, ducking=True, target_lufs=-16.0,
        background_volume=0.15,
    )

    assert report["vocal_separation"] is True
    assert report["ducking"] is True
    assert report["target_lufs"] == -16.0
    # lufs_applied may be True or False depending on ffmpeg, but key must exist
    assert "lufs_applied" in report


def test_vocal_separation_with_background(tmp_path):
    """Vocal separation + background volume should produce a valid WAV."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        background_volume=0.2, vocal_separation=True,
    )

    assert report["status"] == "ok"
    assert report["background_mix"] is True
    assert output_audio.is_file()
    # Duration should still match the video
    with wave.open(output_audio.as_posix(), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
    assert 7.0 < duration < 10.0


def test_ducking_with_background(tmp_path):
    """Ducking + background volume should produce a valid WAV."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "ok", "start": 2.0, "end": 3.0,
             "output_audio": (gen / "000002.wav").as_posix()},
        ],
    }
    # Write second segment WAV
    _write_tone_wav(gen / "000002.wav", 1.0)
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        background_volume=0.2, ducking=True,
    )

    assert report["status"] == "ok"
    assert report["background_mix"] is True
    assert report["ducking"] is True
    assert output_audio.is_file()


def test_lufs_normalization(tmp_path):
    """LUFS normalization should produce a valid WAV."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        target_lufs=-16.0,
    )

    assert report["status"] == "ok"
    assert report["target_lufs"] == -16.0
    # LUFS may or may not succeed depending on ffmpeg version, but output must exist
    assert output_audio.is_file()


def test_vocal_separation_ducking_lufs_combined(tmp_path):
    """All advanced features together should produce a valid WAV."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    _write_tone_wav(gen / "000002.wav", 0.8)
    manifest = {
        "ok": 2, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.5, "end": 1.5,
             "output_audio": (gen / "000001.wav").as_posix()},
            {"id": 2, "status": "ok", "start": 2.0, "end": 2.8,
             "output_audio": (gen / "000002.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        background_volume=0.15, vocal_separation=True,
        ducking=True, target_lufs=-16.0,
    )

    assert report["status"] == "ok"
    assert report["segments_placed"] == 2
    assert report["background_mix"] is True
    assert report["vocal_separation"] is True
    assert report["ducking"] is True
    assert output_audio.is_file()


def test_report_includes_warnings_list(tmp_path):
    """The build report should always include a warnings list (even if empty)."""
    gen = tmp_path / "generated_audio"
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio
    )

    assert "warnings" in report
    assert isinstance(report["warnings"], list)
    # No warnings for a simple build with no background/LUFS
    assert len(report["warnings"]) == 0


def test_build_report_includes_vocal_separation_method(tmp_path):
    """build report should include vocal_separation_method field."""
    gen = tmp_path / "generated_audio"
    gen.mkdir(parents=True)
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        vocal_separation=False,
        vocal_separation_method="demucs",
    )
    assert report["vocal_separation_method"] == "demucs"
    assert report["vocal_separation"] is False


def test_build_accepts_demucs_timeout_param(tmp_path):
    """build_dubbed_audio should accept demucs_timeout without error."""
    gen = tmp_path / "generated_audio"
    gen.mkdir(parents=True)
    _write_tone_wav(gen / "000001.wav", 1.0)
    manifest = {
        "ok": 1, "error": 0, "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    manifest_path = tmp_path / "tts_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_audio = tmp_path / "dubbed_audio.wav"

    report = build_dubbed_audio_from_manifest.build_dubbed_audio(
        manifest_path, TEST_VIDEO, output_audio,
        demucs_timeout=900,
    )
    assert report["status"] == "ok"
