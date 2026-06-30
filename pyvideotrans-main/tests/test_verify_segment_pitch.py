"""Tests for scripts/verify_segment_pitch.py — post-generation pitch verification."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_segment_pitch.py"

SPEC = importlib.util.spec_from_file_location("verify_segment_pitch", SCRIPT)
verify_segment_pitch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_segment_pitch)


def test_semitone_delta_zero():
    assert verify_segment_pitch.semitone_delta(200.0, 200.0) == 0.0


def test_semitone_delta_one_octave():
    # One octave up = +12 semitones
    assert verify_segment_pitch.semitone_delta(200.0, 400.0) == pytest.approx(12.0)


def test_semitone_delta_zero_inputs():
    assert verify_segment_pitch.semitone_delta(0.0, 200.0) == 0.0
    assert verify_segment_pitch.semitone_delta(200.0, 0.0) == 0.0


def test_classify_ok():
    gen = {"median_f0_hz": 200.0, "voiced_ratio": 0.75, "duration": 2.0}
    source = {"pitch": {"median_f0_hz": 205.0}}
    verdict, flags = verify_segment_pitch.classify(
        gen, source, 2.0, semitone_threshold=2.0,
        voiced_ratio_threshold=0.3, duration_ratio_threshold=1.35,
    )
    assert verdict == "ok"
    assert flags == []


def test_classify_pitch_mismatch():
    gen = {"median_f0_hz": 300.0, "voiced_ratio": 0.75, "duration": 2.0}
    source = {"pitch": {"median_f0_hz": 120.0}}
    verdict, flags = verify_segment_pitch.classify(
        gen, source, 2.0, semitone_threshold=2.0,
        voiced_ratio_threshold=0.3, duration_ratio_threshold=1.35,
    )
    assert verdict == "needs_review"
    assert any("pitch_mismatch" in f for f in flags)


def test_classify_low_voiced_ratio():
    gen = {"median_f0_hz": 200.0, "voiced_ratio": 0.1, "duration": 2.0}
    source = {"pitch": {"median_f0_hz": 200.0}}
    verdict, flags = verify_segment_pitch.classify(
        gen, source, 2.0, semitone_threshold=2.0,
        voiced_ratio_threshold=0.3, duration_ratio_threshold=1.35,
    )
    assert verdict == "needs_review"
    assert any("low_voiced_ratio" in f for f in flags)


def test_classify_rewrite_shorter():
    gen = {"median_f0_hz": 200.0, "voiced_ratio": 0.75, "duration": 3.0}
    source = {"pitch": {"median_f0_hz": 200.0}}
    verdict, flags = verify_segment_pitch.classify(
        gen, source, 2.0, semitone_threshold=2.0,
        voiced_ratio_threshold=0.3, duration_ratio_threshold=1.35,
    )
    assert verdict == "rewrite_shorter"
    assert any("duration_too_long" in f for f in flags)


def test_classify_missing_audio():
    verdict, flags = verify_segment_pitch.classify(
        None, None, 2.0, semitone_threshold=2.0,
        voiced_ratio_threshold=0.3, duration_ratio_threshold=1.35,
    )
    assert verdict == "missing"
    assert "generated_audio_unreadable" in flags


def test_classify_no_source_profile():
    """If no source profile, pitch check is skipped but duration still checked."""
    gen = {"median_f0_hz": 200.0, "voiced_ratio": 0.75, "duration": 2.0}
    verdict, flags = verify_segment_pitch.classify(
        gen, None, 2.0, semitone_threshold=2.0,
        voiced_ratio_threshold=0.3, duration_ratio_threshold=1.35,
    )
    assert verdict == "ok"


def test_load_profiles():
    import tempfile
    data = {
        "speakers": [
            {"speaker_id": "SPEAKER_00", "pitch": {"median_f0_hz": 120.0}},
            {"speaker_id": "SPEAKER_01", "pitch": {"median_f0_hz": 210.0}},
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f)
        path = Path(f.name)
    try:
        profiles = verify_segment_pitch.load_profiles(path)
        assert "SPEAKER_00" in profiles
        assert "SPEAKER_01" in profiles
        assert profiles["SPEAKER_00"]["pitch"]["median_f0_hz"] == 120.0
    finally:
        path.unlink()
