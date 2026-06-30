"""Tests for scripts/analyze_speakers.py — speaker profiling logic.

These tests mock pyannote/librosa and verify the pure-Python logic:
gender/age estimation, segment assignment, reference clip selection.
"""
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_speakers.py"

SPEC = importlib.util.spec_from_file_location("analyze_speakers", SCRIPT)
analyze_speakers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_speakers)


# --- Gender estimation ---

def test_estimate_gender_male():
    label, conf = analyze_speakers.estimate_gender(120.0, 0.75)
    assert label == "male"
    assert 0.5 < conf <= 0.95


def test_estimate_gender_female():
    label, conf = analyze_speakers.estimate_gender(210.0, 0.75)
    assert label == "female"
    assert 0.5 < conf <= 0.95


def test_estimate_gender_unknown_low_voiced_ratio():
    label, conf = analyze_speakers.estimate_gender(120.0, 0.1)
    assert label == "unknown"
    assert conf == 0.0


def test_estimate_gender_unknown_zero_f0():
    label, conf = analyze_speakers.estimate_gender(0.0, 0.8)
    assert label == "unknown"


def test_estimate_gender_confidence_increases_with_distance():
    _, conf_close = analyze_speakers.estimate_gender(166.0, 0.8)
    _, conf_far = analyze_speakers.estimate_gender(250.0, 0.8)
    assert conf_far > conf_close


# --- Age band estimation ---

def test_estimate_age_band_child():
    band, years, conf = analyze_speakers.estimate_age_band(280.0, 0.8, 200.0, 350.0)
    assert band == "child"
    assert years < 15


def test_estimate_age_band_teen_high_f0():
    band, years, conf = analyze_speakers.estimate_age_band(240.0, 0.8, 180.0, 300.0)
    assert band == "teen"


def test_estimate_age_band_teen_high_variance():
    band, years, conf = analyze_speakers.estimate_age_band(200.0, 0.8, 100.0, 250.0)
    assert band == "teen"


def test_estimate_age_band_adult():
    band, years, conf = analyze_speakers.estimate_age_band(120.0, 0.8, 90.0, 170.0)
    assert band == "adult"


def test_estimate_age_band_senior_low_voiced_ratio():
    band, years, conf = analyze_speakers.estimate_age_band(120.0, 0.4, 90.0, 170.0)
    assert band == "senior"
    assert years > 60


def test_estimate_age_band_unknown():
    band, years, conf = analyze_speakers.estimate_age_band(0.0, 0.1, 0.0, 0.0)
    assert band == "unknown"


# --- Segment assignment ---

def test_assign_segments_single_speaker_full_overlap():
    turns = [{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 10.0}]
    segments = [
        {"segment_index": 0, "start": 1.0, "end": 4.0},
        {"segment_index": 1, "start": 5.0, "end": 8.0},
    ]
    result = analyze_speakers.assign_segments_to_speakers(turns, segments)
    assert all(r["speaker_id"] == "SPEAKER_00" for r in result)


def test_assign_segments_multi_speaker_max_overlap():
    turns = [
        {"speaker_id": "SPEAKER_00", "start": 0.0, "end": 5.0},
        {"speaker_id": "SPEAKER_01", "start": 5.0, "end": 10.0},
    ]
    segments = [
        {"segment_index": 0, "start": 1.0, "end": 4.0},   # mostly SPEAKER_00
        {"segment_index": 1, "start": 6.0, "end": 9.0},   # mostly SPEAKER_01
    ]
    result = analyze_speakers.assign_segments_to_speakers(turns, segments)
    assert result[0]["speaker_id"] == "SPEAKER_00"
    assert result[1]["speaker_id"] == "SPEAKER_01"


def test_assign_segments_no_overlap_falls_back():
    turns = [{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 1.0}]
    segments = [{"segment_index": 0, "start": 10.0, "end": 15.0}]
    result = analyze_speakers.assign_segments_to_speakers(turns, segments)
    assert result[0]["speaker_id"] == "SPEAKER_00"  # fallback to first speaker


def test_assign_segments_empty_turns():
    turns = []
    segments = [{"segment_index": 0, "start": 1.0, "end": 4.0}]
    result = analyze_speakers.assign_segments_to_speakers(turns, segments)
    assert result[0]["speaker_id"] == "SPEAKER_00"


# --- Reference clip selection ---

def test_select_reference_clip_picks_longest_run():
    turns = [
        {"speaker_id": "SPEAKER_00", "start": 0.0, "end": 2.0},
        {"speaker_id": "SPEAKER_00", "start": 2.1, "end": 6.0},  # merged: 0-6
        {"speaker_id": "SPEAKER_01", "start": 10.0, "end": 20.0},
    ]
    clip = analyze_speakers.select_reference_clip(
        Path("/fake.wav"), turns, "SPEAKER_00", min_s=3.0, max_s=8.0
    )
    assert clip is not None
    start, end = clip
    assert end - start >= 3.0
    assert end - start <= 8.0


def test_select_reference_clip_no_turns():
    turns = []
    clip = analyze_speakers.select_reference_clip(
        Path("/fake.wav"), turns, "SPEAKER_00", min_s=3.0, max_s=8.0
    )
    assert clip is None


def test_select_reference_clip_short_turn_fallback():
    turns = [{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 1.5}]
    clip = analyze_speakers.select_reference_clip(
        Path("/fake.wav"), turns, "SPEAKER_00", min_s=3.0, max_s=8.0
    )
    # Below min_s but >= 1.0, should return as fallback
    assert clip is not None
    assert clip[1] - clip[0] == 1.5


# --- HF token resolution ---

def test_resolve_hf_token_explicit():
    assert analyze_speakers.resolve_hf_token("abc123") == "abc123"


def test_resolve_hf_token_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_token")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert analyze_speakers.resolve_hf_token(None) == "env_token"


def test_resolve_hf_token_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert analyze_speakers.resolve_hf_token(None) is None
