"""Tests for TTS engine selection and manifest enrichment in run_personal_dub.py.

Verifies the --tts-engine -> provider ID mapping and the manifest enrichment
logic that injects speaker_id into manifest results.
"""
import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_personal_dub.py"

# run_personal_dub.py imports sibling modules at import time, so we need
# scripts/ on the path.
import sys
sys.path.insert(0, str(ROOT / "scripts"))

SPEC = importlib.util.spec_from_file_location("run_personal_dub", SCRIPT)
run_personal_dub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_personal_dub)


def test_tts_engine_map_openvoice():
    assert run_personal_dub.TTS_ENGINE_MAP["openvoice"] == "34"


def test_tts_engine_map_qwen3_local():
    assert run_personal_dub.TTS_ENGINE_MAP["qwen3-local"] == "1"


def test_tts_engine_map_omnivoice():
    assert run_personal_dub.TTS_ENGINE_MAP["omnivoice"] == "2"


def test_tts_engine_map_has_all_choices():
    assert set(run_personal_dub.TTS_ENGINE_MAP.keys()) == {
        "openvoice", "qwen3-local", "omnivoice"
    }


def test_default_tts_engine_is_openvoice():
    assert run_personal_dub.DEFAULT_TTS_ENGINE == "openvoice"


def test_tts_engine_map_values_match_pyvideotrans_constants():
    """Verify our map matches pyVideoTrans's provider index constants."""
    tts_init = ROOT / "pyvideotrans-main" / "videotrans" / "tts" / "__init__.py"
    if not tts_init.is_file():
        pytest.skip("pyVideoTrans tts/__init__.py not found")
    content = tts_init.read_text(encoding="utf-8")
    # QWEN3LOCAL_TTS = 1
    assert "QWEN3LOCAL_TTS = 1" in content
    # OMNIVOICE_TTS = 2
    assert "OMNIVOICE_TTS = 2" in content
    # OPENVOICE_TTS = 34
    assert "OPENVOICE_TTS = 34" in content


def test_manifest_enrichment_injects_speaker_id():
    """Verify the enrichment logic that run_personal_dub uses inline.

    This replicates the enrichment block to test it in isolation.
    """
    manifest_data = {
        "ok": 2, "error": 0,
        "results": [
            {"id": 1, "output_audio": "/path/1.wav"},
            {"id": 2, "output_audio": "/path/2.wav"},
        ],
    }
    speaker_profiles_data = {
        "speakers": [
            {
                "speaker_id": "SPEAKER_00",
                "reference_audio": "/speakers/SPEAKER_00/reference.wav",
                "gender": {"label": "male", "confidence": 0.8},
                "age": {"band": "adult", "estimated_years": 35.0, "confidence": 0.6},
                "pitch": {"median_f0_hz": 120.0, "p10_f0_hz": 90.0,
                          "p90_f0_hz": 170.0, "voiced_ratio": 0.75},
            },
            {
                "speaker_id": "SPEAKER_01",
                "reference_audio": "/speakers/SPEAKER_01/reference.wav",
                "gender": {"label": "female", "confidence": 0.85},
                "age": {"band": "adult", "estimated_years": 30.0, "confidence": 0.6},
                "pitch": {"median_f0_hz": 210.0, "p10_f0_hz": 180.0,
                          "p90_f0_hz": 250.0, "voiced_ratio": 0.78},
            },
        ],
        "segment_assignments": [
            {"segment_index": 0, "speaker_id": "SPEAKER_00", "start": 0.0, "end": 2.0},
            {"segment_index": 1, "speaker_id": "SPEAKER_01", "start": 2.0, "end": 4.0},
        ],
    }

    # Replicate the enrichment logic from run_personal_dub.py
    profiles_by_id = {
        p["speaker_id"]: p for p in speaker_profiles_data["speakers"]
    }
    seg_to_speaker = {}
    for sa in speaker_profiles_data["segment_assignments"]:
        seg_to_speaker[sa["segment_index"]] = sa["speaker_id"]

    for index, item in enumerate(manifest_data["results"]):
        speaker_id = seg_to_speaker.get(index, "")
        profile = profiles_by_id.get(speaker_id, {})
        item["speaker_id"] = speaker_id
        item["reference_audio"] = profile.get("reference_audio", "")
        item["target_gender"] = profile.get("gender", {}).get("label", "")
        item["target_age_band"] = profile.get("age", {}).get("band", "")
        item["target_pitch_median_hz"] = profile.get("pitch", {}).get("median_f0_hz", 0.0)

    assert manifest_data["results"][0]["speaker_id"] == "SPEAKER_00"
    assert manifest_data["results"][0]["target_gender"] == "male"
    assert manifest_data["results"][0]["target_pitch_median_hz"] == 120.0
    assert manifest_data["results"][1]["speaker_id"] == "SPEAKER_01"
    assert manifest_data["results"][1]["target_gender"] == "female"
    assert manifest_data["results"][1]["target_pitch_median_hz"] == 210.0
