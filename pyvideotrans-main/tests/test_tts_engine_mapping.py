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


# ---------------------------------------------------------------------------
# Tests for parse_srt_to_segments (v0.9: split pipeline)
# ---------------------------------------------------------------------------

SAMPLE_SRT = """1
00:00:01,200 --> 00:00:04,800
Hello, how are you?

2
00:00:05,000 --> 00:00:08,500
I'm fine, thank you.

3
00:00:09,000 --> 00:00:12,000
Goodbye then.
"""


def test_parse_srt_to_segments_basic():
    """Verify SRT parsing extracts correct timings and text."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as f:
        f.write(SAMPLE_SRT)
        f.flush()
        srt_path = Path(f.name)
    try:
        segments = run_personal_dub.parse_srt_to_segments(srt_path)
        assert len(segments) == 3
        assert segments[0]["segment_index"] == 0
        assert segments[0]["start"] == 1.2
        assert segments[0]["end"] == 4.8
        assert "Hello" in segments[0]["text"]
        assert segments[1]["segment_index"] == 1
        assert segments[1]["start"] == 5.0
        assert segments[1]["end"] == 8.5
        assert segments[2]["segment_index"] == 2
        assert segments[2]["start"] == 9.0
        assert segments[2]["end"] == 12.0
    finally:
        srt_path.unlink(missing_ok=True)


def test_parse_srt_to_segments_empty():
    """Empty SRT returns no segments."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as f:
        f.write("")
        f.flush()
        srt_path = Path(f.name)
    try:
        segments = run_personal_dub.parse_srt_to_segments(srt_path)
        assert segments == []
    finally:
        srt_path.unlink(missing_ok=True)


def test_parse_srt_to_segments_bom():
    """SRT with UTF-8 BOM is handled correctly."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8-sig"
    ) as f:
        f.write(SAMPLE_SRT)
        f.flush()
        srt_path = Path(f.name)
    try:
        segments = run_personal_dub.parse_srt_to_segments(srt_path)
        assert len(segments) == 3
        assert segments[0]["start"] == 1.2
    finally:
        srt_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests for build_queue_tts_with_speakers (v0.9: per-speaker ref_wav routing)
# ---------------------------------------------------------------------------

SPEAKER_PROFILES = {
    "speakers": [
        {
            "speaker_id": "SPEAKER_00",
            "reference_audio": "",  # filled per-test
            "gender": {"label": "male", "confidence": 0.8},
            "pitch": {"median_f0_hz": 120.0},
        },
        {
            "speaker_id": "SPEAKER_01",
            "reference_audio": "",  # filled per-test
            "gender": {"label": "female", "confidence": 0.85},
            "pitch": {"median_f0_hz": 210.0},
        },
    ],
    "segment_assignments": [
        {"segment_index": 0, "speaker_id": "SPEAKER_00", "start": 1.2, "end": 4.8},
        {"segment_index": 1, "speaker_id": "SPEAKER_01", "start": 5.0, "end": 8.5},
        {"segment_index": 2, "speaker_id": "SPEAKER_00", "start": 9.0, "end": 12.0},
    ],
}


def test_build_queue_tts_with_speakers_routes_ref_wav(tmp_path):
    """Verify each queue item gets the correct per-speaker ref_wav."""
    # Create dummy reference WAVs
    ref0 = tmp_path / "SPEAKER_00" / "reference.wav"
    ref1 = tmp_path / "SPEAKER_01" / "reference.wav"
    ref0.parent.mkdir(parents=True)
    ref1.parent.mkdir(parents=True)
    ref0.write_bytes(b"RIFF\x00")
    ref1.write_bytes(b"RIFF\x00")

    profiles = json.loads(json.dumps(SPEAKER_PROFILES))
    profiles["speakers"][0]["reference_audio"] = ref0.as_posix()
    profiles["speakers"][1]["reference_audio"] = ref1.as_posix()

    # Write a translated SRT
    srt_path = tmp_path / "translated.srt"
    srt_path.write_text(SAMPLE_SRT, encoding="utf-8")

    cache_folder = tmp_path / "cache"
    cache_folder.mkdir()

    queue = run_personal_dub.build_queue_tts_with_speakers(
        translated_srt=srt_path,
        speaker_profiles=profiles,
        cache_folder=cache_folder,
        tts_type=34,
    )

    assert len(queue) == 3
    # Segment 0 -> SPEAKER_00
    assert queue[0]["ref_wav"] == ref0.as_posix()
    assert queue[0]["voice_reference"] == ref0.as_posix()
    assert queue[0]["role"] == "clone"
    assert queue[0]["line"] == 1
    assert queue[0]["start_time"] == 1200
    assert queue[0]["end_time"] == 4800
    # Segment 1 -> SPEAKER_01
    assert queue[1]["ref_wav"] == ref1.as_posix()
    assert queue[1]["voice_reference"] == ref1.as_posix()
    # Segment 2 -> SPEAKER_00 (back to speaker 0)
    assert queue[2]["ref_wav"] == ref0.as_posix()
    assert queue[2]["voice_reference"] == ref0.as_posix()


def test_build_queue_tts_fails_on_missing_ref_wav(tmp_path):
    """Queue building fails hard when a speaker's reference_audio is missing."""
    profiles = json.loads(json.dumps(SPEAKER_PROFILES))
    profiles["speakers"][0]["reference_audio"] = "/nonexistent/ref0.wav"
    profiles["speakers"][1]["reference_audio"] = "/nonexistent/ref1.wav"

    srt_path = tmp_path / "translated.srt"
    srt_path.write_text(SAMPLE_SRT, encoding="utf-8")

    with pytest.raises(RuntimeError, match="no reference_audio"):
        run_personal_dub.build_queue_tts_with_speakers(
            translated_srt=srt_path,
            speaker_profiles=profiles,
            cache_folder=tmp_path / "cache",
            tts_type=34,
        )


def test_build_queue_tts_fails_on_missing_segment_assignments(tmp_path):
    """Queue building with empty segment_assignments raises RuntimeError.

    With no segment_assignments, every segment gets speaker_id="" which
    maps to no profile, so ref_wav is "" -> the function raises.
    This is the v0.9 "no fake progress" guarantee.
    """
    profiles = {
        "speakers": SPEAKER_PROFILES["speakers"],
        "segment_assignments": [],
    }
    srt_path = tmp_path / "translated.srt"
    srt_path.write_text(SAMPLE_SRT, encoding="utf-8")

    with pytest.raises(RuntimeError, match="no reference_audio"):
        run_personal_dub.build_queue_tts_with_speakers(
            translated_srt=srt_path,
            speaker_profiles=profiles,
            cache_folder=tmp_path / "cache",
            tts_type=34,
        )


def test_build_queue_tts_empty_srt(tmp_path):
    """Empty translated SRT produces empty queue."""
    profiles = {"speakers": [], "segment_assignments": []}
    srt_path = tmp_path / "empty.srt"
    srt_path.write_text("", encoding="utf-8")
    # With empty SRT, parse_srt_to_segments returns [], so the loop
    # doesn't execute and we get an empty queue (no ref_wav check needed)
    queue = run_personal_dub.build_queue_tts_with_speakers(
        translated_srt=srt_path,
        speaker_profiles=profiles,
        cache_folder=tmp_path / "cache",
        tts_type=34,
    )
    assert queue == []


# ---------------------------------------------------------------------------
# Tests for split-pipeline engine guard (v0.10: no more fake TTS engine routing)
# ---------------------------------------------------------------------------

def _run_main_with_args(argv: list[str], tmp_path: Path) -> tuple[int, str]:
    """Invoke run_personal_dub.main() with a fake sys.argv.

    Returns (exit_code, stderr_text). We mock the input video to exist
    so argparse validation passes up to the engine guard.
    """
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"\x00" * 1024)
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF\x00")
    full_argv = [
        "run_personal_dub.py",
        "--input", input_video.as_posix(),
        "--reference", reference.as_posix(),
        "--output", (tmp_path / "output" / "dubbed.mp4").as_posix(),
        *argv,
    ]
    import io
    import unittest.mock as mock
    stderr_buf = io.StringIO()
    with mock.patch.object(sys, "argv", full_argv), \
         mock.patch.object(sys, "stderr", stderr_buf):
        rc = run_personal_dub.main()
    return rc, stderr_buf.getvalue()


def test_split_pipeline_rejects_qwen3_local(tmp_path):
    """--tts-engine qwen3-local + --speaker-profiling must fail.

    The split pipeline only supports OpenVoice for per-speaker ref_wav
    routing. Accepting qwen3-local would silently run OpenVoice anyway,
    which is a lie. This test proves the guard works.
    """
    rc, stderr = _run_main_with_args(
        ["--speaker-profiling", "--tts-engine", "qwen3-local"],
        tmp_path,
    )
    assert rc != 0, "qwen3-local should be rejected in split mode"
    assert "not supported" in stderr or "qwen3-local" in stderr


def test_split_pipeline_rejects_omnivoice(tmp_path):
    """--tts-engine omnivoice + --speaker-profiling must fail."""
    rc, stderr = _run_main_with_args(
        ["--speaker-profiling", "--tts-engine", "omnivoice"],
        tmp_path,
    )
    assert rc != 0, "omnivoice should be rejected in split mode"
    assert "not supported" in stderr or "omnivoice" in stderr


def test_split_pipeline_rejects_qwen3_fallback(tmp_path):
    """--fallback-tts-engine qwen3-local + --speaker-profiling must fail."""
    rc, stderr = _run_main_with_args(
        ["--speaker-profiling",
         "--tts-engine", "openvoice",
         "--fallback-tts-engine", "qwen3-local"],
        tmp_path,
    )
    assert rc != 0, "qwen3-local fallback should be rejected in split mode"
    assert "not supported" in stderr or "qwen3-local" in stderr


def test_split_pipeline_accepts_openvoice(tmp_path):
    """--tts-engine openvoice + --speaker-profiling must NOT be rejected
    by the engine guard. It will fail later (no checkpoints, etc.) but
    the guard message must not appear."""
    rc, stderr = _run_main_with_args(
        ["--speaker-profiling", "--tts-engine", "openvoice"],
        tmp_path,
    )
    # rc may be non-zero from a later step (missing venv, etc.), but the
    # engine guard message must NOT be in stderr.
    assert "not supported" not in stderr, (
        f"openvoice should not be rejected by engine guard, but stderr "
        f"says: {stderr}"
    )


def test_vtv_pipeline_accepts_qwen3_local(tmp_path):
    """--tts-engine qwen3-local WITHOUT --speaker-profiling must NOT
    be rejected by the engine guard. The VTV pipeline supports all engines."""
    rc, stderr = _run_main_with_args(
        ["--tts-engine", "qwen3-local"],
        tmp_path,
    )
    # rc may be non-zero from a later step, but the engine guard message
    # must NOT appear (we're not in split mode).
    assert "not supported" not in stderr, (
        f"qwen3-local should not be rejected in VTV mode, but stderr "
        f"says: {stderr}"
    )
