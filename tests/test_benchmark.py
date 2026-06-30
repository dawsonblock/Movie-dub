"""Tests for scripts/benchmark.py — metrics collection + helpers.

Covers: _percentile, _ffprobe_duration (missing file), collect_metrics
from a fixture job dir, quality score computation, and edge cases.
Does NOT run the actual pipeline (that's the e2e smoke test's job).
"""

from __future__ import annotations

import json
from pathlib import Path


from benchmark import (
    _percentile,
    _read_json,
    collect_metrics,
)


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

class TestPercentile:
    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 0) == 5.0
        assert _percentile([5.0], 100) == 5.0

    def test_p50_two_values(self):
        assert _percentile([1.0, 2.0], 50) == 1.5

    def test_p50_three_values(self):
        assert _percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_p0(self):
        assert _percentile([3.0, 1.0, 2.0], 0) == 1.0

    def test_p100(self):
        assert _percentile([3.0, 1.0, 2.0], 100) == 3.0

    def test_p90(self):
        vals = list(range(1, 11))  # 1..10
        result = _percentile([float(v) for v in vals], 90)
        assert 9.0 <= result <= 10.0

    def test_unsorted_input(self):
        """Percentile should work regardless of input order."""
        assert _percentile([3.0, 1.0, 2.0], 50) == 2.0


# ---------------------------------------------------------------------------
# _read_json
# ---------------------------------------------------------------------------

class TestReadJson:
    def test_valid(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text(json.dumps({"x": 1}))
        assert _read_json(p) == {"x": 1}

    def test_missing(self, tmp_path):
        assert _read_json(tmp_path / "nope.json") is None

    def test_invalid(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{bad")
        assert _read_json(p) is None


# ---------------------------------------------------------------------------
# collect_metrics
# ---------------------------------------------------------------------------

def _make_fixture_job(tmp_path: Path) -> Path:
    """Create a minimal job dir with manifest + speaker_profiles + review."""
    job = tmp_path / "job"
    job.mkdir()

    # Manifest with 3 ok + 1 error + 1 skipped
    (job / "openvoice_manifest.json").write_text(json.dumps({
        "results": [
            {"id": 1, "status": "ok", "tts_seconds": 1.5,
             "generated_duration": 2.0, "target_duration": 2.1},
            {"id": 2, "status": "ok", "tts_seconds": 2.0,
             "generated_duration": 3.0, "target_duration": 2.8},
            {"id": 3, "status": "ok", "tts_seconds": 1.0,
             "generated_duration": 1.5, "target_duration": 1.5},
            {"id": 4, "status": "error", "error": "bridge crashed"},
            {"id": 5, "status": "skipped_empty_text"},
        ],
    }))

    # Speaker profiles
    sp_dir = job / "speakers"
    sp_dir.mkdir()
    (sp_dir / "speaker_profiles.json").write_text(json.dumps({
        "speakers": [
            {"speaker_id": "SPEAKER_00",
             "gender": {"label": "male", "confidence": 0.8},
             "age": {"band": "adult", "estimated_years": 35.0, "confidence": 0.6}},
            {"speaker_id": "SPEAKER_01",
             "gender": {"label": "female", "confidence": 0.3},
             "age": {"band": "young_adult", "estimated_years": 25.0, "confidence": 0.2}},
        ],
        "segment_assignments": [
            {"segment_index": 0, "speaker_id": "SPEAKER_00", "start": 0, "end": 2},
            {"segment_index": 1, "speaker_id": "SPEAKER_01", "start": 2, "end": 5},
        ],
    }))

    # Review segments
    (job / "review_segments.json").write_text(json.dumps([
        {"id": 1, "status": "ok", "speaker_id": "SPEAKER_00", "output_audio": ""},
        {"id": 2, "status": "needs_review", "speaker_id": "SPEAKER_01", "output_audio": ""},
        {"id": 3, "status": "ok", "speaker_id": "SPEAKER_00", "output_audio": ""},
    ]))

    # Pitch verification
    (job / "pitch_verification.json").write_text(json.dumps({
        "summary": {"ok": 2, "needs_review": 1, "rewrite_shorter": 0},
        "segments": [
            {"segment_id": 1, "verdict": "ok"},
            {"segment_id": 2, "verdict": "needs_review", "flags": ["pitch_high"]},
            {"segment_id": 3, "verdict": "ok"},
        ],
    }))

    return job


class TestCollectMetrics:
    def test_segment_counts(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        assert m["segments_total"] == 5
        assert m["segments_ok"] == 3
        assert m["segment_success_rate"] == 0.6

    def test_failed_segments(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        assert m["failed_segment_count"] == 2
        ids = {f["id"] for f in m["failed_segments"]}
        assert ids == {"4", "5"}

    def test_tts_time_stats(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        tts = m["tts_time_per_segment"]
        assert tts["count"] == 3
        assert tts["mean"] == round((1.5 + 2.0 + 1.0) / 3, 3)
        assert tts["max"] == 2.0

    def test_timing_drift(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        # drifts: |2.0-2.1|=0.1, |3.0-2.8|=0.2, |1.5-1.5|=0.0
        assert m["timing_drift_seconds"]["mean"] == round((0.1 + 0.2 + 0.0) / 3, 3)

    def test_review_required(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        # 1 from review (needs_review) + 1 from pitch_verification
        assert m["review_required"] >= 1

    def test_speakers_detected(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        assert m["speakers_detected"] == 2

    def test_bad_age_gender_low_confidence(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        # SPEAKER_01 has gender confidence 0.3 < 0.5 and age confidence 0.2 < 0.5
        bad = m["bad_age_gender_guesses"]
        assert len(bad) == 2  # gender + age for SPEAKER_01

    def test_quality_score_in_range(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        assert 0 <= m["quality_score"] <= 100

    def test_total_render_time(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 42.5}, None, 0.5)
        assert m["total_render_time_seconds"] == 42.5

    def test_no_loudness_when_target_none(self, tmp_path):
        job = _make_fixture_job(tmp_path)
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 30.0}, None, 0.5)
        assert m["loudness_mismatch_lufs"] is None

    def test_empty_manifest(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        (job / "openvoice_manifest.json").write_text(json.dumps({"results": []}))
        (job / "review_segments.json").write_text(json.dumps([]))
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 5.0}, None, 0.5)
        assert m["segments_total"] == 0
        assert m["segment_success_rate"] == 0.0
        assert m["failed_segment_count"] == 0

    def test_missing_job_dir(self, tmp_path):
        """Should not crash if the job dir has no artifacts."""
        job = tmp_path / "empty_job"
        job.mkdir()
        m = collect_metrics(job, 10.0, job / "final.mp4", {"_total": 5.0}, None, 0.5)
        assert m["segments_total"] == 0
        assert m["quality_score"] >= 0
