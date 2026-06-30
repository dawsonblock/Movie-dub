"""Tests for scripts/review_loop.py — failed_segments, changes, shorten_text.

Covers: failed_segments.json merging from manifest + review + pitch
verification, speaker re-assignment, reference change, shorten_text edge
cases, and CLI-irrelevant pure functions.
"""

from __future__ import annotations

import json


from review_loop import (
    apply_reference_change,
    apply_speaker_change,
    shorten_text,
    write_failed_segments,
)


# ---------------------------------------------------------------------------
# shorten_text
# ---------------------------------------------------------------------------

class TestShortenText:
    def test_short_text_unchanged(self):
        assert shorten_text("hello world", 10.0) == "hello world"

    def test_empty_text(self):
        assert shorten_text("", 5.0) == ""

    def test_zero_duration(self):
        assert shorten_text("hello world", 0.0) == "hello world"

    def test_negative_duration(self):
        assert shorten_text("hello world", -1.0) == "hello world"

    def test_truncates_long_text(self):
        text = "one two three four five six seven eight nine ten"
        result = shorten_text(text, 1.0)  # ~2.7 words max
        words = result.split()
        assert len(words) <= 3

    def test_preserves_sentence_boundary(self):
        text = "First sentence. Second sentence. Third. Fourth. Fifth."
        result = shorten_text(text, 1.0)
        # Should end with a period if a sentence boundary is found in range
        assert result.rstrip().endswith(".") or result.rstrip().endswith("…")

    def test_custom_words_per_second(self):
        text = "a b c d e f g h i j"
        result = shorten_text(text, 1.0, words_per_second=5.0)
        # 5 words max + possible trailing "…" = 6 tokens
        tokens = result.split()
        real_words = [t for t in tokens if t != "…"]
        assert len(real_words) <= 5

    def test_exact_fit(self):
        """Text at exactly the word limit should be unchanged."""
        text = "one two three"
        # 3 words / 2.7 wps = 1.11s needed; at 2.0s we have room for 5.4 words
        assert shorten_text(text, 2.0) == "one two three"


# ---------------------------------------------------------------------------
# write_failed_segments
# ---------------------------------------------------------------------------

class TestWriteFailedSegments:
    def test_from_manifest_only(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "ok"},
        ]))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"results": [
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "error", "error": "bridge failed"},
            {"id": 3, "status": "skipped_empty_text"},
        ]}))
        out = tmp_path / "failed.json"
        result = write_failed_segments(review, manifest, None, out)
        ids = {f["segment_id"] for f in result["segments"]}
        assert ids == {"2", "3"}
        assert result["failed_count"] == 2

    def test_from_review_only(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "needs_review", "reason": "pitch mismatch"},
            {"id": 3, "status": "error", "reason": "tts fail"},
        ]))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"results": [
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "ok"},
            {"id": 3, "status": "ok"},
        ]}))
        out = tmp_path / "failed.json"
        result = write_failed_segments(review, manifest, None, out)
        ids = {f["segment_id"] for f in result["segments"]}
        assert ids == {"2", "3"}

    def test_from_pitch_verification(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([{"id": 1, "status": "ok"}]))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"results": [{"id": 1, "status": "ok"}]}))
        pv = tmp_path / "pv.json"
        pv.write_text(json.dumps({
            "segments": [
                {"segment_id": 1, "verdict": "ok"},
                {"segment_id": 2, "verdict": "needs_review", "flags": ["pitch_high"]},
                {"segment_id": 3, "verdict": "rewrite_shorter", "flags": ["too_long"]},
                {"segment_id": 4, "verdict": "missing"},
            ],
        }))
        out = tmp_path / "failed.json"
        result = write_failed_segments(review, manifest, pv, out)
        ids = {f["segment_id"] for f in result["segments"]}
        assert ids == {"2", "3", "4"}

    def test_merge_deduplicates(self, tmp_path):
        """A segment failing in both manifest and review should appear once."""
        review = tmp_path / "review.json"
        review.write_text(json.dumps([
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "needs_review", "reason": "pitch"},
        ]))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"results": [
            {"id": 1, "status": "ok"},
            {"id": 2, "status": "error", "error": "bridge"},
        ]}))
        out = tmp_path / "failed.json"
        result = write_failed_segments(review, manifest, None, out)
        seg2 = [f for f in result["segments"] if f["segment_id"] == "2"]
        assert len(seg2) == 1
        # Should have both review + manifest info
        assert seg2[0].get("review_status") == "needs_review"

    def test_no_failures(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([{"id": 1, "status": "ok"}]))
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"results": [{"id": 1, "status": "ok"}]}))
        out = tmp_path / "failed.json"
        result = write_failed_segments(review, manifest, None, out)
        assert result["failed_count"] == 0
        assert result["segments"] == []

    def test_missing_manifest(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([{"id": 1, "status": "error"}]))
        out = tmp_path / "failed.json"
        result = write_failed_segments(review, None, None, out)
        assert result["failed_count"] == 1

    def test_missing_review_file(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"results": [{"id": 1, "status": "error"}]}))
        out = tmp_path / "failed.json"
        result = write_failed_segments(tmp_path / "nope.json", manifest, None, out)
        assert result["failed_count"] == 1

    def test_output_file_written(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([{"id": 1, "status": "error"}]))
        out = tmp_path / "failed.json"
        write_failed_segments(review, None, None, out)
        assert out.is_file()
        data = json.loads(out.read_text())
        assert "schema_version" in data
        assert data["failed_count"] == 1


# ---------------------------------------------------------------------------
# apply_speaker_change
# ---------------------------------------------------------------------------

class TestApplySpeakerChange:
    def _setup(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([
            {"id": 1, "status": "ok", "speaker_id": "SPEAKER_00", "ref_wav": "/a.wav"},
            {"id": 2, "status": "ok", "speaker_id": "SPEAKER_01", "ref_wav": "/b.wav"},
            {"id": 3, "status": "ok", "speaker_id": "SPEAKER_00", "ref_wav": "/a.wav"},
        ]))
        profiles = tmp_path / "profiles.json"
        profiles.write_text(json.dumps({
            "speakers": [
                {"speaker_id": "SPEAKER_00", "reference_audio": "/a.wav"},
                {"speaker_id": "SPEAKER_01", "reference_audio": "/b.wav"},
                {"speaker_id": "SPEAKER_02", "reference_audio": "/c.wav"},
            ],
        }))
        return review, profiles

    def test_change_single_segment(self, tmp_path):
        review, profiles = self._setup(tmp_path)
        n = apply_speaker_change(review, profiles, segment_id="2", to_speaker_id="SPEAKER_02")
        assert n == 1
        data = json.loads(review.read_text())
        seg2 = [s for s in data if s["id"] == 2][0]
        assert seg2["speaker_id"] == "SPEAKER_02"
        assert seg2["ref_wav"] == "/c.wav"
        assert seg2["status"] == "pending_regen"

    def test_change_all_from_speaker(self, tmp_path):
        review, profiles = self._setup(tmp_path)
        n = apply_speaker_change(review, profiles, from_speaker_id="SPEAKER_00", to_speaker_id="SPEAKER_02")
        assert n == 2
        data = json.loads(review.read_text())
        for s in data:
            if s["speaker_id"] == "SPEAKER_02":
                assert s["status"] == "pending_regen"

    def test_unknown_target_speaker(self, tmp_path):
        review, profiles = self._setup(tmp_path)
        n = apply_speaker_change(review, profiles, segment_id="1", to_speaker_id="SPEAKER_99")
        assert n == 0

    def test_missing_review_file(self, tmp_path):
        profiles = tmp_path / "profiles.json"
        profiles.write_text(json.dumps({"speakers": []}))
        n = apply_speaker_change(tmp_path / "nope.json", profiles, segment_id="1", to_speaker_id="SPEAKER_00")
        assert n == 0


# ---------------------------------------------------------------------------
# apply_reference_change
# ---------------------------------------------------------------------------

class TestApplyReferenceChange:
    def test_change_reference(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([
            {"id": 1, "status": "ok", "ref_wav": "/old.wav"},
            {"id": 2, "status": "ok", "ref_wav": "/old.wav"},
        ]))
        n = apply_reference_change(review, "1", "/new.wav")
        assert n == 1
        data = json.loads(review.read_text())
        assert data[0]["ref_wav"] == "/new.wav"
        assert data[0]["status"] == "pending_regen"
        assert data[1]["ref_wav"] == "/old.wav"  # unchanged

    def test_unknown_segment(self, tmp_path):
        review = tmp_path / "review.json"
        review.write_text(json.dumps([{"id": 1, "status": "ok"}]))
        n = apply_reference_change(review, "99", "/new.wav")
        assert n == 0

    def test_missing_review_file(self, tmp_path):
        n = apply_reference_change(tmp_path / "nope.json", "1", "/new.wav")
        assert n == 0
