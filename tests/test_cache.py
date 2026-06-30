"""Tests for scripts/cache.py — hash-based segment cache + stage resume.

Covers: hash determinism, cache key sensitivity to each input, SegmentCache
round-trip (put/get/save/load), hash tamper detection, ResumePlan in all
modes (resume / from_stage / only_segment / skip_existing), parse_from_stage
aliases + validation, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cache import (
    CACHE_SCHEMA_VERSION,
    PIPELINE_STAGES,
    ResumePlan,
    SegmentCache,
    content_hash,
    dict_hash,
    parse_from_stage,
    segment_cache_key,
    text_hash,
)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"hello world")
        assert content_hash(p) == content_hash(p)

    def test_different_content(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"hello")
        b.write_bytes(b"world")
        assert content_hash(a) != content_hash(b)

    def test_truncated_to_16(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"x" * 1000)
        h = content_hash(p)
        assert len(h) == 16

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        h = content_hash(p)
        assert len(h) == 16


class TestTextHash:
    def test_deterministic(self):
        assert text_hash("hello") == text_hash("hello")

    def test_different_text(self):
        assert text_hash("hello") != text_hash("world")

    def test_case_sensitive(self):
        assert text_hash("Hello") != text_hash("hello")

    def test_unicode(self):
        assert text_hash("你好") == text_hash("你好")
        assert len(text_hash("你好")) == 16


class TestDictHash:
    def test_deterministic(self):
        assert dict_hash({"a": 1, "b": 2}) == dict_hash({"b": 2, "a": 1})

    def test_order_independent(self):
        """Sorted keys mean dict ordering doesn't matter."""
        assert dict_hash({"a": 1, "b": 2}) == dict_hash({"b": 2, "a": 1})

    def test_different_values(self):
        assert dict_hash({"a": 1}) != dict_hash({"a": 2})


# ---------------------------------------------------------------------------
# segment_cache_key
# ---------------------------------------------------------------------------

class TestSegmentCacheKey:
    BASE = dict(
        source_audio_hash="aaa",
        subtitle_hash="bbb",
        speaker_profile_hash="ccc",
        translated_text="hello",
        tts_engine="openvoice",
        tts_model="v2",
        reference_audio_hash="ddd",
    )

    def test_deterministic(self):
        k1 = segment_cache_key(**self.BASE)
        k2 = segment_cache_key(**self.BASE)
        assert k1 == k2

    def test_length_24(self):
        assert len(segment_cache_key(**self.BASE)) == 24

    def test_text_change(self):
        base = dict(self.BASE)
        base["translated_text"] = "goodbye"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_engine_change(self):
        base = dict(self.BASE)
        base["tts_engine"] = "qwen3-local"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_model_change(self):
        base = dict(self.BASE)
        base["tts_model"] = "v3"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_reference_change(self):
        base = dict(self.BASE)
        base["reference_audio_hash"] = "eee"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_source_audio_change(self):
        base = dict(self.BASE)
        base["source_audio_hash"] = "fff"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_subtitle_hash_change(self):
        base = dict(self.BASE)
        base["subtitle_hash"] = "ggg"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_speaker_profile_change(self):
        base = dict(self.BASE)
        base["speaker_profile_hash"] = "hhh"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_language_change(self):
        base = dict(self.BASE)
        base["language"] = "zh"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)

    def test_voice_rate_change(self):
        base = dict(self.BASE)
        base["voice_rate"] = "+10%"
        assert segment_cache_key(**base) != segment_cache_key(**self.BASE)


# ---------------------------------------------------------------------------
# SegmentCache
# ---------------------------------------------------------------------------

class TestSegmentCache:
    def test_put_get_round_trip(self, tmp_path):
        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"audio data")
        key = "abc123"
        cache.put(key, wav)
        cache.save()
        # Reload from disk
        cache2 = SegmentCache(tmp_path / "cache")
        entry = cache2.get(key)
        assert entry is not None
        assert entry["output_wav"] == wav.as_posix()

    def test_get_missing_key(self, tmp_path):
        cache = SegmentCache(tmp_path / "cache")
        assert cache.get("nonexistent") is None

    def test_get_missing_wav(self, tmp_path):
        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"audio")
        cache.put("key1", wav)
        wav.unlink()  # delete the WAV
        assert cache.get("key1") is None

    def test_tamper_detection(self, tmp_path):
        """If the WAV content changes after caching, get() returns None."""
        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"original audio")
        cache.put("key1", wav)
        cache.save()
        # Tamper with the WAV
        wav.write_bytes(b"tampered audio!!!")
        cache2 = SegmentCache(tmp_path / "cache")
        assert cache2.get("key1") is None

    def test_save_creates_dirs(self, tmp_path):
        cache = SegmentCache(tmp_path / "nested" / "cache")
        cache.save()
        assert (tmp_path / "nested" / "cache").is_dir()
        assert (tmp_path / "nested" / "cache" / "segments").is_dir()
        assert (tmp_path / "nested" / "cache" / "segment_cache.json").is_file()

    def test_save_schema_version(self, tmp_path):
        cache = SegmentCache(tmp_path / "cache")
        cache.save()
        data = json.loads((tmp_path / "cache" / "segment_cache.json").read_text())
        assert data["schema_version"] == CACHE_SCHEMA_VERSION

    def test_segments_path_safe_id(self, tmp_path):
        cache = SegmentCache(tmp_path / "cache")
        p = cache.segments_path("key123", "seg 1/2")
        # Unsafe chars should be replaced with _
        assert " " not in p.name
        assert "/" not in p.name
        assert p.name.startswith("key123_")

    def test_stats(self, tmp_path):
        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"x")
        cache.put("k1", wav)
        s = cache.stats()
        assert s["entries"] == 1

    def test_load_corrupt_json(self, tmp_path):
        """Corrupt index file should not crash; returns empty index."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "segment_cache.json").write_text("{bad json")
        cache = SegmentCache(cache_dir)
        assert cache.get("any") is None


# ---------------------------------------------------------------------------
# ResumePlan
# ---------------------------------------------------------------------------

class TestResumePlan:
    def test_no_flags_runs_all(self):
        rp = ResumePlan({})
        for stage in PIPELINE_STAGES:
            assert rp.should_run_stage(stage) is True

    def test_resume_skips_passing(self):
        stages = {"transcription": "pass", "translation": "pass", "tts": "fail"}
        rp = ResumePlan(stages, resume=True)
        assert rp.should_run_stage("transcription") is False
        assert rp.should_run_stage("translation") is False
        assert rp.should_run_stage("tts") is True
        assert rp.should_run_stage("remux") is True

    def test_resume_reruns_non_pass(self):
        stages = {"transcription": "fail"}
        rp = ResumePlan(stages, resume=True)
        assert rp.should_run_stage("transcription") is True

    def test_resume_reruns_pending(self):
        stages = {"transcription": "pending"}
        rp = ResumePlan(stages, resume=True)
        assert rp.should_run_stage("transcription") is True

    def test_from_stage_tts(self):
        rp = ResumePlan({}, from_stage="tts")
        assert rp.should_run_stage("subtitle_extraction") is False
        assert rp.should_run_stage("transcription") is False
        assert rp.should_run_stage("translation") is False
        assert rp.should_run_stage("speaker_profiling") is False
        assert rp.should_run_stage("tts") is True
        assert rp.should_run_stage("audio_build") is True
        assert rp.should_run_stage("remux") is True
        assert rp.should_run_stage("verification") is True

    def test_from_stage_first_stage(self):
        rp = ResumePlan({}, from_stage="subtitle_extraction")
        for stage in PIPELINE_STAGES:
            assert rp.should_run_stage(stage) is True

    def test_from_stage_last_stage(self):
        rp = ResumePlan({}, from_stage="verification")
        for stage in PIPELINE_STAGES[:-1]:
            assert rp.should_run_stage(stage) is False
        assert rp.should_run_stage("verification") is True

    def test_only_segment_runs_tts_and_downstream(self):
        rp = ResumePlan({}, only_segment="42")
        assert rp.should_run_stage("transcription") is False
        assert rp.should_run_stage("tts") is True
        assert rp.should_run_stage("audio_build") is True
        assert rp.should_run_stage("remux") is True
        assert rp.should_run_stage("verification") is True

    def test_only_segment_skips_other_segments(self):
        rp = ResumePlan({}, only_segment="42")
        assert rp.segment_should_run("42", None) is True
        assert rp.segment_should_run("43", None) is False
        assert rp.segment_should_run(42, None) is True  # int vs str

    def test_skip_existing(self):
        import tempfile
        rp = ResumePlan({}, skip_existing=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            wav = Path(f.name)
            assert rp.segment_should_run("1", wav) is False
        assert rp.segment_should_run("2", None) is True

    def test_skip_existing_with_missing_wav(self):
        rp = ResumePlan({}, skip_existing=True)
        assert rp.segment_should_run("1", Path("/nonexistent.wav")) is True

    def test_stage_order(self):
        rp = ResumePlan({}, from_stage="tts")
        order = rp.stage_order()
        assert order == ["tts", "audio_build", "remux", "verification"]

    def test_summary(self):
        rp = ResumePlan({"tts": "pass"}, resume=True)
        s = rp.summary()
        assert s["resume"] is True
        assert "stages_to_run" in s
        assert "stages_to_skip" in s
        assert "tts" in s["stages_to_skip"]


# ---------------------------------------------------------------------------
# parse_from_stage
# ---------------------------------------------------------------------------

class TestParseFromStage:
    def test_empty_returns_none(self):
        assert parse_from_stage("") is None

    def test_canonical_names(self):
        for stage in PIPELINE_STAGES:
            assert parse_from_stage(stage) == stage

    @pytest.mark.parametrize("alias,canonical", [
        ("subs", "subtitle_extraction"),
        ("subtitles", "subtitle_extraction"),
        ("stt", "transcription"),
        ("asr", "transcription"),
        ("transcribe", "transcription"),
        ("sts", "translation"),
        ("translate", "translation"),
        ("profile", "speaker_profiling"),
        ("speakers", "speaker_profiling"),
        ("tts", "tts"),
        ("audio", "audio_build"),
        ("build", "audio_build"),
        ("remux", "remux"),
        ("mux", "remux"),
        ("verify", "verification"),
        ("verification", "verification"),
    ])
    def test_aliases(self, alias, canonical):
        assert parse_from_stage(alias) == canonical

    def test_case_insensitive(self):
        assert parse_from_stage("TTS") == "tts"
        assert parse_from_stage("Subs") == "subtitle_extraction"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="unknown stage"):
            parse_from_stage("bogus")

    def test_whitespace_stripped(self):
        assert parse_from_stage("  tts  ") == "tts"
