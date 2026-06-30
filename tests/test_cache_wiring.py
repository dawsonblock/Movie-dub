"""Tests for the SegmentCache integration and character voice-lock routing
in run_personal_dub.py.

Covers:
  - ``_locked_voice_overrides``: extracting locked voice references from
    character_profiles.json.
  - ``build_queue_tts_with_speakers`` with ``character_profiles``: a locked
    voice overrides the speaker profile's reference_audio.
  - ``_compute_segment_cache_keys``: deterministic, input-sensitive cache
    keys for queue items.
  - ``_tts_model_id``: stable model identifier per engine.

These tests import only the helper functions (no subprocess, no model deps).
"""

from __future__ import annotations

from pathlib import Path

from cache import SegmentCache
from run_personal_dub import (
    _compute_segment_cache_keys,
    _locked_voice_overrides,
    _tts_model_id,
    build_queue_tts_with_speakers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_srt(path: Path, segments: list[tuple[int, float, float, str]]) -> None:
    """Write a minimal SRT file."""
    lines = []
    for i, (idx, start, end, text) in enumerate(segments, 1):
        def _fmt(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines.append(str(i))
        lines.append(f"{_fmt(start)} --> {_fmt(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _speaker_profiles(ref_a: str, ref_b: str) -> dict:
    return {
        "speakers": [
            {"speaker_id": "SPEAKER_00", "reference_audio": ref_a},
            {"speaker_id": "SPEAKER_01", "reference_audio": ref_b},
        ],
        "segment_assignments": [
            {"segment_index": 0, "speaker_id": "SPEAKER_00"},
            {"segment_index": 1, "speaker_id": "SPEAKER_01"},
        ],
    }


def _character_profiles_locked(speaker_id: str, ref: str) -> dict:
    return {
        "schema_version": "1.0",
        "characters": [
            {
                "character_id": "CHAR_001",
                "speaker_id": speaker_id,
                "name": "Test Character",
                "voice_reference": ref,
                "voice_locked": True,
            },
        ],
    }


# ---------------------------------------------------------------------------
# _locked_voice_overrides
# ---------------------------------------------------------------------------

class TestLockedVoiceOverrides:
    def test_no_profiles_returns_empty(self):
        assert _locked_voice_overrides(None) == {}

    def test_empty_profiles_returns_empty(self):
        assert _locked_voice_overrides({"characters": []}) == {}

    def test_unlocked_character_not_included(self):
        cp = {"characters": [
            {"speaker_id": "SPEAKER_00", "voice_locked": False,
             "voice_reference": "/a/ref.wav"},
        ]}
        assert _locked_voice_overrides(cp) == {}

    def test_locked_character_with_ref_included(self):
        cp = {"characters": [
            {"speaker_id": "SPEAKER_00", "voice_locked": True,
             "voice_reference": "/locked/ref.wav"},
        ]}
        assert _locked_voice_overrides(cp) == {"SPEAKER_00": "/locked/ref.wav"}

    def test_locked_character_no_ref_excluded(self):
        cp = {"characters": [
            {"speaker_id": "SPEAKER_00", "voice_locked": True,
             "voice_reference": ""},
        ]}
        assert _locked_voice_overrides(cp) == {}

    def test_multiple_locked_characters(self):
        cp = {"characters": [
            {"speaker_id": "SPEAKER_00", "voice_locked": True,
             "voice_reference": "/a.wav"},
            {"speaker_id": "SPEAKER_01", "voice_locked": True,
             "voice_reference": "/b.wav"},
            {"speaker_id": "SPEAKER_02", "voice_locked": False,
             "voice_reference": "/c.wav"},
        ]}
        overrides = _locked_voice_overrides(cp)
        assert overrides == {"SPEAKER_00": "/a.wav", "SPEAKER_01": "/b.wav"}
        assert "SPEAKER_02" not in overrides


# ---------------------------------------------------------------------------
# build_queue_tts_with_speakers + character_profiles voice-lock routing
# ---------------------------------------------------------------------------

class TestVoiceLockRouting:
    def test_no_character_profiles_uses_speaker_ref(self, tmp_path):
        ref_a = tmp_path / "ref_a.wav"
        ref_b = tmp_path / "ref_b.wav"
        ref_a.write_bytes(b"audio a")
        ref_b.write_bytes(b"audio b")
        srt = tmp_path / "translated.srt"
        _write_srt(srt, [(0, 0.0, 2.0, "Hello"), (1, 2.0, 4.0, "World")])
        queue = build_queue_tts_with_speakers(
            translated_srt=srt,
            speaker_profiles=_speaker_profiles(
                ref_a.as_posix(), ref_b.as_posix()),
            cache_folder=tmp_path / "cache",
            language="en",
        )
        assert queue[0]["ref_wav"] == ref_a.as_posix()
        assert queue[1]["ref_wav"] == ref_b.as_posix()

    def test_locked_voice_overrides_speaker_ref(self, tmp_path):
        ref_a = tmp_path / "ref_a.wav"
        ref_b = tmp_path / "ref_b.wav"
        locked_ref = tmp_path / "locked_ref.wav"
        ref_a.write_bytes(b"audio a")
        ref_b.write_bytes(b"audio b")
        locked_ref.write_bytes(b"locked audio")
        srt = tmp_path / "translated.srt"
        _write_srt(srt, [(0, 0.0, 2.0, "Hello"), (1, 2.0, 4.0, "World")])
        cp = _character_profiles_locked("SPEAKER_00", locked_ref.as_posix())
        queue = build_queue_tts_with_speakers(
            translated_srt=srt,
            speaker_profiles=_speaker_profiles(
                ref_a.as_posix(), ref_b.as_posix()),
            cache_folder=tmp_path / "cache",
            language="en",
            character_profiles=cp,
        )
        # SPEAKER_00's ref should be overridden by the locked voice
        assert queue[0]["ref_wav"] == locked_ref.as_posix()
        assert queue[0]["voice_reference"] == locked_ref.as_posix()
        # SPEAKER_01's ref should remain the speaker profile's reference
        assert queue[1]["ref_wav"] == ref_b.as_posix()

    def test_unlocked_character_does_not_override(self, tmp_path):
        ref_a = tmp_path / "ref_a.wav"
        ref_b = tmp_path / "ref_b.wav"
        unlocked_ref = tmp_path / "unlocked_ref.wav"
        ref_a.write_bytes(b"audio a")
        ref_b.write_bytes(b"audio b")
        unlocked_ref.write_bytes(b"unlocked audio")
        srt = tmp_path / "translated.srt"
        _write_srt(srt, [(0, 0.0, 2.0, "Hello"), (1, 2.0, 4.0, "World")])
        cp = {
            "characters": [
                {"character_id": "CHAR_001", "speaker_id": "SPEAKER_00",
                 "voice_reference": unlocked_ref.as_posix(),
                 "voice_locked": False},
            ],
        }
        queue = build_queue_tts_with_speakers(
            translated_srt=srt,
            speaker_profiles=_speaker_profiles(
                ref_a.as_posix(), ref_b.as_posix()),
            cache_folder=tmp_path / "cache",
            language="en",
            character_profiles=cp,
        )
        # SPEAKER_00's ref should NOT be overridden (voice not locked)
        assert queue[0]["ref_wav"] == ref_a.as_posix()


# ---------------------------------------------------------------------------
# _tts_model_id
# ---------------------------------------------------------------------------

class TestTtsModelId:
    def test_openvoice(self):
        args = type("A", (), {"qwen3_model": "some/model"})()
        assert _tts_model_id("openvoice", args) == "openvoice-v2"

    def test_qwen3_local(self):
        args = type("A", (), {"qwen3_model": "Qwen/Qwen3-TTS-1.7B"})()
        assert _tts_model_id("qwen3-local", args) == "Qwen/Qwen3-TTS-1.7B"


# ---------------------------------------------------------------------------
# _compute_segment_cache_keys
# ---------------------------------------------------------------------------

class TestComputeSegmentCacheKeys:
    def _base_queue(self) -> list[dict]:
        return [
            {"line": 1, "text": "Hello world", "ref_wav": "/a/ref.wav",
             "rate": "+0%", "volume": "+0%", "pitch": "+0Hz"},
            {"line": 2, "text": "Goodbye world", "ref_wav": "/b/ref.wav",
             "rate": "+0%", "volume": "+0%", "pitch": "+0Hz"},
        ]

    def _base_kwargs(self) -> dict:
        return dict(
            source_audio_hash="aaa",
            subtitle_hash="bbb",
            speaker_profile_hash="ccc",
            tts_engine="openvoice",
            tts_model="openvoice-v2",
            language="en",
        )

    def test_deterministic(self):
        q = self._base_queue()
        k1 = _compute_segment_cache_keys(q, **self._base_kwargs())
        k2 = _compute_segment_cache_keys(q, **self._base_kwargs())
        assert k1 == k2
        assert len(k1) == 2

    def test_different_text_different_key(self):
        kwargs = self._base_kwargs()
        q1 = self._base_queue()
        q2 = [{"line": 1, "text": "CHANGED", "ref_wav": "/a/ref.wav"}]
        k1 = _compute_segment_cache_keys(q1, **kwargs)
        k2 = _compute_segment_cache_keys(q2, **kwargs)
        assert k1["1"] != k2["1"]

    def test_different_engine_different_key(self):
        kwargs = self._base_kwargs()
        q = self._base_queue()
        k_ov = _compute_segment_cache_keys(q, **kwargs)
        kwargs2 = {**kwargs, "tts_engine": "qwen3-local", "tts_model": "qwen3"}
        k_qw = _compute_segment_cache_keys(q, **kwargs2)
        assert k_ov["1"] != k_qw["1"]

    def test_different_language_different_key(self):
        kwargs = self._base_kwargs()
        q = self._base_queue()
        k_en = _compute_segment_cache_keys(q, **kwargs)
        kwargs2 = {**kwargs, "language": "fr"}
        k_fr = _compute_segment_cache_keys(q, **kwargs2)
        assert k_en["1"] != k_fr["1"]

    def test_different_speaker_profile_different_key(self):
        kwargs = self._base_kwargs()
        q = self._base_queue()
        k1 = _compute_segment_cache_keys(q, **kwargs)
        kwargs2 = {**kwargs, "speaker_profile_hash": "ddd"}
        k2 = _compute_segment_cache_keys(q, **kwargs2)
        assert k1["1"] != k2["1"]

    def test_missing_ref_file_uses_missing_hash(self):
        q = [{"line": 1, "text": "Hello", "ref_wav": "/nonexistent/ref.wav"}]
        keys = _compute_segment_cache_keys(q, **self._base_kwargs())
        # Should still produce a key (deterministic with "missing" hash)
        assert "1" in keys
        assert len(keys["1"]) == 24

    def test_key_length_is_24(self):
        q = self._base_queue()
        keys = _compute_segment_cache_keys(q, **self._base_kwargs())
        for v in keys.values():
            assert len(v) == 24


# ---------------------------------------------------------------------------
# End-to-end cache hit/miss simulation
# ---------------------------------------------------------------------------

class TestCacheHitMissFlow:
    """Simulate the cache hit/miss flow that run_personal_dub.py uses."""

    def test_cache_hit_returns_entry(self, tmp_path):
        """A segment cached in a previous run is a hit on re-run."""
        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"generated audio")
        key = "testkey123"
        cache.put(key, wav)
        cache.save()
        # Reload (simulating a new run)
        cache2 = SegmentCache(tmp_path / "cache")
        entry = cache2.get(key)
        assert entry is not None
        assert entry["output_wav"] == wav.as_posix()

    def test_cache_miss_after_text_change(self, tmp_path):
        """If the translated text changes, the cache key changes → miss."""
        queue = [{"line": 1, "text": "Hello", "ref_wav": "/a.wav"}]
        kwargs = dict(
            source_audio_hash="aaa", subtitle_hash="bbb",
            speaker_profile_hash="ccc", tts_engine="openvoice",
            tts_model="openvoice-v2", language="en",
        )
        keys_original = _compute_segment_cache_keys(queue, **kwargs)
        queue_changed = [{"line": 1, "text": "Hello CHANGED", "ref_wav": "/a.wav"}]
        keys_changed = _compute_segment_cache_keys(queue_changed, **kwargs)
        assert keys_original["1"] != keys_changed["1"]

    def test_cache_record_and_reuse(self, tmp_path):
        """Record a generated WAV, then verify it's a hit on the next run."""
        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "generated.wav"
        wav.write_bytes(b"tts output")
        key = "abc123key"
        cache.put(key, wav)
        cache.save()
        # Simulate a re-run: same key → hit
        cache2 = SegmentCache(tmp_path / "cache")
        assert cache2.get(key) is not None

    def test_tampered_wav_is_cache_miss(self, tmp_path):
        """If the cached WAV is modified, get() returns None (tamper detection)."""
        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "seg.wav"
        wav.write_bytes(b"original")
        cache.put("key1", wav)
        cache.save()
        # Tamper
        wav.write_bytes(b"tampered!!!")
        cache2 = SegmentCache(tmp_path / "cache")
        assert cache2.get("key1") is None
