"""Tests for scripts/character_profiles.py — build, idempotency, mutations.

Covers: schema validation, default name generation, idempotent re-runs
(preserving names/locks/review_status), rename/lock/unlock/set-review
mutations, stable character_ids, segment membership, sorting, and
edge cases (empty speakers, missing files, unknown character_id).
"""

from __future__ import annotations


import pytest

from character_profiles import (
    SCHEMA_VERSION,
    _default_name,
    build_character_profiles,
    load_character_profiles,
    lock_voice,
    rename_character,
    set_review_status,
    write_character_profiles,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _speaker_profiles_two() -> dict:
    """Two speakers with segment assignments."""
    return {
        "speakers": [
            {
                "speaker_id": "SPEAKER_00",
                "reference_audio": "/a/ref.wav",
                "total_speech_seconds": 10.5,
                "gender": {"label": "male", "confidence": 0.8},
                "age": {"band": "adult", "estimated_years": 35.0, "source": "heuristic"},
                "pitch": {"median_f0_hz": 120.0},
            },
            {
                "speaker_id": "SPEAKER_01",
                "reference_audio": "/b/ref.wav",
                "total_speech_seconds": 20.0,
                "gender": {"label": "female", "confidence": 0.9},
                "age": {"band": "young_adult", "estimated_years": 28.0, "source": "model"},
                "pitch": {"median_f0_hz": 210.0},
            },
        ],
        "segment_assignments": [
            {"segment_index": 0, "speaker_id": "SPEAKER_00", "start": 0, "end": 2},
            {"segment_index": 1, "speaker_id": "SPEAKER_01", "start": 2, "end": 5},
            {"segment_index": 2, "speaker_id": "SPEAKER_00", "start": 5, "end": 7},
            {"segment_index": 3, "speaker_id": "SPEAKER_01", "start": 7, "end": 10},
        ],
    }


def _speaker_profiles_empty() -> dict:
    return {"speakers": [], "segment_assignments": []}


# ---------------------------------------------------------------------------
# _default_name
# ---------------------------------------------------------------------------

class TestDefaultName:
    def test_male_adult(self):
        assert _default_name("male", "adult", 0, set()) == "Man 1"

    def test_female_adult(self):
        assert _default_name("female", "adult", 0, set()) == "Woman 1"

    def test_child_male(self):
        assert _default_name("male", "child", 0, set()) == "Child Man 1"

    def test_teen_female(self):
        assert _default_name("female", "teen", 0, set()) == "Teen Woman 1"

    def test_senior_male(self):
        assert _default_name("male", "senior", 0, set()) == "Older Man 1"

    def test_unknown_gender(self):
        assert _default_name("unknown", "adult", 0, set()) == "Speaker 1"

    def test_disambiguation(self):
        assert _default_name("male", "adult", 0, {"Man 1"}) == "Man 2"
        assert _default_name("male", "adult", 0, {"Man 1", "Man 2"}) == "Man 3"


# ---------------------------------------------------------------------------
# build_character_profiles
# ---------------------------------------------------------------------------

class TestBuildCharacterProfiles:
    def test_basic_two_speakers(self):
        cp = build_character_profiles(_speaker_profiles_two(), tts_engine="qwen3-local")
        assert cp["schema_version"] == SCHEMA_VERSION
        assert cp["tts_engine"] == "qwen3-local"
        assert len(cp["characters"]) == 2

    def test_segment_membership(self):
        cp = build_character_profiles(_speaker_profiles_two())
        by_sid = {c["speaker_id"]: c for c in cp["characters"]}
        assert by_sid["SPEAKER_00"]["segments"] == [0, 2]
        assert by_sid["SPEAKER_01"]["segments"] == [1, 3]
        assert by_sid["SPEAKER_00"]["segment_count"] == 2
        assert by_sid["SPEAKER_01"]["segment_count"] == 2

    def test_character_ids_are_stable_and_sequential(self):
        cp = build_character_profiles(_speaker_profiles_two())
        ids = sorted(c["character_id"] for c in cp["characters"])
        assert ids == ["CHAR_001", "CHAR_002"]

    def test_gender_and_age_propagated(self):
        cp = build_character_profiles(_speaker_profiles_two())
        by_sid = {c["speaker_id"]: c for c in cp["characters"]}
        assert by_sid["SPEAKER_00"]["gender"] == "male"
        assert by_sid["SPEAKER_00"]["age_band"] == "adult"
        assert by_sid["SPEAKER_01"]["gender"] == "female"
        assert by_sid["SPEAKER_01"]["age_band"] == "young_adult"
        assert by_sid["SPEAKER_01"]["estimated_years"] == 28.0

    def test_f0_propagated(self):
        cp = build_character_profiles(_speaker_profiles_two())
        by_sid = {c["speaker_id"]: c for c in cp["characters"]}
        assert by_sid["SPEAKER_00"]["median_f0_hz"] == 120.0
        assert by_sid["SPEAKER_01"]["median_f0_hz"] == 210.0

    def test_default_review_status_pending(self):
        cp = build_character_profiles(_speaker_profiles_two())
        for ch in cp["characters"]:
            assert ch["review_status"] == "pending"

    def test_default_voice_not_locked(self):
        cp = build_character_profiles(_speaker_profiles_two())
        for ch in cp["characters"]:
            assert ch["voice_locked"] is False

    def test_voice_reference_follows_speaker_profile(self):
        cp = build_character_profiles(_speaker_profiles_two())
        by_sid = {c["speaker_id"]: c for c in cp["characters"]}
        assert by_sid["SPEAKER_00"]["voice_reference"] == "/a/ref.wav"
        assert by_sid["SPEAKER_01"]["voice_reference"] == "/b/ref.wav"

    def test_sorted_by_first_segment(self):
        cp = build_character_profiles(_speaker_profiles_two())
        # SPEAKER_00 has segment 0, SPEAKER_01 has segment 1
        assert cp["characters"][0]["speaker_id"] == "SPEAKER_00"
        assert cp["characters"][1]["speaker_id"] == "SPEAKER_01"

    def test_empty_speakers(self):
        cp = build_character_profiles(_speaker_profiles_empty())
        assert cp["characters"] == []
        assert cp["schema_version"] == SCHEMA_VERSION

    def test_speaker_with_no_segments(self):
        sp = {
            "speakers": [{
                "speaker_id": "SPEAKER_00",
                "reference_audio": "/a.wav",
                "gender": {"label": "male"},
                "age": {"band": "adult"},
                "pitch": {"median_f0_hz": 100},
            }],
            "segment_assignments": [],
        }
        cp = build_character_profiles(sp)
        assert cp["characters"][0]["segments"] == []
        assert cp["characters"][0]["segment_count"] == 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_preserves_user_name(self):
        sp = _speaker_profiles_two()
        cp1 = build_character_profiles(sp, tts_engine="qwen3-local")
        # Simulate a user rename
        cp1["characters"][0]["name"] = "Alice"
        cp2 = build_character_profiles(sp, tts_engine="qwen3-local", existing=cp1)
        by_sid = {c["speaker_id"]: c for c in cp2["characters"]}
        assert by_sid["SPEAKER_00"]["name"] == "Alice"

    def test_preserves_voice_lock(self):
        sp = _speaker_profiles_two()
        cp1 = build_character_profiles(sp)
        cp1["characters"][0]["voice_locked"] = True
        cp1["characters"][0]["voice_reference"] = "/locked/alice.wav"
        cp2 = build_character_profiles(sp, existing=cp1)
        by_sid = {c["speaker_id"]: c for c in cp2["characters"]}
        assert by_sid["SPEAKER_00"]["voice_locked"] is True
        assert by_sid["SPEAKER_00"]["voice_reference"] == "/locked/alice.wav"

    def test_preserves_review_status(self):
        sp = _speaker_profiles_two()
        cp1 = build_character_profiles(sp)
        cp1["characters"][0]["review_status"] = "approved"
        cp2 = build_character_profiles(sp, existing=cp1)
        by_sid = {c["speaker_id"]: c for c in cp2["characters"]}
        assert by_sid["SPEAKER_00"]["review_status"] == "approved"

    def test_preserves_character_id(self):
        sp = _speaker_profiles_two()
        cp1 = build_character_profiles(sp)
        original_id = cp1["characters"][0]["character_id"]
        cp2 = build_character_profiles(sp, existing=cp1)
        assert cp2["characters"][0]["character_id"] == original_id

    def test_unlocked_voice_follows_new_reference(self):
        """When not locked, voice_reference should follow the speaker profile."""
        sp = _speaker_profiles_two()
        cp1 = build_character_profiles(sp)
        # Change the speaker profile's reference_audio
        sp["speakers"][0]["reference_audio"] = "/new/ref.wav"
        cp2 = build_character_profiles(sp, existing=cp1)
        by_sid = {c["speaker_id"]: c for c in cp2["characters"]}
        assert by_sid["SPEAKER_00"]["voice_reference"] == "/new/ref.wav"

    def test_locked_voice_does_not_follow_new_reference(self):
        sp = _speaker_profiles_two()
        cp1 = build_character_profiles(sp)
        cp1["characters"][0]["voice_locked"] = True
        cp1["characters"][0]["voice_reference"] = "/locked/ref.wav"
        sp["speakers"][0]["reference_audio"] = "/new/ref.wav"
        cp2 = build_character_profiles(sp, existing=cp1)
        by_sid = {c["speaker_id"]: c for c in cp2["characters"]}
        assert by_sid["SPEAKER_00"]["voice_reference"] == "/locked/ref.wav"

    def test_removed_speaker_dropped(self):
        sp = _speaker_profiles_two()
        cp1 = build_character_profiles(sp)
        # Remove SPEAKER_01 from the speaker profiles
        sp["speakers"] = [sp["speakers"][0]]
        cp2 = build_character_profiles(sp, existing=cp1)
        sids = {c["speaker_id"] for c in cp2["characters"]}
        assert sids == {"SPEAKER_00"}


# ---------------------------------------------------------------------------
# write / load round-trip
# ---------------------------------------------------------------------------

class TestWriteLoad:
    def test_round_trip(self, tmp_path):
        sp = _speaker_profiles_two()
        cp = build_character_profiles(sp, tts_engine="openvoice")
        path = tmp_path / "cp.json"
        write_character_profiles(path, cp, source="/sp.json")
        assert path.is_file()
        loaded = load_character_profiles(path)
        assert loaded is not None
        assert loaded["schema_version"] == SCHEMA_VERSION
        assert loaded["generated_from"] == "/sp.json"
        assert len(loaded["characters"]) == 2

    def test_load_missing_returns_none(self, tmp_path):
        assert load_character_profiles(tmp_path / "nope.json") is None

    def test_load_invalid_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert load_character_profiles(path) is None

    def test_write_creates_parent_dirs(self, tmp_path):
        cp = build_character_profiles(_speaker_profiles_empty())
        path = tmp_path / "nested" / "dir" / "cp.json"
        write_character_profiles(path, cp)
        assert path.is_file()


# ---------------------------------------------------------------------------
# Mutations: rename, lock, unlock, set-review
# ---------------------------------------------------------------------------

class TestMutations:
    @pytest.fixture
    def cp_file(self, tmp_path):
        cp = build_character_profiles(_speaker_profiles_two(), tts_engine="qwen3-local")
        path = tmp_path / "cp.json"
        write_character_profiles(path, cp)
        return path

    def test_rename(self, cp_file):
        ret = rename_character(cp_file, "CHAR_001", "Alice")
        assert ret == 0
        data = load_character_profiles(cp_file)
        by_id = {c["character_id"]: c for c in data["characters"]}
        assert by_id["CHAR_001"]["name"] == "Alice"

    def test_rename_unknown_id(self, cp_file):
        ret = rename_character(cp_file, "CHAR_999", "Bob")
        assert ret == 1

    def test_lock_voice(self, cp_file):
        ret = lock_voice(cp_file, "CHAR_001", "/locked.wav")
        assert ret == 0
        data = load_character_profiles(cp_file)
        by_id = {c["character_id"]: c for c in data["characters"]}
        assert by_id["CHAR_001"]["voice_locked"] is True
        assert by_id["CHAR_001"]["voice_reference"] == "/locked.wav"

    def test_unlock_voice(self, cp_file):
        lock_voice(cp_file, "CHAR_001", "/locked.wav")
        ret = lock_voice(cp_file, "CHAR_001", "", locked=False)
        assert ret == 0
        data = load_character_profiles(cp_file)
        by_id = {c["character_id"]: c for c in data["characters"]}
        assert by_id["CHAR_001"]["voice_locked"] is False

    def test_set_review(self, cp_file):
        ret = set_review_status(cp_file, "CHAR_002", "approved")
        assert ret == 0
        data = load_character_profiles(cp_file)
        by_id = {c["character_id"]: c for c in data["characters"]}
        assert by_id["CHAR_002"]["review_status"] == "approved"

    def test_set_review_unknown_id(self, cp_file):
        ret = set_review_status(cp_file, "CHAR_999", "approved")
        assert ret == 1

    def test_mutation_on_missing_file(self, tmp_path):
        path = tmp_path / "nope.json"
        assert rename_character(path, "CHAR_001", "X") == 1
        assert lock_voice(path, "CHAR_001", "/x.wav") == 1
        assert set_review_status(path, "CHAR_001", "approved") == 1
