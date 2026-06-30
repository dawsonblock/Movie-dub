"""Tests for scripts/age_model.py — age regression plugin + heuristic fallback.

Covers: band mapping, source resolution, fallback behavior, model_path
preference, singleton reset, CLI status, and edge cases. No third-party
deps required (the model is never installed in CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import age_model
from age_model import (
    DEFAULT_AGE_MODEL_REPO,
    _heuristic_age,
    _resolve_source,
    _years_to_band,
    estimate_age,
    get_regressor,
    is_available,
    last_load_error,
    loaded_source,
    reset_for_tests,
)


# ---------------------------------------------------------------------------
# _years_to_band
# ---------------------------------------------------------------------------

class TestYearsToBand:
    def test_child(self):
        assert _years_to_band(5) == "child"
        assert _years_to_band(12.9) == "child"

    def test_teen(self):
        assert _years_to_band(13) == "teen"
        assert _years_to_band(19.9) == "teen"

    def test_young_adult(self):
        assert _years_to_band(20) == "young_adult"
        assert _years_to_band(34.9) == "young_adult"

    def test_adult(self):
        assert _years_to_band(35) == "adult"
        assert _years_to_band(59.9) == "adult"

    def test_senior(self):
        assert _years_to_band(60) == "senior"
        assert _years_to_band(95) == "senior"

    def test_unknown(self):
        assert _years_to_band(0) == "unknown"
        assert _years_to_band(-1) == "unknown"

    def test_boundaries(self):
        """Boundary values go to the lower band (exclusive upper)."""
        assert _years_to_band(13) == "teen"
        assert _years_to_band(20) == "young_adult"
        assert _years_to_band(35) == "adult"
        assert _years_to_band(60) == "senior"


# ---------------------------------------------------------------------------
# _resolve_source
# ---------------------------------------------------------------------------

class TestResolveSource:
    def test_explicit_model_path_wins(self, monkeypatch, tmp_path):
        fake_dir = tmp_path / "models" / "age_reg_ann_ecapa_librosa_combined"
        fake_dir.mkdir(parents=True)
        monkeypatch.setattr(age_model, "DEFAULT_AGE_MODEL_DIR", fake_dir)
        src = _resolve_source("/custom/path", DEFAULT_AGE_MODEL_REPO)
        assert src == "/custom/path"

    def test_local_dir_when_present(self, monkeypatch, tmp_path):
        fake_dir = tmp_path / "models" / "age_reg_ann_ecapa_librosa_combined"
        fake_dir.mkdir(parents=True)
        monkeypatch.setattr(age_model, "DEFAULT_AGE_MODEL_DIR", fake_dir)
        src = _resolve_source(None, DEFAULT_AGE_MODEL_REPO)
        assert src == fake_dir.as_posix()

    def test_repo_id_when_no_local_dir(self, monkeypatch, tmp_path):
        fake_dir = tmp_path / "nonexistent"
        monkeypatch.setattr(age_model, "DEFAULT_AGE_MODEL_DIR", fake_dir)
        src = _resolve_source(None, DEFAULT_AGE_MODEL_REPO)
        assert src == DEFAULT_AGE_MODEL_REPO

    def test_empty_string_model_path_falls_through(self, monkeypatch, tmp_path):
        """Empty string is falsy, so should fall through to local dir or repo."""
        fake_dir = tmp_path / "models" / "age_reg_ann_ecapa_librosa_combined"
        fake_dir.mkdir(parents=True)
        monkeypatch.setattr(age_model, "DEFAULT_AGE_MODEL_DIR", fake_dir)
        src = _resolve_source("", DEFAULT_AGE_MODEL_REPO)
        assert src == fake_dir.as_posix()


# ---------------------------------------------------------------------------
# _heuristic_age
# ---------------------------------------------------------------------------

class TestHeuristicAge:
    def test_child_high_f0(self):
        band, years, conf = _heuristic_age(300, 0.8, 250, 350)
        assert band == "child"
        assert years == 8.0
        assert conf == 0.4

    def test_teen(self):
        band, years, conf = _heuristic_age(240, 0.7, 200, 280)
        assert band == "teen"
        assert years == 15.0

    def test_senior_low_voiced(self):
        band, years, conf = _heuristic_age(120, 0.4, 90, 160)
        assert band == "senior"
        assert years == 70.0

    def test_adult(self):
        band, years, conf = _heuristic_age(120, 0.7, 90, 160)
        assert band == "adult"
        assert years == 35.0

    def test_unknown_low_voiced_ratio(self):
        band, years, conf = _heuristic_age(120, 0.2, 90, 160)
        assert band == "unknown"
        assert years == 0.0
        assert conf == 0.0

    def test_unknown_zero_f0(self):
        band, years, conf = _heuristic_age(0, 0.8, 0, 0)
        assert band == "unknown"


# ---------------------------------------------------------------------------
# estimate_age (fallback path — model not installed)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the plugin singleton before each test so tests are isolated."""
    reset_for_tests()
    yield
    reset_for_tests()


class TestEstimateAgeFallback:
    PITCH_ADULT = {
        "median_f0_hz": 120, "voiced_ratio": 0.7,
        "p10_f0_hz": 90, "p90_f0_hz": 160,
    }

    def test_off_uses_heuristic(self):
        r = estimate_age(Path("/nonexistent.wav"), self.PITCH_ADULT, use_model="off")
        assert r["source"] == "heuristic"
        assert r["method"] == "pitch_heuristic"
        assert r["band"] == "adult"
        assert "model_error" not in r

    def test_auto_falls_back_when_model_unavailable(self):
        r = estimate_age(Path("/nonexistent.wav"), self.PITCH_ADULT, use_model="auto")
        assert r["source"] == "heuristic"
        assert r["method"] == "pitch_heuristic"
        # model_error should be present since a load was attempted
        assert "model_error" in r
        assert "voice_age_regressor" in r["model_error"]

    def test_on_raises_when_model_unavailable(self):
        with pytest.raises(RuntimeError, match="make setup-age"):
            estimate_age(Path("/nonexistent.wav"), self.PITCH_ADULT, use_model="on")

    def test_off_has_no_model_error(self):
        """use_model=off should NOT attempt a load, so no model_error."""
        r = estimate_age(Path("/nonexistent.wav"), self.PITCH_ADULT, use_model="off")
        assert "model_error" not in r

    def test_auto_with_model_path_still_falls_back(self):
        """Even with a model_path, if the package isn't installed it falls back."""
        r = estimate_age(
            Path("/nonexistent.wav"), self.PITCH_ADULT,
            use_model="auto", model_path="/fake/path",
        )
        assert r["source"] == "heuristic"
        assert "model_error" in r


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_is_available_false_when_not_installed(self):
        assert is_available() is False

    def test_get_regressor_returns_none_when_not_installed(self):
        assert get_regressor() is None

    def test_last_load_error_populated_after_attempt(self):
        get_regressor()
        err = last_load_error()
        assert err != ""
        assert "voice_age_regressor" in err

    def test_reset_clears_error(self):
        get_regressor()
        assert last_load_error() != ""
        reset_for_tests()
        assert last_load_error() == ""
        assert loaded_source() == ""

    def test_loaded_source_empty_before_load(self):
        assert loaded_source() == ""


# ---------------------------------------------------------------------------
# CLI status
# ---------------------------------------------------------------------------

class TestCliStatus:
    def test_cli_status_exits_nonzero_when_unavailable(self, capsys):
        ret = age_model.cli_status()
        assert ret == 1  # unavailable
        out = capsys.readouterr().out
        assert "age_model: unavailable" in out
        assert "make setup-age" in out
        assert "fallback: pitch heuristic" in out

    def test_cli_status_shows_local_dir(self, capsys, monkeypatch, tmp_path):
        fake_dir = tmp_path / "models" / "age_reg_ann_ecapa_librosa_combined"
        monkeypatch.setattr(age_model, "DEFAULT_AGE_MODEL_DIR", fake_dir)
        age_model.cli_status()
        out = capsys.readouterr().out
        assert "local dir:" in out
        assert "missing" in out  # fake_dir doesn't exist
