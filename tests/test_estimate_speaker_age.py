"""Tests for scripts/estimate_speaker_age.py — standalone age CLI wrapper.

Covers: age_band mapping, estimate_age function (mocked), CLI argument
validation, missing audio file handling, and output file writing.
Does NOT require the actual model to be installed.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from estimate_speaker_age import age_band, estimate_age, main


@pytest.fixture
def mock_age_regressor():
    """Inject a fake age_regressor module into sys.modules for the test."""
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [42.0]
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value = mock_pipeline

    fake_module = types.ModuleType("age_regressor")
    fake_module.AgeRegressionPipeline = mock_cls
    old = sys.modules.get("age_regressor")
    sys.modules["age_regressor"] = fake_module
    try:
        yield mock_cls, mock_pipeline
    finally:
        if old is not None:
            sys.modules["age_regressor"] = old
        else:
            sys.modules.pop("age_regressor", None)


# ---------------------------------------------------------------------------
# age_band
# ---------------------------------------------------------------------------

class TestAgeBand:
    def test_child(self):
        assert age_band(5) == "child"
        assert age_band(12.9) == "child"

    def test_teen(self):
        assert age_band(13) == "teen"
        assert age_band(19.9) == "teen"

    def test_young_adult(self):
        assert age_band(20) == "young_adult"
        assert age_band(34.9) == "young_adult"

    def test_adult(self):
        assert age_band(35) == "adult"
        assert age_band(59.9) == "adult"

    def test_senior(self):
        assert age_band(60) == "senior"
        assert age_band(100) == "senior"

    def test_zero(self):
        assert age_band(0) == "child"

    def test_negative(self):
        assert age_band(-5) == "child"


# ---------------------------------------------------------------------------
# estimate_age (mocked — no real model needed)
# ---------------------------------------------------------------------------

class TestEstimateAgeMocked:
    def test_returns_correct_structure(self, mock_age_regressor):
        mock_cls, mock_pipeline = mock_age_regressor
        mock_pipeline.return_value = [42.7]
        result = estimate_age(Path("/fake.wav"), None)
        assert result["estimated_years"] == 42.7
        assert result["band"] == "adult"
        assert "method" in result
        assert "note" in result
        assert result["confidence"] is None

    def test_with_model_path(self, mock_age_regressor):
        mock_cls, mock_pipeline = mock_age_regressor
        mock_pipeline.return_value = [25.0]
        result = estimate_age(Path("/fake.wav"), "/local/model")
        assert result["estimated_years"] == 25.0
        assert result["band"] == "young_adult"
        assert result["method"] == "/local/model"

    def test_teen_age(self, mock_age_regressor):
        _, mock_pipeline = mock_age_regressor
        mock_pipeline.return_value = [16.5]
        result = estimate_age(Path("/fake.wav"), None)
        assert result["band"] == "teen"

    def test_child_age(self, mock_age_regressor):
        _, mock_pipeline = mock_age_regressor
        mock_pipeline.return_value = [8.0]
        result = estimate_age(Path("/fake.wav"), None)
        assert result["band"] == "child"

    def test_senior_age(self, mock_age_regressor):
        _, mock_pipeline = mock_age_regressor
        mock_pipeline.return_value = [72.0]
        result = estimate_age(Path("/fake.wav"), None)
        assert result["band"] == "senior"


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestCliMain:
    def test_missing_audio_returns_1(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "estimate_speaker_age.py",
            "--audio", "/nonexistent.wav",
            "--output", "/tmp/out.json",
        ])
        ret = main()
        assert ret == 1
        err = capsys.readouterr().err
        assert "Missing audio file" in err

    def test_writes_output_json(self, tmp_path, mock_age_regressor, monkeypatch):
        mock_cls, mock_pipeline = mock_age_regressor
        mock_pipeline.return_value = [42.0]
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake audio")
        output = tmp_path / "out.json"
        monkeypatch.setattr(sys, "argv", [
            "estimate_speaker_age.py",
            "--audio", str(audio),
            "--output", str(output),
            "--model-path", "/local/model",
        ])
        ret = main()
        assert ret == 0
        assert output.is_file()
        data = json.loads(output.read_text())
        assert data["estimated_years"] == 42.0
        assert data["band"] == "adult"

    def test_failure_returns_1(self, tmp_path, capsys, monkeypatch):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake audio")
        output = tmp_path / "out.json"
        # Don't mock — the real import will fail since age_regressor isn't installed
        monkeypatch.setattr(sys, "argv", [
            "estimate_speaker_age.py",
            "--audio", str(audio),
            "--output", str(output),
        ])
        ret = main()
        assert ret == 1
        err = capsys.readouterr().err
        assert "age estimation failed" in err or "setup-age" in err

    def test_creates_output_parent_dir(self, tmp_path, mock_age_regressor, monkeypatch):
        _, mock_pipeline = mock_age_regressor
        mock_pipeline.return_value = [30.0]
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"fake audio")
        output = tmp_path / "nested" / "dir" / "out.json"
        monkeypatch.setattr(sys, "argv", [
            "estimate_speaker_age.py",
            "--audio", str(audio),
            "--output", str(output),
        ])
        ret = main()
        assert ret == 0
        assert output.is_file()
