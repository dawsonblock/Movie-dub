"""Tests for CLI flag wiring in the pipeline scripts.

Verifies that the new v0.12 flags are accepted by the argparser of
run_personal_dub.py, analyze_speakers.py, and regenerate_segment.py
without actually running the pipeline. Uses subprocess to check --help
output so we test the real entry point.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _help_flags(script: str) -> str:
    """Run `python <script> --help` and return stdout."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"{script} --help failed: {result.stderr}"
    return result.stdout


class TestRunPersonalDubFlags:
    def test_age_model_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--age-model" in out
        assert "auto" in out
        assert "on" in out
        assert "off" in out

    def test_age_model_path_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--age-model-path" in out

    def test_reference_clip_scoring_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--reference-clip-scoring" in out

    def test_character_profiles_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--character-profiles" in out

    def test_resume_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--resume" in out

    def test_from_stage_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--from-stage" in out

    def test_only_segment_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--only-segment" in out

    def test_skip_existing_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--skip-existing" in out

    def test_no_cache_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--no-cache" in out

    def test_omnivoice_url_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--omnivoice-url" in out

    def test_tts_speed_flag(self):
        out = _help_flags("run_personal_dub.py")
        assert "--tts-speed" in out


class TestOmniVoiceBridgeFlags:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "bridge" / "omnivoice_segment_tts.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "--queue-tts-file" in out
        assert "--manifest-file" in out
        assert "--api-url" in out
        assert "--studio" in out


class TestAnalyzeSpeakersFlags:
    def test_age_model_flag(self):
        out = _help_flags("analyze_speakers.py")
        assert "--age-model" in out

    def test_age_model_path_flag(self):
        out = _help_flags("analyze_speakers.py")
        assert "--age-model-path" in out

    def test_reference_clip_scoring_flag(self):
        out = _help_flags("analyze_speakers.py")
        assert "--reference-clip-scoring" in out


class TestRegenerateSegmentFlags:
    def test_tts_engine_flag(self):
        out = _help_flags("regenerate_segment.py")
        assert "--tts-engine" in out
        assert "openvoice" in out
        assert "qwen3-local" in out
        assert "omnivoice" in out

    def test_omnivoice_url_flag(self):
        out = _help_flags("regenerate_segment.py")
        assert "--omnivoice-url" in out

    def test_change_speaker_flag(self):
        out = _help_flags("regenerate_segment.py")
        assert "--change-speaker" in out

    def test_change_reference_flag(self):
        out = _help_flags("regenerate_segment.py")
        assert "--change-reference" in out

    def test_shorten_flag(self):
        out = _help_flags("regenerate_segment.py")
        assert "--shorten" in out

    def test_qwen3_model_flag(self):
        out = _help_flags("regenerate_segment.py")
        assert "--qwen3-model" in out


class TestBenchmarkFlags:
    def test_input_required(self):
        out = _help_flags("benchmark.py")
        assert "--input" in out
        assert "--output-dir" in out

    def test_tts_engine_flag(self):
        out = _help_flags("benchmark.py")
        assert "--tts-engine" in out

    def test_age_model_flag(self):
        out = _help_flags("benchmark.py")
        assert "--age-model" in out


class TestReviewLoopFlags:
    def test_failed_subcommand(self):
        out = _help_flags("review_loop.py")
        assert "failed" in out
        assert "change-speaker" in out
        assert "change-reference" in out


class TestEstimateSpeakerAgeFlags:
    def test_audio_required(self):
        out = _help_flags("estimate_speaker_age.py")
        assert "--audio" in out
        assert "--output" in out
        assert "--model-path" in out

    def test_no_fallback_flag(self):
        out = _help_flags("estimate_speaker_age.py")
        assert "--no-fallback" in out


class TestCharacterProfilesFlags:
    def test_required_flags(self):
        out = _help_flags("character_profiles.py")
        assert "--speaker-profiles" in out
        assert "--output" in out
        assert "--rename" in out
        assert "--lock-voice" in out
        assert "--set-review" in out


class TestCacheFlags:
    def test_cache_dir_required(self):
        out = _help_flags("cache.py")
        assert "--cache-dir" in out
        assert "--summary" in out
