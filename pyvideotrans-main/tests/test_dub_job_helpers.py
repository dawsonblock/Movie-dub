"""Tests for the shared dub_job_helpers module.

Verifies that write_review_file and write_remux_command produce the
artifacts required by regenerate_segment.py --remux, so that personal
dub jobs (not just smoke jobs) support the regeneration workflow.
"""

import importlib.util
import json
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPERS_PATH = ROOT / "scripts" / "dub_job_helpers.py"
TEST_VIDEO = ROOT / "assets" / "pyvideotrans_test_clip.mp4"

SPEC = importlib.util.spec_from_file_location("dub_job_helpers", HELPERS_PATH)
dub_job_helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dub_job_helpers)


def _write_tone_wav(path: Path, duration: float, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration * sr)
    frames = b"".join(struct.pack("<h", int(16000 * (i % 200 / 200))) for i in range(n))
    with wave.open(path.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(frames)


def _make_manifest(gen_dir: Path) -> dict:
    """Build a realistic OpenVoice manifest with preserved_audio paths."""
    gen_dir.mkdir(parents=True, exist_ok=True)
    _write_tone_wav(gen_dir / "000001-1.wav", 1.5)
    _write_tone_wav(gen_dir / "000002-2.wav", 1.0)
    return {
        "status": "ok",
        "ok": 2,
        "error": 0,
        "skipped": 0,
        "language": "EN",
        "device": "mps",
        "results": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.5,
             "target_duration": 1.5, "output_audio": "/tmp/gone/seg1.wav",
             "preserved_audio": (gen_dir / "000001-1.wav").as_posix(),
             "timing_status": "accept"},
            {"id": 2, "status": "ok", "start": 2.0, "end": 3.0,
             "target_duration": 1.0, "output_audio": "/tmp/gone/seg2.wav",
             "preserved_audio": (gen_dir / "000002-2.wav").as_posix(),
             "timing_status": "rewrite_shorter"},
        ],
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.5,
             "target_duration": 1.5, "output_audio": "/tmp/gone/seg1.wav",
             "preserved_audio": (gen_dir / "000001-1.wav").as_posix(),
             "timing_status": "accept"},
            {"id": 2, "status": "ok", "start": 2.0, "end": 3.0,
             "target_duration": 1.0, "output_audio": "/tmp/gone/seg2.wav",
             "preserved_audio": (gen_dir / "000002-2.wav").as_posix(),
             "timing_status": "rewrite_shorter"},
        ],
    }


def test_write_review_file_produces_all_required_fields(tmp_path):
    """review_segments.json must have id, start, end, output_audio, status, etc."""
    job_dir = tmp_path / "personal_dub" / "123"
    gen_dir = job_dir / "generated_audio"
    manifest_data = _make_manifest(gen_dir)
    manifest_path = job_dir / "openvoice_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    review_path = job_dir / "review_segments.json"
    srt_path = tmp_path / "nonexistent.srt"  # no SRT → fallback to manifest

    dub_job_helpers.write_review_file(
        review_path, None, manifest_path, srt_path, job_dir, gen_dir
    )

    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(review) == 2

    required_keys = {"id", "start", "end", "source_text", "translated_text",
                     "edited_text", "output_audio", "status", "reason",
                     "role", "ref_wav", "language", "device"}
    for entry in review:
        assert required_keys.issubset(entry.keys()), f"missing keys: {required_keys - entry.keys()}"

    # Segment 2 has timing_status=rewrite_shorter → needs_review
    assert review[0]["status"] == "ok"
    assert review[1]["status"] == "needs_review"

    # output_audio should point to generated_audio dir with preserved WAVs
    assert Path(review[0]["output_audio"]).is_file()
    assert Path(review[1]["output_audio"]).is_file()


def test_write_remux_command_produces_dict_format(tmp_path):
    """remux_command.json must have command + build_audio_command keys."""
    job_dir = tmp_path / "personal_dub" / "123"
    remux_path = job_dir / "remux_command.json"
    input_video = TEST_VIDEO
    dubbed_audio = job_dir / "dubbed_audio.wav"
    output_video = job_dir / "final_dubbed.mp4"
    manifest_path = job_dir / "openvoice_manifest.json"

    dub_job_helpers.write_remux_command(
        remux_path, input_video, dubbed_audio, output_video, manifest_path
    )

    cmd = json.loads(remux_path.read_text(encoding="utf-8"))
    assert "command" in cmd
    assert "build_audio_command" in cmd

    # command must reference remux_dubbed_video.py
    assert any("remux_dubbed_video.py" in str(part) for part in cmd["command"])
    # build_audio_command must reference build_dubbed_audio_from_manifest.py
    assert any("build_dubbed_audio_from_manifest.py" in str(part) for part in cmd["build_audio_command"])

    # Both must reference the manifest path
    assert manifest_path.as_posix() in [str(p) for p in cmd["build_audio_command"]]


def test_personal_dub_job_has_all_regeneration_artifacts(tmp_path):
    """A personal dub job dir must contain all files needed for regeneration."""
    job_dir = tmp_path / "personal_dub" / "123"
    gen_dir = job_dir / "generated_audio"
    manifest_data = _make_manifest(gen_dir)
    manifest_path = job_dir / "openvoice_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    review_path = job_dir / "review_segments.json"
    remux_path = job_dir / "remux_command.json"
    dubbed_audio = job_dir / "dubbed_audio.wav"
    output_video = job_dir / "final_dubbed.mp4"
    srt_path = tmp_path / "nonexistent.srt"

    # Write review + remux (as run_personal_dub.py does)
    dub_job_helpers.write_review_file(
        review_path, None, manifest_path, srt_path, job_dir, gen_dir
    )
    dub_job_helpers.write_remux_command(
        remux_path, TEST_VIDEO, dubbed_audio, output_video, manifest_path
    )

    # All regeneration artifacts must exist
    assert manifest_path.is_file(), "openvoice_manifest.json missing"
    assert review_path.is_file(), "review_segments.json missing"
    assert remux_path.is_file(), "remux_command.json missing"
    assert gen_dir.is_dir(), "generated_audio/ dir missing"

    # The remux_command must be consumable by regenerate_segment.py
    remux_cmd = json.loads(remux_path.read_text(encoding="utf-8"))
    assert isinstance(remux_cmd, dict)
    assert isinstance(remux_cmd.get("command"), list)
    assert isinstance(remux_cmd.get("build_audio_command"), list)

    # The review must have entries matching the manifest
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(review) == len(manifest_data["results"])
    review_ids = {str(r["id"]) for r in review}
    manifest_ids = {str(r["id"]) for r in manifest_data["results"]}
    assert review_ids == manifest_ids


def test_extract_manifest_from_output_finds_manifest():
    text = 'some log noise\n{"ok": 2, "error": 0, "results": [{"id": 1}]}\nmore noise'
    result = dub_job_helpers.extract_manifest_from_output(text)
    assert result is not None
    assert result["ok"] == 2
    assert len(result["results"]) == 1


def test_extract_manifest_from_output_returns_none_when_absent():
    assert dub_job_helpers.extract_manifest_from_output("no json here") is None


def test_write_review_file_handles_none_srt(tmp_path):
    """write_review_file must not crash when srt_file is None (no SRT found)."""
    job_dir = tmp_path / "personal_dub" / "456"
    gen_dir = job_dir / "generated_audio"
    manifest_data = _make_manifest(gen_dir)
    manifest_path = job_dir / "openvoice_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    review_path = job_dir / "review_segments.json"

    # Pass None for srt_file — should not raise
    dub_job_helpers.write_review_file(
        review_path, None, manifest_path, None, job_dir, gen_dir
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(review) == 2


def test_write_review_file_preserves_manifest_text_without_srt(tmp_path):
    """When no SRT/queue exists, review text must fall back to manifest text."""
    job_dir = tmp_path / "personal_dub" / "789"
    gen_dir = job_dir / "generated_audio"
    gen_dir.mkdir(parents=True, exist_ok=True)
    _write_tone_wav(gen_dir / "000001-1.wav", 1.0)
    manifest_data = {
        "status": "ok", "ok": 1, "error": 0, "skipped": 0,
        "language": "EN", "device": "mps",
        "results": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "target_duration": 1.0, "text": "Hello from the manifest",
             "output_audio": "/tmp/gone/seg1.wav",
             "preserved_audio": (gen_dir / "000001-1.wav").as_posix(),
             "timing_status": "accept"},
        ],
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.0,
             "text": "Hello from the manifest",
             "preserved_audio": (gen_dir / "000001-1.wav").as_posix(),
             "timing_status": "accept"},
        ],
    }
    manifest_path = job_dir / "openvoice_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    review_path = job_dir / "review_segments.json"

    # No queue, no SRT — text must come from manifest
    dub_job_helpers.write_review_file(
        review_path, None, manifest_path, None, job_dir, gen_dir
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(review) == 1
    assert review[0]["translated_text"] == "Hello from the manifest"


def test_write_remux_command_preserves_audio_options(tmp_path):
    """remux_command.json must include audio-quality flags in build_audio_command."""
    job_dir = tmp_path / "job"
    remux_path = job_dir / "remux_command.json"
    input_video = TEST_VIDEO
    dubbed_audio = job_dir / "dubbed_audio.wav"
    output_video = job_dir / "final_dubbed.mp4"
    manifest_path = job_dir / "openvoice_manifest.json"

    audio_options = {
        "background_volume": 0.15,
        "voice_volume": 1.2,
        "final_gain": 0.9,
        "no_normalize": False,
        "vocal_separation": True,
        "ducking": True,
        "target_lufs": -16.0,
        "background_timeout": 600,
        "lufs_timeout": 900,
        "fail_if_background_mix_fails": True,
    }

    dub_job_helpers.write_remux_command(
        remux_path, input_video, dubbed_audio, output_video,
        manifest_path, audio_options=audio_options,
    )

    cmd = json.loads(remux_path.read_text(encoding="utf-8"))
    build_cmd = cmd["build_audio_command"]
    build_str = " ".join(str(p) for p in build_cmd)

    # Verify each audio option appears in the build command
    assert "--background-volume" in build_str
    assert "0.15" in build_str
    assert "--voice-volume" in build_str
    assert "1.2" in build_str
    assert "--final-gain" in build_str
    assert "0.9" in build_str
    assert "--vocal-separation" in build_str
    assert "--ducking" in build_str
    assert "--target-lufs" in build_str
    assert "-16.0" in build_str or "-16" in build_str
    assert "--background-timeout" in build_str
    assert "600" in build_str
    assert "--lufs-timeout" in build_str
    assert "900" in build_str
    assert "--fail-if-background-mix-fails" in build_str

    # audio_options should be stored in the remux command
    assert cmd["audio_options"]["background_volume"] == 0.15
    assert cmd["audio_options"]["ducking"] is True


def test_write_remux_command_without_audio_options(tmp_path):
    """remux_command.json without audio_options should still work."""
    job_dir = tmp_path / "job"
    remux_path = job_dir / "remux_command.json"
    input_video = TEST_VIDEO
    dubbed_audio = job_dir / "dubbed_audio.wav"
    output_video = job_dir / "final_dubbed.mp4"
    manifest_path = job_dir / "openvoice_manifest.json"

    dub_job_helpers.write_remux_command(
        remux_path, input_video, dubbed_audio, output_video, manifest_path
    )

    cmd = json.loads(remux_path.read_text(encoding="utf-8"))
    assert "build_audio_command" in cmd
    assert cmd["audio_options"] == {}
    # No audio quality flags in build command
    build_str = " ".join(str(p) for p in cmd["build_audio_command"])
    assert "--background-volume" not in build_str
    assert "--ducking" not in build_str
