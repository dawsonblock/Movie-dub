import importlib.util
import json
import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
REGEN_PATH = ROOT / "scripts" / "regenerate_segment.py"
TEST_VIDEO = ROOT / "assets" / "pyvideotrans_test_clip.mp4"

SPEC = importlib.util.spec_from_file_location("regenerate_segment", REGEN_PATH)
regenerate_segment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(regenerate_segment)


def _write_tone_wav(path: Path, duration: float, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration * sr)
    frames = b"".join(struct.pack("<h", int(16000 * (i % 200 / 200))) for i in range(n))
    with wave.open(path.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(frames)


def _make_job(tmp_path: Path) -> Path:
    job = tmp_path / "e2e_short_clip"
    gen = job / "generated_audio"
    gen.mkdir(parents=True, exist_ok=True)
    # Original segment audio (will be replaced).
    _write_tone_wav(gen / "000001.wav", 1.0)
    review = [
        {
            "id": 1,
            "start": 0.0,
            "end": 1.4,
            "source_text": "original line",
            "translated_text": "original line",
            "edited_text": "",
            "output_audio": (gen / "000001.wav").as_posix(),
            "status": "ok",
            "reason": "accept",
            "role": "clone",
            "ref_wav": (ROOT / "voices" / "openvoice_default_reference.wav").as_posix(),
            "language": "EN",
            "device": "cpu",
        }
    ]
    (job / "review_segments.json").write_text(json.dumps(review), encoding="utf-8")
    manifest = {
        "status": "ok",
        "ok": 1,
        "error": 0,
        "skipped": 0,
        "segments": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.4, "target_duration": 1.4,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
        "results": [
            {"id": 1, "status": "ok", "start": 0.0, "end": 1.4, "target_duration": 1.4,
             "output_audio": (gen / "000001.wav").as_posix()},
        ],
    }
    (job / "openvoice_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    remux_cmd = {
        "command": [sys.executable, "scripts/remux_dubbed_video.py", "--input-video",
                    TEST_VIDEO.as_posix(), "--dubbed-audio", (job / "dubbed_audio.wav").as_posix(),
                    "--output-video", (job / "final_dubbed.mp4").as_posix()],
        "build_audio_command": [sys.executable, "scripts/build_dubbed_audio_from_manifest.py",
                                "--manifest", (job / "openvoice_manifest.json").as_posix(),
                                "--input-video", TEST_VIDEO.as_posix(),
                                "--output-audio", (job / "dubbed_audio.wav").as_posix()],
    }
    (job / "remux_command.json").write_text(json.dumps(remux_cmd), encoding="utf-8")
    return job


def test_regenerate_replaces_segment_and_invokes_build_remux(tmp_path, monkeypatch):
    job = _make_job(tmp_path)
    segment_wav = job / "generated_audio" / "000001.wav"
    original_bytes = segment_wav.read_bytes()

    invoked = {"bridge": 0, "build": 0, "remux": 0}

    def fake_run(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        if "openvoice_segment_tts.py" in cmd_str:
            invoked["bridge"] += 1
            # Write a fake regenerated WAV at the queue's filename path.
            queue_idx = cmd.index("--queue-tts-file") + 1
            manifest_idx = cmd.index("--manifest-file") + 1
            queue = json.loads(Path(cmd[queue_idx]).read_text(encoding="utf-8"))
            for item in queue:
                out = Path(item["filename"])
                _write_tone_wav(out, 1.3)
            regen_manifest = {
                "status": "ok", "ok": 1, "error": 0, "skipped": 0,
                "results": [{"id": 1, "status": "ok", "start": 0.0, "end": 1.4,
                             "target_duration": 1.4, "generated_duration": 1.3,
                             "duration_ratio": 0.93, "timing_status": "accept",
                             "output_audio": queue[0]["filename"]}],
                "segments": [{"id": 1, "status": "ok", "start": 0.0, "end": 1.4,
                              "output_audio": queue[0]["filename"]}],
            }
            Path(cmd[manifest_idx]).write_text(json.dumps(regen_manifest), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "build_dubbed_audio_from_manifest.py" in cmd_str:
            invoked["build"] += 1
            out_idx = cmd.index("--output-audio") + 1
            _write_tone_wav(Path(cmd[out_idx]), 8.4)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "remux_dubbed_video.py" in cmd_str:
            invoked["remux"] += 1
            out_idx = cmd.index("--output-video") + 1
            # Copy the test video as a stand-in final output.
            import shutil
            shutil.copy2(TEST_VIDEO, Path(cmd[out_idx]))
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(regenerate_segment.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "regenerate_segment.py", "--job-dir", job.as_posix(),
        "--segment-id", "1", "--text", "This is a regenerated test line.", "--remux",
    ])

    rc = regenerate_segment.main()

    assert rc == 0
    assert invoked["bridge"] == 1
    assert invoked["build"] == 1
    assert invoked["remux"] == 1
    # Segment WAV was replaced (different bytes).
    assert segment_wav.read_bytes() != original_bytes
    # Manifest updated with regenerated result.
    manifest = json.loads((job / "openvoice_manifest.json").read_text(encoding="utf-8"))
    results = manifest.get("results", [])
    assert any(r.get("generated_duration") == 1.3 for r in results)
    # Review file updated.
    review = json.loads((job / "review_segments.json").read_text(encoding="utf-8"))
    assert review[0]["status"] == "regenerated"
    assert review[0]["edited_text"] == "This is a regenerated test line."


def test_regenerate_missing_review_file_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "regenerate_segment.py", "--job-dir", tmp_path.as_posix(),
        "--segment-id", "1", "--text", "x",
    ])
    rc = regenerate_segment.main()
    assert rc == 1


def test_regenerate_remux_syncs_to_user_output(tmp_path, monkeypatch):
    """--remux should sync the rebuilt job video to the user output path in report.json."""
    job = _make_job(tmp_path)
    user_output = tmp_path / "user_output.mp4"
    job_video = job / "final_dubbed.mp4"

    # Write report.json with user output path (as run_personal_dub.py does)
    report = {
        "status": "pass",
        "output_video": user_output.as_posix(),
        "final_video_job": job_video.as_posix(),
    }
    (job / "report.json").write_text(json.dumps(report), encoding="utf-8")

    invoked = {"build": 0, "remux": 0}

    def fake_run(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        if "openvoice_segment_tts.py" in cmd_str:
            queue_idx = cmd.index("--queue-tts-file") + 1
            manifest_idx = cmd.index("--manifest-file") + 1
            queue = json.loads(Path(cmd[queue_idx]).read_text(encoding="utf-8"))
            _write_tone_wav(Path(queue[0]["filename"]), 1.3)
            regen = {"status": "ok", "ok": 1, "error": 0, "skipped": 0,
                     "results": [{"id": 1, "status": "ok", "start": 0.0, "end": 1.4,
                                  "output_audio": queue[0]["filename"],
                                  "preserved_audio": queue[0]["filename"]}],
                     "segments": [{"id": 1, "status": "ok", "start": 0.0, "end": 1.4,
                                   "output_audio": queue[0]["filename"],
                                   "preserved_audio": queue[0]["filename"]}]}
            Path(cmd[manifest_idx]).write_text(json.dumps(regen), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "build_dubbed_audio_from_manifest.py" in cmd_str:
            invoked["build"] += 1
            out_idx = cmd.index("--output-audio") + 1
            _write_tone_wav(Path(cmd[out_idx]), 8.4)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "remux_dubbed_video.py" in cmd_str:
            invoked["remux"] += 1
            out_idx = cmd.index("--output-video") + 1
            import shutil
            shutil.copy2(TEST_VIDEO, Path(cmd[out_idx]))
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(regenerate_segment.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "regenerate_segment.py", "--job-dir", job.as_posix(),
        "--segment-id", "1", "--text", "Synced regeneration test.", "--remux",
    ])

    rc = regenerate_segment.main()

    assert rc == 0
    assert invoked["build"] == 1
    assert invoked["remux"] == 1
    # The job video was rebuilt
    assert job_video.is_file()
    # The user output path was synced
    assert user_output.is_file()


def test_regenerate_accepts_legacy_job_arg(tmp_path, monkeypatch):
    job = _make_job(tmp_path)

    def fake_run(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        if "openvoice_segment_tts.py" in cmd_str:
            queue_idx = cmd.index("--queue-tts-file") + 1
            manifest_idx = cmd.index("--manifest-file") + 1
            queue = json.loads(Path(cmd[queue_idx]).read_text(encoding="utf-8"))
            _write_tone_wav(Path(queue[0]["filename"]), 1.1)
            regen = {"status": "ok", "ok": 1, "error": 0, "skipped": 0,
                     "results": [{"id": 1, "status": "ok", "output_audio": queue[0]["filename"]}],
                     "segments": [{"id": 1, "status": "ok", "output_audio": queue[0]["filename"]}]}
            Path(cmd[manifest_idx]).write_text(json.dumps(regen), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(regenerate_segment.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "regenerate_segment.py", "--job", job.as_posix(),
        "--segment-id", "1", "--text", "legacy arg works",
    ])
    rc = regenerate_segment.main()
    assert rc == 0
