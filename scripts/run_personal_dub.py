#!/usr/bin/env python3
"""Dub a personal video end-to-end with pyVideoTrans + OpenVoice V2.

This is the user-facing CLI promised in Phase 10. It runs the full
pyVideoTrans VTV pipeline with the OpenVoice provider, then rebuilds
the dubbed audio track from the OpenVoice manifest and remuxes it into
a final dubbed MP4 at the requested output path.

Workflow:
  1. Validate inputs (video, reference WAV, venvs, checkpoints).
  2. Point OpenVoice segment preservation at a job generated_audio dir.
  3. Run pyVideoTrans cli.py --task vtv --tts_type 34.
  4. Locate the OpenVoice manifest + preserved segment WAVs.
  5. Build dubbed_audio.wav from the manifest.
  6. Remux into the final output MP4.
  7. Write review_segments.json + remux_command.json for regeneration.
  8. Write report.json.

Example:
  python scripts/run_personal_dub.py \
    --input ~/Movies/test.mp4 \
    --source-language en \
    --target-language en \
    --reference voices/openvoice_default_reference.wav \
    --output ~/Movies/dubbed-test.mp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Shared helpers for review_segments.json + remux_command.json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dub_job_helpers import (
    extract_manifest_from_output as _extract_manifest_from_output,
    write_remux_command as _write_remux_command,
    write_review_file as _write_review_file,
)
from extract_embedded_subtitles import (
    extract_embedded_subtitles as _extract_embedded_subtitles,
)
from job_state import (
    PERSONAL_DUB_ROOT,
    UnsafeJobDirError,
    create_job_dir,
    generate_job_id,
    initial_job_state,
    read_job_state,
    update_job_stage,
    write_job_state,
)
from cache import (
    ResumePlan,
    SegmentCache,
    content_hash,
    dict_hash,
    parse_from_stage,
    segment_cache_key,
)
from character_profiles import (
    build_character_profiles as _build_character_profiles,
    load_character_profiles as _load_character_profiles,
    write_character_profiles as _write_character_profiles,
)
from review_loop import write_failed_segments as _write_failed_segments


ROOT = Path(__file__).resolve().parents[1]
PYVIDEOTRANS = ROOT / "pyvideotrans-main"
OPENVOICE = ROOT / "OpenVoice-main"
DEFAULT_REFERENCE = ROOT / "voices" / "openvoice_default_reference.wav"
MALE_REFERENCE = ROOT / "voices" / "male_reference.wav"
FEMALE_REFERENCE = ROOT / "voices" / "female_reference.wav"
BUILD_AUDIO_SCRIPT = ROOT / "scripts" / "build_dubbed_audio_from_manifest.py"
REMUX_SCRIPT = ROOT / "scripts" / "remux_dubbed_video.py"
GENDER_DETECT_SCRIPT = ROOT / "scripts" / "detect_gender.py"
ANALYZE_SPEAKERS_SCRIPT = ROOT / "scripts" / "analyze_speakers.py"
EXTRACT_SUBTITLES_SCRIPT = ROOT / "scripts" / "extract_embedded_subtitles.py"
VERIFY_PITCH_SCRIPT = ROOT / "scripts" / "verify_segment_pitch.py"
PYVT_CONFIG = PYVIDEOTRANS / "videotrans" / "cfg.json"
PRESERVE_KEY = "openvoice_preserve_dir"
HUGGINGFACE_ASR_PROVIDER = "4"  # HuggingFace ASR for whisper-large-v3-turbo

# Bridge scripts for the split pipeline
OPENVOICE_BRIDGE_SCRIPT = ROOT / "bridge" / "openvoice_segment_tts.py"
QWEN3_BRIDGE_SCRIPT = ROOT / "bridge" / "qwen3_segment_tts.py"
OMNIVOICE_BRIDGE_SCRIPT = ROOT / "bridge" / "omnivoice_segment_tts.py"

# Default Qwen3-TTS model (MLX bf16, 0.6B Base, supports voice cloning).
# This is the mlx-community repo that includes the speech_tokenizer/
# subdir required by mlx-audio's post_load_hook. The aufklarer 4-bit
# repo is Base-talker-only and is missing the speech tokenizer, so
# model.generate() fails with "Speech tokenizer not loaded".
QWEN3_DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
# Local path fallback (if downloaded to models/)
QWEN3_LOCAL_MODEL = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base-bf16"

# TTS engine -> pyVideoTrans provider index (videotrans/tts/__init__.py)
# openvoice   -> 34 (OpenVoice V2 local, current default)
# qwen3-local -> 1  (Qwen3-TTS local built-in)
# omnivoice   -> 2  (OmniVoice local API)
TTS_ENGINE_MAP = {
    "openvoice": "34",
    "qwen3-local": "1",
    "omnivoice": "2",
}
DEFAULT_TTS_ENGINE = "openvoice"


def die(msg: str, code: int = 1) -> int:
    print(f"run_personal_dub: FAIL\nReason: {msg}", file=sys.stderr)
    return code


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def local_ffprobe() -> str:
    """Find ffprobe: prefer bundled, then PATH, then Homebrew locations."""
    candidates = [
        PYVIDEOTRANS / "ffmpeg" / "ffprobe",
        Path(shutil.which("ffprobe") or ""),
        Path("/opt/homebrew/bin/ffprobe"),
        Path("/usr/local/bin/ffprobe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    raise RuntimeError("ffprobe not found")


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [local_ffprobe(), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path.as_posix()],
        text=True, capture_output=True, check=True,
    )
    return float(result.stdout.strip())


def _qwen3_preflight(model_arg: str) -> str | None:
    """Pre-flight checks for the Qwen3-local TTS engine.

    The Qwen3 bridge runs on sys.executable (the same python that runs
    this wrapper), so mlx_audio / soundfile / librosa must be importable
    here. Returns an error string if anything is missing, else None.
    """
    missing: list[str] = []
    for mod in ("mlx_audio", "soundfile", "librosa"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return (
            f"Qwen3-local requires these packages importable from the "
            f"wrapper python ({sys.executable}): "
            f"{', '.join(missing)}. Run: make setup-qwen3"
        )
    # Model resolution: local dir wins, else the HF repo ID is fetched
    # on first generate. We only check the local dir here; the HF path
    # is validated lazily by mlx-audio.
    if model_arg == QWEN3_DEFAULT_MODEL and QWEN3_LOCAL_MODEL.is_dir():
        return None
    if model_arg and Path(model_arg).expanduser().is_dir():
        return None
    # If a custom repo ID was given, we can't prove it without network;
    # let mlx-audio fail at generate time with a real error.
    return None


def _omnivoice_preflight(api_url: str) -> str | None:
    """Pre-flight checks for the OmniVoice TTS engine.

    The OmniVoice bridge runs on sys.executable (the same python that runs
    this wrapper). Studio mode needs ``requests``; Gradio mode also needs
    ``gradio_client``. Returns an error string if required packages are
    missing, else None.
    """
    if not api_url or api_url.strip() == "":
        return "--omnivoice-url is required when --tts-engine omnivoice is used in split mode."
    missing: list[str] = []
    for mod in ("requests", "gradio_client"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return (
            f"OmniVoice requires these packages importable from the "
            f"wrapper python ({sys.executable}): "
            f"{', '.join(missing)}. Install with: "
            f"python3 -m pip install gradio_client requests"
        )
    return None


def _verify_final_video(video_path: Path, source_duration: float) -> dict:
    """Verify a final MP4 with ffprobe (Blocker 11).

    Checks: video stream exists, audio stream exists, duration exists,
    duration is close to source, audio/video codecs exist, file size > 1KB.
    Duration mismatch fails if larger than max(2s, 2% of source duration).
    """
    try:
        result = subprocess.run(
            [local_ffprobe(), "-v", "error", "-show_streams", "-show_format",
             "-of", "json", video_path.as_posix()],
            text=True, capture_output=True, check=True, timeout=30,
        )
        info = json.loads(result.stdout)
    except Exception as exc:
        return {"verified": False, "error": str(exc)}

    streams = info.get("streams", [])
    fmt = info.get("format", {})
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video_codec = next(
        (s.get("codec_name") for s in streams if s.get("codec_type") == "video"),
        None,
    )
    audio_codec = next(
        (s.get("codec_name") for s in streams if s.get("codec_type") == "audio"),
        None,
    )
    duration_str = fmt.get("duration", "0")
    try:
        output_duration = float(duration_str)
    except (ValueError, TypeError):
        output_duration = 0.0

    errors: list[str] = []
    if not has_video:
        errors.append("missing video stream")
    if not has_audio:
        errors.append("missing audio stream")
    if not video_codec:
        errors.append("missing video codec")
    if not audio_codec:
        errors.append("missing audio codec")
    if output_duration <= 0:
        errors.append("invalid duration")
    else:
        delta = abs(output_duration - source_duration)
        tolerance = max(2.0, 0.02 * source_duration)
        if delta > tolerance:
            errors.append(
                f"duration mismatch: {delta:.1f}s exceeds tolerance "
                f"{tolerance:.1f}s (source={source_duration:.1f}, "
                f"output={output_duration:.1f})"
            )
    file_size = video_path.stat().st_size if video_path.is_file() else 0
    if file_size < 1024:
        errors.append(f"file too small: {file_size} bytes")

    return {
        "verified": len(errors) == 0,
        "source_duration": source_duration,
        "output_duration": output_duration,
        "duration_delta": abs(output_duration - source_duration),
        "has_video": has_video,
        "has_audio": has_audio,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "file_size": file_size,
        "errors": errors,
    }


def latest_artifact(root: Path, start_time: float, pattern: str) -> Path | None:
    candidates = [p for p in root.glob(pattern) if p.stat().st_mtime >= start_time]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def parse_srt_to_segments(srt_path: Path) -> list[dict]:
    """Parse an SRT file into a list of segment dicts with timings.

    Returns: [{"segment_index": 0, "start": 1.2, "end": 4.8, "text": "..."}, ...]
    Times are in seconds (float). segment_index is 0-based.
    """
    import re as _re
    text = srt_path.read_text(encoding="utf-8-sig")
    segments: list[dict] = []
    # SRT blocks: index line, time line, text lines, blank line
    blocks = _re.split(r"\n\s*\n", text.strip())
    for idx, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Find the time line (contains -->)
        time_line = next((ln for ln in lines if "-->" in ln), None)
        if not time_line:
            continue
        m = _re.match(
            r"(\d+):(\d+):(\d+)[,\.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,\.](\d+)",
            time_line.strip(),
        )
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
        # Text is everything after the time line
        text_lines = lines[lines.index(time_line) + 1:]
        seg_text = " ".join(ln.strip() for ln in text_lines if ln.strip())
        segments.append({
            "segment_index": idx,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": seg_text,
        })
    return segments


def _locked_voice_overrides(
    character_profiles: dict | None,
) -> dict[str, str]:
    """Build a speaker_id -> voice_reference map for locked characters.

    A character with ``voice_locked=True`` and a non-empty ``voice_reference``
    overrides the speaker profile's reference_audio. This is how a user-locked
    voice in character_profiles.json drives TTS routing.
    """
    if not character_profiles:
        return {}
    overrides: dict[str, str] = {}
    for ch in character_profiles.get("characters", []):
        if ch.get("voice_locked") and ch.get("voice_reference"):
            sid = ch.get("speaker_id", "")
            if sid:
                overrides[sid] = ch["voice_reference"]
    return overrides


def build_queue_tts_with_speakers(
    translated_srt: Path,
    speaker_profiles: dict,
    cache_folder: Path,
    tts_type: int = 34,
    voice_rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    language: str = "en",
    character_profiles: dict | None = None,
) -> list[dict]:
    """Build a queue_tts list with per-speaker ref_wav from speaker profiles.

    Each queue item gets:
      - line: 1-based line number
      - text: translated text
      - role: "clone" (so OpenVoice provider knows to use ref_wav)
      - ref_wav: path to the speaker's reference.wav
      - start_time / end_time: in milliseconds (pyVideoTrans convention)
      - filename: output WAV path in cache_folder
      - tts_type, rate, volume, pitch

    If ``character_profiles`` is provided, any character with
    ``voice_locked=True`` and a ``voice_reference`` overrides the speaker
    profile's ``reference_audio``. This is how a user-locked voice in
    character_profiles.json drives TTS routing.
    """
    segments = parse_srt_to_segments(translated_srt)
    seg_assignments = speaker_profiles.get("segment_assignments", [])
    profiles_by_id = {
        p["speaker_id"]: p for p in speaker_profiles.get("speakers", [])
    }
    # Locked-voice overrides from character_profiles.json (v0.12).
    voice_overrides = _locked_voice_overrides(character_profiles)
    # Build segment_index -> speaker_id map
    seg_to_speaker: dict[int, str] = {}
    for sa in seg_assignments:
        seg_to_speaker[sa.get("segment_index", 0)] = sa.get("speaker_id", "")

    queue: list[dict] = []
    for seg in segments:
        idx = seg["segment_index"]
        speaker_id = seg_to_speaker.get(idx, "")
        profile = profiles_by_id.get(speaker_id, {})
        # A locked character voice overrides the speaker profile's reference.
        ref_wav = voice_overrides.get(speaker_id) or profile.get("reference_audio", "")
        if not ref_wav or not Path(ref_wav).is_file():
            raise RuntimeError(
                f"segment {idx}: speaker '{speaker_id}' has no reference_audio "
                f"file at {ref_wav}"
            )
        text = seg["text"]
        voice = "clone"
        _key = hashlib.md5(
            f"{language}-{text}-{voice}-{voice_rate}-{volume}-{pitch}-{tts_type}".encode()
        ).hexdigest()
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        queue.append({
            "line": idx + 1,
            "text": text,
            "role": voice,
            "ref_wav": ref_wav,
            "voice_reference": ref_wav,
            "start_time": start_ms,
            "end_time": end_ms,
            "rate": voice_rate,
            "volume": volume,
            "pitch": pitch,
            "tts_type": tts_type,
            "filename": f"{cache_folder}/dubb-{idx}-{_key}.wav",
            "openvoice_output": f"{cache_folder}/dubb-{idx}-{_key}.wav-openvoice.wav",
        })
    return queue


def _tts_model_id(engine: str, args) -> str:
    """Return a stable model identifier for cache keying."""
    if engine == "qwen3-local":
        return args.qwen3_model
    if engine == "omnivoice":
        return f"omnivoice-{args.omnivoice_url}"
    return "openvoice-v2"


def _compute_segment_cache_keys(
    queue_tts: list[dict],
    *,
    source_audio_hash: str,
    subtitle_hash: str,
    speaker_profile_hash: str,
    tts_engine: str,
    tts_model: str,
    language: str,
) -> dict[str, str]:
    """Compute a deterministic cache key for each queue item.

    Returns a mapping of ``str(line) -> cache_key``. Reference audio is hashed
    per-unique-path (cached to avoid re-reading the same file).
    """
    ref_hash_cache: dict[str, str] = {}

    def _ref_hash(path_str: str) -> str:
        if path_str not in ref_hash_cache:
            p = Path(path_str)
            ref_hash_cache[path_str] = (
                content_hash(p) if p.is_file() else "missing"
            )
        return ref_hash_cache[path_str]

    keys: dict[str, str] = {}
    for item in queue_tts:
        line = str(item.get("line", item.get("id", "")))
        ref = item.get("ref_wav") or item.get("voice_reference") or ""
        keys[line] = segment_cache_key(
            source_audio_hash=source_audio_hash,
            subtitle_hash=subtitle_hash,
            speaker_profile_hash=speaker_profile_hash,
            translated_text=item.get("text", ""),
            tts_engine=tts_engine,
            tts_model=tts_model,
            reference_audio_hash=_ref_hash(ref),
            language=language,
            voice_rate=item.get("rate", "+0%"),
            volume=item.get("volume", "+0%"),
            pitch=item.get("pitch", "+0Hz"),
        )
    return keys


def latest_manifest(start_time: float) -> Path | None:
    return latest_artifact(PYVIDEOTRANS / "tmp", start_time, "**/openvoice-manifest-*.json")


def latest_queue(start_time: float) -> Path | None:
    return latest_artifact(PYVIDEOTRANS / "tmp", start_time, "**/openvoice-queue-*.json")


def latest_srt(start_time: float) -> Path | None:
    """Find the most recently created SRT file in the pyVideoTrans tmp root."""
    # pyVideoTrans writes SRTs to tmp/ with names like faster-YYYYMMDD-HH_MM_SS.srt
    return latest_artifact(PYVIDEOTRANS / "tmp", start_time, "**/*.srt")


def run_subprocess(cmd: list[str], cwd: Path, label: str,
                   env: dict | None = None) -> subprocess.CompletedProcess:
    print(f"[{label}] running: {' '.join(cmd[:2])} ...")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def run_openvoice_bridge_direct(
    queue_tts: list[dict],
    manifest_path: Path,
    queue_path: Path,
    work_dir: Path,
    logs_path: Path,
    preserve_dir: Path,
    openvoice_python: Path,
    bridge_script: Path,
    openvoice_repo: Path,
    checkpoint_dir: Path,
    language: str = "EN",
    device: str = "auto",
    allow_partial: bool = False,
    base_speaker: str = "",
) -> subprocess.CompletedProcess:
    """Run the OpenVoice bridge directly with a pre-built queue_tts.

    This bypasses pyVideoTrans's TTS layer entirely, giving us full control
    over per-speaker ref_wav routing. Each queue item must have a 'ref_wav'
    or 'voice_reference' key pointing to the speaker's reference.wav.
    """
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(queue_tts, ensure_ascii=False), encoding="utf-8",
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    preserve_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        openvoice_python.as_posix(),
        bridge_script.as_posix(),
        "--queue-tts-file", queue_path.as_posix(),
        "--manifest-file", manifest_path.as_posix(),
        "--work-dir", work_dir.as_posix(),
        "--openvoice-repo", openvoice_repo.as_posix(),
        "--checkpoint-dir", checkpoint_dir.as_posix(),
        "--language", language,
        "--logs-file", logs_path.as_posix(),
        "--preserve-dir", preserve_dir.as_posix(),
    ]
    if device:
        cmd.extend(["--device", device])
    if base_speaker:
        cmd.extend(["--base-speaker", base_speaker])
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        openvoice_repo.as_posix() + os.pathsep + env.get("PYTHONPATH", "")
    )
    result = run_subprocess(cmd, openvoice_repo, "openvoice-bridge-direct", env=env)
    # Bridge returns exit code 2 for partial success (some segments ok, some
    # failed). When allow_partial is True, treat exit code 2 as success (0).
    if allow_partial and result.returncode == 2:
        print("OpenVoice bridge: partial success (exit 2) accepted via --allow-partial")
        return subprocess.CompletedProcess(
            args=result.args, returncode=0, stdout=result.stdout, stderr=result.stderr,
        )
    return result


def run_qwen3_bridge_direct(
    queue_tts: list[dict],
    manifest_path: Path,
    queue_path: Path,
    work_dir: Path,
    logs_path: Path,
    preserve_dir: Path,
    bridge_script: Path,
    model_path: str,
    language: str = "auto",
    temperature: float = 0.9,
    speed: float = 1.0,
    allow_partial: bool = False,
) -> subprocess.CompletedProcess:
    """Run the Qwen3-TTS bridge directly with a pre-built queue_tts.

    This is the Qwen3 equivalent of run_openvoice_bridge_direct. It uses
    mlx-audio for on-device Apple Silicon inference with per-speaker
    voice cloning via ref_audio.
    """
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(queue_tts, ensure_ascii=False), encoding="utf-8",
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    preserve_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        bridge_script.as_posix(),
        "--queue-tts-file", queue_path.as_posix(),
        "--manifest-file", manifest_path.as_posix(),
        "--work-dir", work_dir.as_posix(),
        "--model-path", model_path,
        "--language", language,
        "--temperature", str(temperature),
        "--speed", str(speed),
        "--logs-file", logs_path.as_posix(),
        "--preserve-dir", preserve_dir.as_posix(),
    ]
    result = run_subprocess(cmd, ROOT, "qwen3-bridge-direct")
    # Bridge returns exit code 2 for partial success (some segments ok, some
    # failed). When allow_partial is True, treat exit code 2 as success (0).
    if allow_partial and result.returncode == 2:
        print("Qwen3-TTS bridge: partial success (exit 2) accepted via --allow-partial")
        return subprocess.CompletedProcess(
            args=result.args, returncode=0, stdout=result.stdout, stderr=result.stderr,
        )
    return result


def run_omnivoice_bridge_direct(
    queue_tts: list[dict],
    manifest_path: Path,
    queue_path: Path,
    work_dir: Path,
    logs_path: Path,
    preserve_dir: Path,
    bridge_script: Path,
    api_url: str,
    language: str = "auto",
    speed: float = 1.0,
    allow_partial: bool = False,
) -> subprocess.CompletedProcess:
    """Run the OmniVoice bridge directly with a pre-built queue_tts.

    The OmniVoice bridge runs on sys.executable (the same python that runs
    this wrapper) because it only needs requests/gradio_client.
    """
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        json.dumps(queue_tts, ensure_ascii=False), encoding="utf-8",
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    preserve_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        bridge_script.as_posix(),
        "--queue-tts-file", queue_path.as_posix(),
        "--manifest-file", manifest_path.as_posix(),
        "--work-dir", work_dir.as_posix(),
        "--api-url", api_url,
        "--language", language,
        "--speed", str(speed),
        "--logs-file", logs_path.as_posix(),
        "--preserve-dir", preserve_dir.as_posix(),
    ]
    result = run_subprocess(cmd, ROOT, "omnivoice-bridge-direct")
    # Bridge returns exit code 2 for partial success (some segments ok, some
    # failed). When allow_partial is True, treat exit code 2 as success (0).
    if allow_partial and result.returncode == 2:
        print("OmniVoice bridge: partial success (exit 2) accepted via --allow-partial")
        return subprocess.CompletedProcess(
            args=result.args, returncode=0, stdout=result.stdout, stderr=result.stderr,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dub a personal video with pyVideoTrans + OpenVoice V2"
    )
    parser.add_argument("--input", required=True,
                        help="absolute path to the input video (MP4)")
    parser.add_argument("--source-language", default="auto",
                        help="source language code (e.g. en, ko, auto)")
    parser.add_argument("--target-language", default="en",
                        help="target language code (e.g. en, es, zh-cn)")
    parser.add_argument("--reference", default="",
                        help="reference voice WAV for OpenVoice cloning "
                             "(default: auto-detect gender)")
    parser.add_argument("--output", required=True,
                        help="absolute path for the final dubbed MP4")
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo",
                        help="ASR model (tiny/base/small/medium/large-v3/"
                             "openai/whisper-large-v3-turbo)")
    parser.add_argument("--recogn-type", default="",
                        help="ASR provider index "
                             "(default: auto-select based on model)")
    parser.add_argument("--gender", default="auto",
                        help="voice gender: auto/male/female "
                             "(default: auto-detect from audio)")
    parser.add_argument("--vocal-separation-method",
                        choices=["ffmpeg", "demucs"], default="ffmpeg",
                        help="vocal removal: ffmpeg=center-pan (fast, default), "
                             "demucs=AI (better, requires model download)")
    parser.add_argument("--demucs-timeout", type=int, default=600,
                        help="timeout for Demucs vocal separation in seconds")
    parser.add_argument("--cpu-only", action="store_true",
                        help="force CPU mode for ASR "
                             "(useful when MPS runs out of memory)")
    parser.add_argument("--subtitle-type", default="1",
                        help="0=None 1=Hard 2=Soft 3=HardDual 4=SoftDual")
    parser.add_argument("--prefer-embedded-subtitles", action="store_true",
                        help="split pipeline: extract embedded subtitles as "
                             "the text+timing source before running ASR. "
                             "Falls back to whisper ASR when no usable "
                             "embedded subtitle stream is found. Embedded "
                             "subtitles give WHAT and WHEN; speaker identity "
                             "(WHO) still comes from audio diarization.")
    parser.add_argument("--no-ocr-image-subs", action="store_true",
                        help="with --prefer-embedded-subtitles, skip OCR for "
                             "image-based subtitles (PGS/VobSub) and go "
                             "straight to ASR fallback")
    parser.add_argument("--job-dir", default="",
                        help="working directory "
                             "(default: tmp/personal_dub/<timestamp>)")
    parser.add_argument("--force-overwrite-job-dir", action="store_true",
                        help="allow deleting an existing job dir; "
                             "only works for marked dirs under "
                             "tmp/personal_dub/")
    parser.add_argument("--allow-external-job-dir-delete",
                        action="store_true",
                        help=argparse.SUPPRESS)  # hidden debug escape hatch
    parser.add_argument("--allow-partial", action="store_true",
                        help="allow the final remux even if some segments failed")
    parser.add_argument("--background-volume", type=float, default=0.0,
                        help="mix original audio at this volume (0.0=off, 0.15=quiet bg, 1.0=full)")
    parser.add_argument("--voice-volume", type=float, default=1.0,
                        help="gain for dubbed speech (1.0=normal, 1.5=louder, 0.5=quieter)")
    parser.add_argument("--final-gain", type=float, default=1.0,
                        help="gain applied after normalization (1.0=no change)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip peak normalization (use with --final-gain for manual control)")
    parser.add_argument("--vocal-separation", action="store_true",
                        help="remove vocals from background audio (uses --vocal-separation-method)")
    parser.add_argument("--ducking", action="store_true",
                        help="lower background volume where dubbed speech is present")
    parser.add_argument("--target-lufs", type=float, default=None,
                        help="apply EBU R128 LUFS normalization (-16=web, -23=broadcast)")
    parser.add_argument("--background-timeout", type=int, default=300,
                        help="ffmpeg timeout for background extraction (seconds)")
    parser.add_argument("--lufs-timeout", type=int, default=600,
                        help="ffmpeg timeout for LUFS normalization (seconds)")
    parser.add_argument("--fail-if-background-mix-fails", action="store_true",
                        help="fail if background audio mixing fails "
                             "instead of silent speech-only output")
    parser.add_argument("--allow-demucs-download", action="store_true",
                        help="allow Demucs to download its model at runtime "
                             "(default: refuse to download, fall back to ffmpeg)")
    parser.add_argument("--no-demucs-download", action="store_true",
                        help="explicitly refuse runtime Demucs model download "
                             "(this is the default behavior)")
    parser.add_argument("--legacy-artifact-scan", action="store_true",
                        help="DEPRECATED: use global mtime scan to find "
                             "manifest/queue/SRT instead of explicit paths. "
                             "Only for debugging missing artifacts.")
    # --- Speaker profiling (v0.8) ---
    parser.add_argument("--speaker-profiling", action="store_true",
                        help="enable per-speaker profiling: diarization + "
                             "reference extraction + age/gender/pitch per speaker")
    parser.add_argument("--speaker-diarization",
                        choices=["pyannote", "built", "none"],
                        default="pyannote",
                        help="diarization backend (default: pyannote when "
                             "profiling is on)")
    parser.add_argument("--num-speakers", default="auto",
                        help="auto or a positive integer for diarization")
    parser.add_argument("--speaker-profile-json", default="",
                        help="reuse an existing speaker_profiles.json instead "
                             "of re-running diarization")
    parser.add_argument("--hf-token", default="",
                        help="HuggingFace token for pyannote (or set HF_TOKEN env)")
    parser.add_argument("--min-clip-seconds", type=float, default=3.0,
                        help="minimum reference clip duration per speaker")
    parser.add_argument("--max-clip-seconds", type=float, default=8.0,
                        help="maximum reference clip duration per speaker")
    # --- TTS engine selection (Blocker 4) ---
    parser.add_argument("--tts-engine", choices=list(TTS_ENGINE_MAP.keys()),
                        default=DEFAULT_TTS_ENGINE,
                        help="primary TTS engine (default: openvoice)")
    parser.add_argument("--fallback-tts-engine",
                        choices=list(TTS_ENGINE_MAP.keys()) + ["none"],
                        default="none",
                        help="fallback TTS engine if primary fails (default: none)")
    # --- OpenVoice options (only used when --tts-engine openvoice) ---
    parser.add_argument("--base-speaker", default="",
                        help="OpenVoice base speaker (e.g. EN-US, EN-NEWEST, "
                             "EN-DEFAULT). Empty = first available speaker.")
    # --- Qwen3-TTS options (only used when --tts-engine qwen3-local) ---
    parser.add_argument("--qwen3-model", default=QWEN3_DEFAULT_MODEL,
                        help=f"Qwen3-TTS model path or HF repo ID "
                             f"(default: {QWEN3_DEFAULT_MODEL})")
    parser.add_argument("--qwen3-temperature", type=float, default=0.9,
                        help="Qwen3-TTS sampling temperature (default: 0.9)")
    parser.add_argument("--qwen3-speed", type=float, default=1.0,
                        help="Qwen3-TTS speech speed multiplier (default: 1.0)")
    # --- OmniVoice options (only used when --tts-engine omnivoice) ---
    parser.add_argument("--omnivoice-url", default="",
                        help="OmniVoice server URL (e.g. http://localhost:3900 "
                             "for studio mode or http://localhost:7860 for "
                             "Gradio mode). Required in split mode.")
    # --- Shared TTS options ---
    parser.add_argument("--tts-speed", type=float, default=1.0,
                        help="speech speed multiplier for engines that support "
                             "it (default: 1.0)")
    # --- Post-generation pitch verification (Blocker 6) ---
    parser.add_argument("--verify-pitch", action="store_true",
                        help="run post-generation pitch verification against "
                             "speaker profiles and flag mismatches")
    parser.add_argument("--fail-on-pitch-mismatch", action="store_true",
                        help="fail the job if any segment is flagged needs_review "
                             "or rewrite_shorter by pitch verification")
    parser.add_argument("--semitone-threshold", type=float, default=2.0,
                        help="pitch mismatch threshold in semitones")
    parser.add_argument("--voiced-ratio-threshold", type=float, default=0.3,
                        help="flag segments with voiced_ratio below this")
    parser.add_argument("--duration-ratio-threshold", type=float, default=1.35,
                        help="flag rewrite_shorter if gen/target duration exceeds this")
    # --- Age model plugin (v0.12) ---
    parser.add_argument("--age-model", choices=("auto", "on", "off"),
                        default="auto",
                        help="age estimation: auto=try real model then fall back, "
                             "on=require model, off=pitch heuristic only")
    parser.add_argument("--age-model-path", default="",
                        help="local age model dir, e.g. "
                             "models/age_reg_ann_ecapa_librosa_combined "
                             "(preferred over the HF repo id when present)")
    parser.add_argument("--reference-clip-scoring", choices=("auto", "on", "off"),
                        default="auto",
                        help="reference clip selection: on=quality-scored, "
                             "off=legacy, auto=on when librosa available")
    # --- Character profiles (v0.12) ---
    parser.add_argument("--character-profiles", action="store_true",
                        help="also write character_profiles.json alongside "
                             "speaker_profiles.json (named, reviewable characters)")
    # --- Cache / resume (v0.12) ---
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing job: reuse its job dir and skip "
                             "stages already marked pass in job.json")
    parser.add_argument("--from-stage", default="",
                        help="start at a given stage (subs/stt/sts/profile/tts/"
                             "audio/remux/verify). Implies skipping earlier stages.")
    parser.add_argument("--only-segment", default="",
                        help="only (re)generate a single segment id; then rebuild "
                             "audio + remux. Implies --from-stage tts.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip TTS segments whose output WAV already exists "
                             "(incremental regeneration)")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the hash-based segment TTS cache. By "
                             "default, segments whose inputs (source audio, "
                             "subtitle, speaker profile, translated text, TTS "
                             "engine/model, reference voice) are unchanged are "
                             "reused from a previous run of the same job.")
    args = parser.parse_args()

    input_video = Path(args.input).expanduser().resolve()
    output_video = Path(args.output).expanduser().resolve()
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"

    # --- Validate inputs ---
    if not input_video.is_file():
        return die(f"input video missing: {input_video}")
    if not py_python.is_file():
        return die(f"pyVideoTrans venv python missing: {py_python}")
    # OpenVoice is required only when it is actually in use. The original
    # VTV pipeline always uses OpenVoice (provider 34), so it is required
    # there. The split pipeline (--speaker-profiling) routes TTS through a
    # direct bridge; OpenVoice is only required there when it is the
    # primary or fallback engine. Qwen3-local-only runs must NOT be
    # blocked by a missing OpenVoice install.
    use_split_pipeline_pre = bool(args.speaker_profiling or args.speaker_profile_json)
    # --prefer-embedded-subtitles only has effect in the split pipeline
    # (Step 0 lives inside the split-pipeline block). Reject it in VTV
    # mode rather than silently ignoring it.
    if args.prefer_embedded_subtitles and not use_split_pipeline_pre:
        return die(
            "--prefer-embedded-subtitles requires --speaker-profiling "
            "(or --speaker-profile-json). The embedded-subtitle extraction "
            "step is part of the split pipeline; the VTV pipeline manages "
            "its own subtitles internally."
        )
    openvoice_in_use = (
        not use_split_pipeline_pre
        or args.tts_engine == "openvoice"
        or args.fallback_tts_engine == "openvoice"
    )
    if openvoice_in_use:
        if not (OPENVOICE / ".venv" / "bin" / "python").is_file():
            return die(f"OpenVoice venv python missing: {OPENVOICE / '.venv' / 'bin' / 'python'}")
        if not (OPENVOICE / "checkpoints_v2" / "converter" / "checkpoint.pth").is_file():
            return die("OpenVoice checkpoints missing. Run: make download-openvoice")
    # Qwen3-local pre-flight (only when it is the primary or fallback
    # engine). The bridge runs on sys.executable, so the required
    # packages (mlx_audio, soundfile, librosa) must be importable from
    # this python. The model must also be resolvable.
    qwen3_in_use = (
        use_split_pipeline_pre
        and (args.tts_engine == "qwen3-local"
             or args.fallback_tts_engine == "qwen3-local")
    )
    if qwen3_in_use:
        qwen3_err = _qwen3_preflight(args.qwen3_model)
        if qwen3_err:
            return die(qwen3_err)
    omnivoice_in_use = (
        use_split_pipeline_pre
        and (args.tts_engine == "omnivoice"
             or args.fallback_tts_engine == "omnivoice")
    )
    if omnivoice_in_use:
        ov_err = _omnivoice_preflight(args.omnivoice_url)
        if ov_err:
            return die(ov_err)
    if not BUILD_AUDIO_SCRIPT.is_file() or not REMUX_SCRIPT.is_file():
        return die("build/remux scripts missing from scripts/")

    # --- Select reference voice (auto-detect gender if needed) ---
    detected_gender = "unknown"
    detected_f0 = 0.0
    if args.reference:
        # User explicitly specified a reference voice
        reference = Path(args.reference).expanduser().resolve()
        if not reference.is_file():
            return die(f"reference WAV missing: {reference}")
    elif args.gender == "male":
        reference = MALE_REFERENCE
        if not reference.is_file():
            return die(
                f"male reference voice missing: {reference}. "
                "Run: python scripts/generate_reference_voices.py"
            )
        detected_gender = "male"
    elif args.gender == "female":
        reference = FEMALE_REFERENCE
        if not reference.is_file():
            return die(
                f"female reference voice missing: {reference}. "
                "Run: python scripts/generate_reference_voices.py"
            )
        detected_gender = "female"
    else:
        # Auto-detect gender from the original audio
        reference = DEFAULT_REFERENCE
        if GENDER_DETECT_SCRIPT.is_file():
            print("Detecting speaker gender from audio...")
            try:
                # Run gender detection in the pyVideoTrans venv (has librosa)
                gender_result = subprocess.run(
                    [py_python.as_posix(), GENDER_DETECT_SCRIPT.as_posix(),
                     "--input", input_video.as_posix(), "--json"],
                    capture_output=True, text=True, timeout=120,
                )
                if gender_result.returncode == 0:
                    import json as _json
                    gender_data = _json.loads(gender_result.stdout.strip())
                    detected_gender = gender_data.get("gender", "unknown")
                    detected_f0 = gender_data.get("median_f0_hz", 0.0)
                    print(f"  detected gender: {detected_gender} (F0: {detected_f0:.1f} Hz)")
                    if detected_gender == "male" and MALE_REFERENCE.is_file():
                        reference = MALE_REFERENCE
                    elif detected_gender == "female" and FEMALE_REFERENCE.is_file():
                        reference = FEMALE_REFERENCE
            except Exception as e:
                print(f"  gender detection failed: {e}, using default reference")
        if not reference.is_file():
            reference = DEFAULT_REFERENCE
        if not reference.is_file():
            return die(
                f"reference WAV missing: {reference}. "
                "Run: python scripts/generate_reference_voices.py"
            )

    # --- Set up job dir (safe deletion policy: Blocker 3) ---
    # --resume: reuse an existing job dir + job.json instead of creating a
    # fresh one. The job dir must already exist and have a job.json.
    resume_plan: ResumePlan | None = None
    if args.resume or args.from_stage or args.only_segment:
        candidate = Path(args.job_dir).expanduser().resolve() if args.job_dir else None
        if candidate and candidate.is_dir() and (candidate / "job.json").is_file():
            job_dir = candidate
        elif args.job_dir:
            return die(
                f"--resume/--from-stage/--only-segment requires an existing "
                f"job dir with job.json: {candidate}"
            )
        else:
            # No --job-dir given: pick the most recent job dir.
            pdub = PERSONAL_DUB_ROOT
            job_dirs = sorted(
                [d for d in pdub.glob("*") if d.is_dir() and (d / "job.json").is_file()],
                key=lambda d: d.stat().st_mtime, reverse=True,
            )
            if not job_dirs:
                return die(
                    "--resume/--from-stage/--only-segment requires an existing "
                    f"job dir under {pdub}; none found."
                )
            job_dir = job_dirs[0]
            print(f"Resume: reusing most recent job dir {job_dir}")
        # Load existing job.json to build the resume plan.
        existing_state = read_job_state(job_dir / "job.json") or {}
        try:
            from_stage = parse_from_stage(args.from_stage) if args.from_stage else None
        except ValueError as exc:
            return die(str(exc))
        resume_plan = ResumePlan(
            existing_state.get("stages", {}),
            resume=args.resume,
            from_stage=from_stage,
            only_segment=args.only_segment or None,
            skip_existing=args.skip_existing,
        )
        job_id = existing_state.get("job_id", "resumed")
        print(f"Resume plan: {resume_plan.summary()}")
    else:
        job_id = generate_job_id()
        if args.job_dir:
            job_dir = Path(args.job_dir).expanduser().resolve()
        else:
            job_dir = PERSONAL_DUB_ROOT / job_id
        try:
            create_job_dir(
                job_dir,
                force=args.force_overwrite_job_dir,
                allow_external=args.allow_external_job_dir_delete,
            )
        except UnsafeJobDirError as exc:
            return die(str(exc))
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_audio = job_dir / "generated_audio"
    stable_manifest = job_dir / "openvoice_manifest.json"
    dubbed_audio = job_dir / "dubbed_audio.wav"
    report_path = job_dir / "report.json"
    job_json_path = job_dir / "job.json"
    # The canonical final video lives in the job dir so regeneration can
    # rebuild it without losing the only good copy. We sync to the user's
    # requested output path after the remux succeeds.
    final_video_job = job_dir / "final_dubbed.mp4"
    review_path = job_dir / "review_segments.json"
    remux_path = job_dir / "remux_command.json"
    # Job-scoped subtitles directory (Blocker 4): copy SRTs here instead of
    # relying on global mtime scanning.
    subtitles_dir = job_dir / "subtitles"
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    job_srt = subtitles_dir / "translated.srt"

    # Job-scoped explicit paths (Blocker 5): pass exact file paths to the
    # OpenVoice provider so we never need global mtime scanning.
    openvoice_queue = job_dir / "openvoice_queue.json"
    openvoice_manifest_explicit = job_dir / "openvoice_manifest_explicit.json"
    openvoice_logs = job_dir / "openvoice_bridge.log"
    openvoice_work = job_dir / "openvoice_work"
    # Speaker profiling outputs (v0.8)
    speakers_dir = job_dir / "speakers"
    speaker_profiles_path = speakers_dir / "speaker_profiles.json"
    pitch_verification_path = job_dir / "pitch_verification.json"
    enriched_manifest_path = job_dir / "openvoice_manifest_enriched.json"
    # Character profiles output (v0.12)
    character_profiles_path = job_dir / "character_profiles.json"

    # --- Speaker profiling setup (v0.9: split pipeline) ---
    # When --speaker-profiling is on, we use a SPLIT pipeline:
    #   1. STT (cli.py --task stt) → source SRT with real segment timings
    #   2. STS (cli.py --task sts) → translated SRT
    #   3. analyze_speakers.py --segments-json → speaker_profiles.json with
    #      real segment_assignments (matched to actual SRT timings)
    #   4. Direct OpenVoice bridge with per-speaker ref_wav
    #   5. Build dubbed audio + remux (existing code)
    #
    # When --speaker-profiling is off, we use the original VTV pipeline.
    #
    # If a profile is already provided via --speaker-profile-json, we still
    # need the source SRT to validate segment_assignments. If the profile has
    # no segment_assignments, we fail hard (v0.9: no more fake progress).
    speaker_profiles_data: dict | None = None
    use_split_pipeline = bool(args.speaker_profiling or args.speaker_profile_json)
    if use_split_pipeline:
        # The split pipeline uses bridge scripts directly for per-speaker
        # ref_wav routing. Only engines with a bridge implementation are
        # The split pipeline uses bridge scripts directly for per-speaker
        # ref_wav routing. Only engines with a bridge implementation are
        # supported here.
        SUPPORTED_SPLIT_ENGINES = {"openvoice", "qwen3-local", "omnivoice"}
        if args.tts_engine not in SUPPORTED_SPLIT_ENGINES:
            return die(
                f"--tts-engine '{args.tts_engine}' is not supported in "
                f"speaker-profiling mode. The split pipeline requires a "
                f"bridge script for per-speaker ref_wav routing, but "
                f"'{args.tts_engine}' has no bridge implementation. "
                f"Either drop --speaker-profiling / --speaker-profile-json "
                f"to use the VTV pipeline (which supports all engines), or "
                f"use --tts-engine openvoice, --tts-engine qwen3-local, or "
                f"--tts-engine omnivoice. "
                f"Supported engines in split mode: "
                f"{', '.join(sorted(SUPPORTED_SPLIT_ENGINES))}."
            )
        if args.fallback_tts_engine != "none" and args.fallback_tts_engine not in SUPPORTED_SPLIT_ENGINES:
            return die(
                f"--fallback-tts-engine '{args.fallback_tts_engine}' is not "
                f"supported in speaker-profiling mode for the same reason. "
                f"Use --fallback-tts-engine none, openvoice, qwen3-local, or "
                f"omnivoice. "
                f"Supported: {', '.join(sorted(SUPPORTED_SPLIT_ENGINES))}."
            )
        speakers_dir.mkdir(parents=True, exist_ok=True)
        if args.speaker_profile_json:
            src_profile = Path(args.speaker_profile_json).expanduser().resolve()
            if not src_profile.is_file():
                return die(f"speaker profile JSON not found: {src_profile}")
            shutil.copy2(src_profile, speaker_profiles_path)
            print(f"Speaker profiles (reused): {speaker_profiles_path}")

    # --- Initialize job.json (Blocker 12) ---
    job_state = initial_job_state(
        job_id=job_id,
        input_video=input_video.as_posix(),
        output_video=output_video.as_posix(),
        reference_voice=reference.as_posix(),
        source_language=args.source_language,
        target_language=args.target_language,
        job_dir=job_dir.as_posix(),
    )
    job_state["paths"].update({
        "openvoice_manifest": stable_manifest.as_posix(),
        "openvoice_queue": openvoice_queue.as_posix(),
        "openvoice_logs": openvoice_logs.as_posix(),
        "dubbed_audio": dubbed_audio.as_posix(),
        "final_video": final_video_job.as_posix(),
        "review_segments": review_path.as_posix(),
        "subtitles_dir": subtitles_dir.as_posix(),
        "speakers_dir": speakers_dir.as_posix(),
        "speaker_profiles": speaker_profiles_path.as_posix(),
        "pitch_verification": pitch_verification_path.as_posix(),
        "enriched_manifest": enriched_manifest_path.as_posix(),
    })
    if speaker_profiles_data:
        job_state["speaker_profiling"] = True
        job_state["tts_engine"] = args.tts_engine
        job_state["fallback_tts_engine"] = args.fallback_tts_engine
    write_job_state(job_json_path, job_state)

    print(f"Input:    {input_video}")
    print(f"Output:   {output_video}")
    print(f"Reference: {reference}")
    if detected_gender != "unknown":
        print(f"Gender:   {detected_gender} (F0: {detected_f0:.1f} Hz)")
    print(f"TTS engine: {args.tts_engine} (provider {TTS_ENGINE_MAP[args.tts_engine]})")
    if args.fallback_tts_engine != "none":
        print(f"Fallback:   {args.fallback_tts_engine}")
    print(f"Model:    {args.model}")
    print(f"Source:   {args.source_language}")
    print(f"Target:   {args.target_language}")
    print(f"Job dir:  {job_dir}")

    # --- Determine ASR provider type ---
    if args.recogn_type:
        recogn_type = args.recogn_type
    elif "/" in args.model or args.model.startswith("whisper-large"):
        recogn_type = HUGGINGFACE_ASR_PROVIDER
    else:
        recogn_type = "0"  # faster-whisper

    # Build environment for subprocess
    pyvt_env = dict(os.environ)
    if args.cpu_only:
        pyvt_env["PYTORCH_MPS_DISABLE"] = "1"
        pyvt_env["CUDA_VISIBLE_DEVICES"] = ""

    # ====================================================================
    # SPLIT PIPELINE (when --speaker-profiling is active)
    # ====================================================================
    # STT → STS → analyze_speakers(with real timings) → direct OpenVoice bridge
    # This bypasses pyVideoTrans's TTS layer entirely for per-speaker routing.
    # ====================================================================
    srt_file: Path | None = None
    if use_split_pipeline:
        # --- Resume: decide which early stages to skip ---
        # When --from-stage / --resume is set, reuse existing artifacts for
        # any stage the plan says to skip, instead of re-running STT/STS/
        # diarization. Each skipped stage requires its artifact to exist.
        def _stage_skipped(stage: str) -> bool:
            return resume_plan is not None and not resume_plan.should_run_stage(stage)

        skip_subs = _stage_skipped("subtitle_extraction")
        skip_sts = _stage_skipped("translation")
        skip_profile = _stage_skipped("speaker_profiling")
        # STT (transcription) is implicitly skipped when source_srt is reused
        # above, since the STT step only runs when source_srt is None.

        # --- Step 0: embedded subtitles (text + timing source) ---
        # Embedded subtitles give WHAT was said and WHEN. They never carry
        # speaker identity — WHO comes from diarization in Step 3. When
        # --prefer-embedded-subtitles is set, try to extract an embedded
        # subtitle stream as the source SRT before paying for ASR. If no
        # usable stream is found (none, image-based OCR failed, or OCR
        # disabled), fall through to whisper ASR below.
        source_srt: Path | None = None
        subtitle_source_note = "asr"
        # Resume: reuse an already-extracted source SRT if present.
        if skip_subs and (subtitles_dir / "source.srt").is_file():
            source_srt = subtitles_dir / "source.srt"
            subtitle_source_note = "reused"
            print(f"Resume: reusing source SRT {source_srt}")
        elif args.prefer_embedded_subtitles:
            update_job_stage(job_json_path, "subtitle_extraction", "running")
            try:
                sub_result = _extract_embedded_subtitles(
                    video=input_video,
                    output_dir=subtitles_dir,
                    source_language=args.source_language,
                    ocr_image_subs=not args.no_ocr_image_subs,
                    ffprobe=local_ffprobe(),
                    ffmpeg=shutil.which("ffmpeg") or None,
                )
            except Exception as exc:
                update_job_stage(job_json_path, "subtitle_extraction", "fail")
                print(f"Embedded subtitle extraction error: {exc}")
                sub_result = None
            if sub_result and sub_result.srt_path:
                extracted = Path(sub_result.srt_path)
                # Validate the extracted SRT actually has cues before using
                # it — a malformed/empty extract must not silently replace
                # ASR.
                cues = parse_srt_to_segments(extracted)
                if cues:
                    source_srt = subtitles_dir / "source.srt"
                    if extracted != source_srt:
                        shutil.copy2(extracted, source_srt)
                    subtitle_source_note = sub_result.source or "embedded"
                    print(
                        f"Embedded subtitles: {sub_result.note} "
                        f"({len(cues)} cues) — skipping ASR"
                    )
                    update_job_stage(
                        job_json_path, "subtitle_extraction", "pass",
                        extra={"subtitle_source": subtitle_source_note},
                    )
                else:
                    print(
                        f"Embedded subtitle extract had no cues "
                        f"({sub_result.note}); falling back to ASR"
                    )
                    update_job_stage(
                        job_json_path, "subtitle_extraction", "skip",
                        extra={"subtitle_source": "asr"},
                    )
            else:
                note = sub_result.note if sub_result else "no result"
                print(f"Embedded subtitles: {note}; falling back to ASR")
                update_job_stage(
                    job_json_path, "subtitle_extraction", "skip",
                    extra={"subtitle_source": "asr"},
                )

        # --- Step 1: STT (speech-to-text) → source SRT ---
        # Only run ASR when embedded subtitles did not provide a source SRT.
        if source_srt is None:
            update_job_stage(job_json_path, "transcription", "running")
            stt_output = job_dir / "stt_output"
            stt_output.mkdir(parents=True, exist_ok=True)
            stt_cmd = [
                py_python.as_posix(), "cli.py",
                "--task", "stt",
                "--name", input_video.as_posix(),
                "--recogn_type", recogn_type,
                "--model_name", args.model,
                "--detect_language", args.source_language,
                "--output-dir", stt_output.as_posix(),
                "--no-clear-cache",
                "--verbose",
            ]
            stt_result = run_subprocess(stt_cmd, PYVIDEOTRANS, "stt", env=pyvt_env)
            if stt_result.returncode != 0:
                update_job_stage(job_json_path, "transcription", "fail")
                return die(f"STT failed: {stt_result.stderr[-1500:]}")
            # Find the source SRT in stt_output
            source_srt = stt_output / f"{input_video.stem}.srt"
            if not source_srt.is_file():
                # Fallback: scan for any SRT
                srts = list(stt_output.glob("*.srt"))
                if not srts:
                    update_job_stage(job_json_path, "transcription", "fail")
                    return die(f"STT produced no SRT in {stt_output}")
                source_srt = srts[0]
            shutil.copy2(source_srt, subtitles_dir / "source.srt")
            print(f"Source SRT (ASR): {source_srt}")
            update_job_stage(job_json_path, "transcription", "pass")
        else:
            # Embedded subtitles already produced source.srt; nothing to do.
            print(f"Source SRT (embedded): {source_srt}")

        # --- Step 2: STS (subtitle translation) → translated SRT ---
        if skip_sts and job_srt.is_file():
            # Resume: reuse the existing translated SRT.
            srt_file = job_srt
            print(f"Resume: reusing translated SRT {job_srt}")
            update_job_stage(job_json_path, "translation", "pass")
        else:
            update_job_stage(job_json_path, "translation", "running")
            sts_output = job_dir / "sts_output"
            sts_output.mkdir(parents=True, exist_ok=True)
            sts_cmd = [
                py_python.as_posix(), "cli.py",
                "--task", "sts",
                "--name", source_srt.as_posix(),
                "--translate_type", "0",
                "--source_language_code", args.source_language,
                "--target_language_code", args.target_language,
                "--output-dir", sts_output.as_posix(),
                "--no-clear-cache",
                "--verbose",
            ]
            sts_result = run_subprocess(sts_cmd, PYVIDEOTRANS, "sts", env=pyvt_env)
            if sts_result.returncode != 0:
                update_job_stage(job_json_path, "translation", "fail")
                return die(f"STS failed: {sts_result.stderr[-1500:]}")
            # Find the translated SRT
            # STS names it {stem}.{target_language_code}.srt where
            # target_language_code is the raw value we passed (e.g. "zh-cn").
            # Try the raw code first, then the normalized (no hyphen) version.
            translated_srt = sts_output / f"{source_srt.stem}.{args.target_language}.srt"
            if not translated_srt.is_file():
                lang_code = args.target_language.lower().replace("-", "")
                translated_srt = sts_output / f"{source_srt.stem}.{lang_code}.srt"
            if not translated_srt.is_file():
                srts = list(sts_output.glob("*.srt"))
                if not srts:
                    update_job_stage(job_json_path, "translation", "fail")
                    return die(f"STS produced no SRT in {sts_output}")
                translated_srt = srts[0]
            shutil.copy2(translated_srt, job_srt)
            srt_file = job_srt
            print(f"Translated SRT: {translated_srt}")
            update_job_stage(job_json_path, "translation", "pass")

        # --- Step 2b: validate source/translated segment-count alignment ---
        # Speaker assignments from analyze_speakers are indexed by
        # segment_index from the SOURCE SRT. build_queue_tts_with_speakers
        # looks up speakers by the same index in the TRANSLATED SRT. This
        # only works if STS preserved the cue count and order.
        # pyVideoTrans check_target_sub() normally forces the translated
        # count back to the source count (reconstructing with source
        # timecodes), but LLM re-sentence can break this. Fail hard here
        # rather than silently misassigning speakers to wrong lines.
        source_seg_count = len(parse_srt_to_segments(source_srt))
        translated_seg_count = len(parse_srt_to_segments(job_srt))
        if source_seg_count != translated_seg_count:
            update_job_stage(job_json_path, "translation", "fail")
            return die(
                f"STS changed segment count: source={source_seg_count}, "
                f"translated={translated_seg_count}. Speaker assignments "
                f"are indexed by source segment_index and would misalign "
                f"with the translated cues. This usually means LLM "
                f"re-sentence is enabled in pyVideoTrans config — disable "
                f"it for the split pipeline, or re-run without "
                f"--speaker-profiling to use the VTV pipeline."
            )

        # --- Step 3: analyze_speakers with real segment timings ---
        # Parse the source SRT to get segments with real start/end times,
        # then run diarization + profiling with those segments.
        # Resume: skip diarization when the plan says to and profiles exist.
        if skip_profile and speaker_profiles_path.is_file() and not args.speaker_profile_json:
            print(f"Resume: reusing speaker profiles {speaker_profiles_path}")
            update_job_stage(job_json_path, "speaker_profiling", "pass")
        elif not args.speaker_profile_json:
            if not ANALYZE_SPEAKERS_SCRIPT.is_file():
                return die(f"analyze_speakers.py missing: {ANALYZE_SPEAKERS_SCRIPT}")
            segments = parse_srt_to_segments(source_srt)
            if not segments:
                return die(f"source SRT has no segments: {source_srt}")
            segments_json_path = speakers_dir / "segments.json"
            segments_json_path.write_text(
                json.dumps(segments, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Running speaker profiling with {len(segments)} segments...")
            analyze_cmd = [
                py_python.as_posix(),
                ANALYZE_SPEAKERS_SCRIPT.as_posix(),
                "--input", input_video.as_posix(),
                "--output-dir", speakers_dir.as_posix(),
                "--diarization", args.speaker_diarization,
                "--num-speakers", args.num_speakers,
                "--min-clip-seconds", str(args.min_clip_seconds),
                "--max-clip-seconds", str(args.max_clip_seconds),
                "--segments-json", segments_json_path.as_posix(),
                "--age-model", args.age_model,
                "--reference-clip-scoring", args.reference_clip_scoring,
            ]
            if args.age_model_path:
                analyze_cmd.extend(["--age-model-path", args.age_model_path])
            if args.hf_token:
                analyze_cmd.extend(["--hf-token", args.hf_token])
            analyze_result = run_subprocess(
                analyze_cmd, ROOT, "analyze-speakers",
            )
            if analyze_result.returncode != 0 or not speaker_profiles_path.is_file():
                return die(
                    f"speaker profiling failed: "
                    f"{analyze_result.stderr[-1500:]}"
                )
        # Load and validate speaker profiles
        try:
            speaker_profiles_data = read_json(speaker_profiles_path)
        except Exception as exc:
            return die(f"failed to read speaker_profiles.json: {exc}")
        speakers = speaker_profiles_data.get("speakers", [])
        if not speakers:
            return die("speaker profiling produced no speakers")
        # v0.9: segment_assignments is MANDATORY. No more fake progress.
        segment_assignments = speaker_profiles_data.get("segment_assignments", [])
        if not segment_assignments:
            return die(
                "speaker_profiles.json has no segment_assignments. "
                "This means speaker profiling ran without real segment "
                "timings. Re-run without --speaker-profile-json or ensure "
                "the profile was generated with --segments-json."
            )
        # Validate segment_assignments count matches the source SRT segments.
        # If the profile was generated from a different video/SRT, the indices
        # won't align and per-speaker routing will be wrong.
        source_segments = parse_srt_to_segments(source_srt)
        if source_segments and len(segment_assignments) != len(source_segments):
            print(
                f"WARNING: segment_assignments count ({len(segment_assignments)}) "
                f"does not match source SRT segment count ({len(source_segments)}). "
                f"This may indicate the profile was generated from a different "
                f"video. Per-speaker routing may be incorrect."
            )
        print(f"  {len(speakers)} speaker(s) profiled, "
              f"{len(segment_assignments)} segment assignments")
        for spk in speakers:
            g = spk.get("gender", {})
            a = spk.get("age", {})
            p = spk.get("pitch", {})
            print(
                f"  {spk['speaker_id']}: {g.get('label','?')} "
                f"({g.get('confidence',0):.2f}), {a.get('band','?')}, "
                f"F0={p.get('median_f0_hz',0):.1f}Hz"
            )

        # --- Build character_profiles.json (v0.12) ---
        # Upgrade speaker profiles to named, reviewable characters. Idempotent:
        # preserves user edits (names, voice locks, review_status) across runs.
        character_profiles_path = job_dir / "character_profiles.json"
        if args.character_profiles:
            existing_cp = _load_character_profiles(character_profiles_path)
            cp_data = _build_character_profiles(
                speaker_profiles_data,
                tts_engine=args.tts_engine,
                existing=existing_cp,
            )
            _write_character_profiles(
                character_profiles_path, cp_data,
                source=speaker_profiles_path.as_posix(),
            )
            job_state["paths"]["character_profiles"] = character_profiles_path.as_posix()
            write_job_state(job_json_path, job_state)
            print(f"Character profiles: {character_profiles_path} "
                  f"({len(cp_data['characters'])} characters)")

        # --- Step 4: Build queue_tts with per-speaker ref_wav ---
        update_job_stage(job_json_path, "tts", "running")
        # Compute the pyVideoTrans cache folder for output WAV paths
        try:
            stat = input_video.stat()
            uuid_input = f"{input_video.as_posix()}-{stat.st_size}-{stat.st_mtime}"
            pyvt_uuid = hashlib.md5(uuid_input.encode()).hexdigest()[:10]
            cache_folder = PYVIDEOTRANS / "tmp" / pyvt_uuid
        except Exception:
            cache_folder = job_dir / "tts_cache"
        cache_folder.mkdir(parents=True, exist_ok=True)

        # Load character profiles for voice-lock routing. A locked voice in
        # character_profiles.json overrides the speaker profile's
        # reference_audio during queue building (v0.12).
        cp_for_routing: dict | None = None
        if character_profiles_path.is_file():
            cp_for_routing = _load_character_profiles(character_profiles_path)

        tts_provider_int = int(TTS_ENGINE_MAP[args.tts_engine])
        queue_tts = build_queue_tts_with_speakers(
            translated_srt=job_srt,
            speaker_profiles=speaker_profiles_data,
            cache_folder=cache_folder,
            tts_type=tts_provider_int,
            language=args.target_language,
            character_profiles=cp_for_routing,
        )
        if not queue_tts:
            update_job_stage(job_json_path, "tts", "fail")
            return die("no TTS queue items built from translated SRT + speaker profiles")
        print(f"Built {len(queue_tts)} TTS queue items with per-speaker ref_wav")

        # --- Hash-based segment cache (v0.12) ---
        # Segments whose inputs (source audio, subtitle, speaker profile,
        # translated text, TTS engine/model, reference voice) are unchanged
        # are reused from a previous run of the same job. This is stronger
        # than --skip-existing (which only checks file existence): changing
        # one segment's text invalidates only that segment, not all.
        seg_cache: SegmentCache | None = None
        cache_keys: dict[str, str] = {}
        cache_hit_ids: set[str] = set()
        if not args.no_cache:
            seg_cache = SegmentCache(job_dir / "cache")
            try:
                source_audio_hash = content_hash(input_video)
            except Exception:
                source_audio_hash = "unknown"
            try:
                subtitle_hash = content_hash(source_srt) if (source_srt and source_srt.is_file()) else "none"
            except Exception:
                subtitle_hash = "none"
            speaker_profile_hash = dict_hash(speaker_profiles_data)
            tts_model_id = _tts_model_id(args.tts_engine, args)
            cache_keys = _compute_segment_cache_keys(
                queue_tts,
                source_audio_hash=source_audio_hash,
                subtitle_hash=subtitle_hash,
                speaker_profile_hash=speaker_profile_hash,
                tts_engine=args.tts_engine,
                tts_model=tts_model_id,
                language=args.target_language,
            )
            # Check each segment for a cache hit. On hit, copy the cached WAV
            # to the expected filename so downstream code finds it.
            for item in queue_tts:
                seg_id = str(item.get("line", item.get("id", "")))
                key = cache_keys.get(seg_id)
                if not key:
                    continue
                entry = seg_cache.get(key)
                if entry:
                    cached_wav = Path(entry["output_wav"])
                    dest = Path(item.get("filename", ""))
                    if cached_wav.is_file() and dest != cached_wav:
                        try:
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(cached_wav, dest)
                        except Exception as exc:
                            print(f"WARNING: cache copy failed for segment "
                                  f"{seg_id}: {exc}", file=sys.stderr)
                    cache_hit_ids.add(seg_id)
            if cache_hit_ids:
                print(f"Segment cache: {len(cache_hit_ids)} hit(s), "
                      f"{len(queue_tts) - len(cache_hit_ids)} miss(es)")

        # --- Cache / resume: filter the queue to segments that need (re)gen ---
        # --skip-existing drops segments whose output WAV already exists.
        # --only-segment keeps just one segment id.
        # Hash-based cache hits (above) are also skipped.
        # Filtered-out segments are not sent to the bridge; their existing
        # manifest entries are preserved and merged back after the run.
        full_queue = queue_tts
        existing_manifest_for_merge: dict | None = None
        if openvoice_manifest_explicit.is_file():
            try:
                existing_manifest_for_merge = read_json(openvoice_manifest_explicit)
            except Exception:
                existing_manifest_for_merge = None

        def _queue_filter(q_item: dict) -> bool:
            seg_id = str(q_item.get("line", q_item.get("id", "")))
            if args.only_segment and seg_id != str(args.only_segment):
                return False
            if seg_id in cache_hit_ids:
                return False
            if args.skip_existing:
                out = Path(q_item.get("filename", ""))
                if out.is_file() and out.stat().st_size > 0:
                    return False
            return True

        queue_tts = [item for item in full_queue if _queue_filter(item)]
        skipped_count = len(full_queue) - len(queue_tts)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} segment(s) "
                  f"(cache hit / --skip-existing / --only-segment)")
        if not queue_tts and (existing_manifest_for_merge or cache_hit_ids):
            # Nothing to regenerate; reuse the existing manifest as-is.
            print("All segments already generated; reusing existing manifest.")
            update_job_stage(job_json_path, "tts", "pass")

        # --- Step 5: Run TTS bridge directly ---
        # Dispatch to the correct bridge based on the selected engine.
        # Each bridge accepts the same queue_tts format and produces the
        # same manifest format, so downstream code is engine-agnostic.
        device = "cpu" if args.cpu_only else "auto"
        started = time.time()

        def _run_bridge(engine: str, q_tts: list[dict]) -> subprocess.CompletedProcess:
            """Dispatch to the correct TTS bridge for the given engine.

            Always returns a subprocess.CompletedProcess (never a bare int
            from die()) so callers can safely read .returncode / .stderr.
            Pre-flight failures are encoded as a CompletedProcess with
            returncode=1 and the error message in stderr.
            """
            def _fail(msg: str) -> subprocess.CompletedProcess:
                # Print the same way die() would, but return a
                # CompletedProcess so the caller's .returncode access is
                # safe and the fallback logic can trigger cleanly.
                print(f"run_personal_dub: {msg}", file=sys.stderr)
                return subprocess.CompletedProcess(
                    args=["_run_bridge", engine], returncode=1,
                    stdout="", stderr=msg,
                )

            if engine == "openvoice":
                openvoice_python = OPENVOICE / ".venv" / "bin" / "python"
                if not openvoice_python.is_file():
                    return _fail(f"OpenVoice venv python missing: {openvoice_python}")
                if not OPENVOICE_BRIDGE_SCRIPT.is_file():
                    return _fail(f"OpenVoice bridge script missing: {OPENVOICE_BRIDGE_SCRIPT}")
                checkpoint_dir = OPENVOICE / "checkpoints_v2"
                if not (checkpoint_dir / "converter" / "checkpoint.pth").is_file():
                    return _fail(f"OpenVoice checkpoints missing: {checkpoint_dir}")
                # Normalize language for OpenVoice
                ov_lang = args.target_language.upper()
                if ov_lang.startswith("ZH"):
                    ov_lang = "ZH"
                elif ov_lang.startswith("EN"):
                    ov_lang = "EN"
                return run_openvoice_bridge_direct(
                    queue_tts=q_tts,
                    manifest_path=openvoice_manifest_explicit,
                    queue_path=openvoice_queue,
                    work_dir=openvoice_work,
                    logs_path=openvoice_logs,
                    preserve_dir=generated_audio,
                    openvoice_python=openvoice_python,
                    bridge_script=OPENVOICE_BRIDGE_SCRIPT,
                    openvoice_repo=OPENVOICE,
                    checkpoint_dir=checkpoint_dir,
                    language=ov_lang,
                    device=device,
                    allow_partial=args.allow_partial,
                    base_speaker=args.base_speaker,
                )
            elif engine == "qwen3-local":
                if not QWEN3_BRIDGE_SCRIPT.is_file():
                    return _fail(f"Qwen3 bridge script missing: {QWEN3_BRIDGE_SCRIPT}")
                # Resolve model path: use local dir if it exists, else HF repo ID
                model_path = args.qwen3_model
                if QWEN3_LOCAL_MODEL.is_dir() and model_path == QWEN3_DEFAULT_MODEL:
                    model_path = QWEN3_LOCAL_MODEL.as_posix()
                return run_qwen3_bridge_direct(
                    queue_tts=q_tts,
                    manifest_path=openvoice_manifest_explicit,
                    queue_path=openvoice_queue,
                    work_dir=openvoice_work,
                    logs_path=openvoice_logs,
                    preserve_dir=generated_audio,
                    bridge_script=QWEN3_BRIDGE_SCRIPT,
                    model_path=model_path,
                    language=args.target_language,
                    temperature=args.qwen3_temperature,
                    speed=args.qwen3_speed,
                    allow_partial=args.allow_partial,
                )
            elif engine == "omnivoice":
                if not OMNIVOICE_BRIDGE_SCRIPT.is_file():
                    return _fail(f"OmniVoice bridge script missing: {OMNIVOICE_BRIDGE_SCRIPT}")
                if not args.omnivoice_url:
                    return _fail("--omnivoice-url is required when using --tts-engine omnivoice in split mode")
                return run_omnivoice_bridge_direct(
                    queue_tts=q_tts,
                    manifest_path=openvoice_manifest_explicit,
                    queue_path=openvoice_queue,
                    work_dir=openvoice_work,
                    logs_path=openvoice_logs,
                    preserve_dir=generated_audio,
                    bridge_script=OMNIVOICE_BRIDGE_SCRIPT,
                    api_url=args.omnivoice_url,
                    language=args.target_language,
                    speed=args.tts_speed,
                    allow_partial=args.allow_partial,
                )
            else:
                return _fail(f"Unsupported TTS engine in split mode: {engine}")

        # --- Run the bridge (skip if everything was cached) ---
        if queue_tts:
            bridge_result = _run_bridge(args.tts_engine, queue_tts)
        else:
            # All segments skipped; synthesize a successful no-op result so
            # downstream code treats the existing manifest as the output.
            bridge_result = subprocess.CompletedProcess(
                args=["_cached"], returncode=0, stdout="", stderr="",
            )
        tts_engine_used = args.tts_engine
        fallback_attempted = False

        # Handle fallback engine (only engines with bridges are supported;
        # others are rejected at the entry point)
        if bridge_result.returncode != 0 and args.fallback_tts_engine != "none":
            print(
                f"Primary TTS engine '{args.tts_engine}' failed (exit "
                f"{bridge_result.returncode}). Retrying with fallback "
                f"'{args.fallback_tts_engine}'..."
            )
            fallback_attempted = True
            # Rebuild queue with fallback provider (full queue; fallback
            # should attempt every segment that was originally requested)
            fallback_provider_int = int(TTS_ENGINE_MAP[args.fallback_tts_engine])
            queue_tts = build_queue_tts_with_speakers(
                translated_srt=job_srt,
                speaker_profiles=speaker_profiles_data,
                cache_folder=cache_folder,
                tts_type=fallback_provider_int,
                language=args.target_language,
                character_profiles=cp_for_routing,
            )
            bridge_result = _run_bridge(args.fallback_tts_engine, queue_tts)
            tts_engine_used = args.fallback_tts_engine

        # Wrap into a CompletedProcess-like result for downstream code
        result = bridge_result

        if result.returncode != 0:
            update_job_stage(job_json_path, "tts", "fail")
            return die(f"TTS bridge ({tts_engine_used}) failed: {result.stderr[-1500:]}")
        if not openvoice_manifest_explicit.is_file():
            update_job_stage(job_json_path, "tts", "fail")
            return die(f"TTS manifest not found: {openvoice_manifest_explicit}")

        # --- Merge regenerated results with existing manifest entries ---
        # When --skip-existing / --only-segment filtered the queue, the bridge
        # only produced results for the regenerated subset. Merge them back
        # into the full manifest so downstream audio build sees every segment.
        if existing_manifest_for_merge and skipped_count > 0:
            try:
                new_manifest = read_json(openvoice_manifest_explicit)
                old_by_id = {
                    str(r.get("id", r.get("index", ""))): r
                    for r in existing_manifest_for_merge.get("results", [])
                }
                merged_results = []
                seen_ids: set[str] = set()
                # Preserve order from the full queue; use new result if present,
                # else the old (cached) result.
                new_by_id = {
                    str(r.get("id", r.get("index", ""))): r
                    for r in new_manifest.get("results", [])
                }
                for q_item in full_queue:
                    sid = str(q_item.get("line", q_item.get("id", "")))
                    if sid in new_by_id:
                        merged_results.append(new_by_id[sid])
                        seen_ids.add(sid)
                    elif sid in old_by_id:
                        merged_results.append(old_by_id[sid])
                        seen_ids.add(sid)
                # Append any results not in the queue (defensive)
                for r in new_manifest.get("results", []):
                    rid = str(r.get("id", r.get("index", "")))
                    if rid not in seen_ids:
                        merged_results.append(r)
                        seen_ids.add(rid)
                new_manifest["results"] = merged_results
                new_manifest["segments"] = merged_results
                new_manifest["ok"] = sum(1 for r in merged_results if r.get("status") == "ok")
                new_manifest["error"] = sum(1 for r in merged_results if r.get("status") == "error")
                new_manifest["skipped"] = sum(1 for r in merged_results if r.get("status") == "skipped_empty_text")
                new_manifest["skipped_empty_text"] = new_manifest["skipped"]
                write_json(openvoice_manifest_explicit, new_manifest)
                print(f"Merged manifest: {len(merged_results)} segments "
                      f"({skipped_count} cached + {len(new_by_id)} regenerated)")
            except Exception as exc:
                print(f"WARNING: manifest merge failed: {exc}", file=sys.stderr)
        # Validate the manifest has at least one successful segment. A
        # bridge can exit 0 with an empty manifest or all-failed results;
        # without this check, downstream build_dubbed_audio would produce
        # a silent track.
        try:
            _manifest = read_json(openvoice_manifest_explicit)
        except Exception as exc:
            update_job_stage(job_json_path, "tts", "fail")
            return die(f"failed to read TTS manifest: {exc}")
        _results = _manifest.get("results", _manifest.get("segments", []))
        _ok_count = sum(1 for r in _results if r.get("status") == "ok")
        if _ok_count == 0:
            update_job_stage(job_json_path, "tts", "fail")
            return die(
                f"TTS bridge produced no successful segments "
                f"(manifest has {len(_results)} entries, 0 ok). "
                f"Bridge status: {_manifest.get('status', '?')}"
            )
        if not args.allow_partial and _ok_count < len(_results):
            update_job_stage(job_json_path, "tts", "fail")
            return die(
                f"TTS bridge had partial failures: {_ok_count}/"
                f"{len(_results)} segments ok. Use --allow-partial to "
                f"proceed with silent gaps for failed segments."
            )
        if _ok_count < len(_results):
            print(
                f"WARNING: TTS bridge partial success: "
                f"{_ok_count}/{len(_results)} segments ok "
                f"(--allow-partial accepted)"
            )
        update_job_stage(job_json_path, "tts", "pass")

        # --- Record successful segments in the hash-based cache (v0.12) ---
        # After the bridge succeeds (or all segments were cache hits), record
        # each successful output WAV in the SegmentCache so future re-runs of
        # this job can skip segments whose inputs haven't changed.
        if seg_cache and cache_keys:
            try:
                _cached_count = 0
                for r in _results:
                    if r.get("status") != "ok":
                        continue
                    rid = str(r.get("id", r.get("index", "")))
                    key = cache_keys.get(rid)
                    if not key:
                        continue
                    out_audio = r.get("output_audio") or r.get("preserved_audio")
                    if out_audio and Path(out_audio).is_file():
                        seg_cache.put(key, Path(out_audio))
                        _cached_count += 1
                if _cached_count:
                    seg_cache.save()
                    print(f"Segment cache: recorded {_cached_count} segment(s)")
            except Exception as exc:
                print(f"WARNING: cache write failed: {exc}", file=sys.stderr)

    # ====================================================================
    # ORIGINAL VTV PIPELINE (when --speaker-profiling is off)
    # ====================================================================
    else:
        # --- Configure OpenVoice settings in pyVideoTrans (Blocker 4) ---
        PYVT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        if PYVT_CONFIG.is_file():
            try:
                cfg = read_json(PYVT_CONFIG)
            except Exception:
                cfg = {}
        else:
            cfg = {}
        cfg_original: dict[str, dict] = {}
        cfg_tracked_keys = (
            "openvoice_default_reference",
            PRESERVE_KEY,
            "openvoice_queue_file",
            "openvoice_manifest_file",
            "openvoice_logs_file",
            "openvoice_work_dir",
            "openvoice_allow_partial",
            "line_roles",
            "speaker_type",
            "hf_token",
            "enable_diariz",
        )
        for key in cfg_tracked_keys:
            cfg_original[key] = {
                "existed": key in cfg,
                "value": cfg.get(key),
            }
        cfg["openvoice_default_reference"] = reference.as_posix()
        cfg[PRESERVE_KEY] = generated_audio.as_posix()
        cfg["openvoice_queue_file"] = openvoice_queue.as_posix()
        cfg["openvoice_manifest_file"] = openvoice_manifest_explicit.as_posix()
        cfg["openvoice_logs_file"] = openvoice_logs.as_posix()
        cfg["openvoice_work_dir"] = openvoice_work.as_posix()
        cfg["openvoice_allow_partial"] = bool(args.allow_partial)
        write_json(PYVT_CONFIG, cfg)
        print(f"Active config: reference={reference.as_posix()} "
              f"preserve_dir={generated_audio.as_posix()} "
              f"cfg_path={PYVT_CONFIG.as_posix()}")

        tts_provider = TTS_ENGINE_MAP[args.tts_engine]
        fallback_provider = (
            TTS_ENGINE_MAP[args.fallback_tts_engine]
            if args.fallback_tts_engine != "none"
            else None
        )

        def build_pyvt_cmd(provider: str) -> list[str]:
            return [
                py_python.as_posix(), "cli.py",
                "--task", "vtv",
                "--name", input_video.as_posix(),
                "--recogn_type", recogn_type,
                "--model_name", args.model,
                "--source_language_code", args.source_language,
                "--target_language_code", args.target_language,
                "--tts_type", provider,
                "--voice_role", "default",
                "--subtitle_type", args.subtitle_type,
                "--output-dir", output_dir.as_posix(),
                "--no-clear-cache",
                "--verbose",
            ]

        update_job_stage(job_json_path, "transcription", "running")
        update_job_stage(job_json_path, "translation", "running")
        update_job_stage(job_json_path, "tts", "running")
        started = time.time()
        cmd = build_pyvt_cmd(tts_provider)
        tts_engine_used = args.tts_engine
        fallback_attempted = False
        try:
            result = run_subprocess(cmd, PYVIDEOTRANS, "pyVideoTrans", env=pyvt_env)
            if result.returncode != 0 and fallback_provider and fallback_provider != tts_provider:
                print(
                    f"Primary TTS engine '{args.tts_engine}' failed (exit "
                    f"{result.returncode}). Retrying with fallback "
                    f"'{args.fallback_tts_engine}'..."
                )
                fallback_attempted = True
                fallback_cmd = build_pyvt_cmd(fallback_provider)
                result = run_subprocess(
                    fallback_cmd, PYVIDEOTRANS, "pyVideoTrans-fallback", env=pyvt_env,
                )
                tts_engine_used = args.fallback_tts_engine
        finally:
            if PYVT_CONFIG.is_file():
                try:
                    current = read_json(PYVT_CONFIG)
                except Exception:
                    current = {}
                for key, state in cfg_original.items():
                    if state["existed"]:
                        current[key] = state["value"]
                    else:
                        current.pop(key, None)
                write_json(PYVT_CONFIG, current)

    # --- Locate the OpenVoice manifest (Blocker 5) ---
    # Use the explicit job-scoped manifest path. Fall back to extracting
    # from stdout/stderr only if the explicit path is missing.
    # Global mtime scanning is only used behind --legacy-artifact-scan.
    manifest: Path | None = None
    if openvoice_manifest_explicit.is_file():
        shutil.copy2(openvoice_manifest_explicit, stable_manifest)
        manifest = stable_manifest
    elif args.legacy_artifact_scan:
        legacy_manifest = latest_manifest(started)
        if legacy_manifest and legacy_manifest.is_file():
            print(
                "WARNING: Using legacy global tmp scan for manifest. "
                "This is not deterministic.",
                file=sys.stderr,
            )
            shutil.copy2(legacy_manifest, stable_manifest)
            manifest = stable_manifest
    if not manifest:
        extracted = _extract_manifest_from_output(f"{result.stdout}\n{result.stderr}")
        if extracted:
            write_json(stable_manifest, extracted)
            manifest = stable_manifest

    # --- Locate the SRT (Blocker 7: deterministic paths) ---
    # In the split pipeline, srt_file is already set from the STS step.
    # In the VTV pipeline, we need to find it in the pyVideoTrans cache.
    job_source_srt = subtitles_dir / "source.srt"
    if not srt_file:
        # Compute the pyVideoTrans cache folder uuid (help_ffmpeg.format_video)
        try:
            stat = input_video.stat()
            uuid_input = f"{input_video.as_posix()}-{stat.st_size}-{stat.st_mtime}"
            pyvt_uuid = hashlib.md5(uuid_input.encode()).hexdigest()[:10]
            pyvt_cache = PYVIDEOTRANS / "tmp" / pyvt_uuid
        except Exception:
            pyvt_cache = None

        if pyvt_cache and pyvt_cache.is_dir():
            # Look for source + translated SRTs in the deterministic cache path
            lang_lower = args.target_language.lower().replace("-", "")
            src_lower = args.source_language.lower().replace("-", "")
            cache_srts = list(pyvt_cache.glob("*.srt"))
            if cache_srts:
                lang_matches = [p for p in cache_srts if lang_lower in p.name.lower()]
                if not lang_matches:
                    lang_matches = [p for p in cache_srts if src_lower not in p.name.lower()]
                chosen = (
                    lang_matches[0] if lang_matches
                    else max(cache_srts, key=lambda p: p.stat().st_mtime)
                )
                shutil.copy2(chosen, job_srt)
                srt_file = job_srt
                src_matches = [p for p in cache_srts if src_lower in p.name.lower()]
                if src_matches:
                    shutil.copy2(src_matches[0], job_source_srt)

        if not srt_file and args.legacy_artifact_scan:
            legacy_srt = latest_srt(started)
            if legacy_srt and legacy_srt.is_file():
                shutil.copy2(legacy_srt, job_srt)
                srt_file = job_srt
        elif not srt_file:
            new_srts = [
                p for p in (PYVIDEOTRANS / "tmp").rglob("*.srt")
                if p.stat().st_mtime >= started
            ]
            if new_srts:
                lang_lower = args.target_language.lower().replace("-", "")
                lang_matches = [p for p in new_srts if lang_lower in p.name.lower()]
                chosen = lang_matches[0] if lang_matches else max(new_srts, key=lambda p: p.stat().st_mtime)
                shutil.copy2(chosen, job_srt)
                srt_file = job_srt

    # --- Enrich manifest with speaker_id (Blocker 5) ---
    # If speaker profiling is active, inject speaker_id + reference info into
    # each manifest result so downstream tools (pitch verification, review)
    # know which speaker each segment belongs to.
    if manifest and manifest.is_file() and speaker_profiles_data:
        try:
            manifest_data = read_json(manifest)
            profiles_by_id = {
                p["speaker_id"]: p for p in speaker_profiles_data.get("speakers", [])
            }
            seg_assignments = speaker_profiles_data.get("segment_assignments", [])
            # Build segment_index -> speaker_id map
            seg_to_speaker: dict[int, str] = {}
            for sa in seg_assignments:
                seg_to_speaker[sa.get("segment_index", 0)] = sa.get("speaker_id", "")
            results = manifest_data.get("results", [])
            for index, item in enumerate(results):
                speaker_id = seg_to_speaker.get(index, "")
                if not speaker_id:
                    continue
                profile = profiles_by_id.get(speaker_id, {})
                item["speaker_id"] = speaker_id
                item["reference_audio"] = profile.get("reference_audio", "")
                item["target_gender"] = profile.get("gender", {}).get("label", "")
                item["target_age_band"] = profile.get("age", {}).get("band", "")
                item["target_pitch_median_hz"] = profile.get("pitch", {}).get(
                    "median_f0_hz", 0.0
                )
                item["tts_engine"] = tts_engine_used
                item["fallback_tts_engine"] = args.fallback_tts_engine
            write_json(enriched_manifest_path, manifest_data)
            print(f"Enriched manifest: {enriched_manifest_path}")
        except Exception as exc:
            print(f"WARNING: manifest enrichment failed: {exc}", file=sys.stderr)

    report = {
        "status": "fail",
        "input_video": input_video.as_posix(),
        "output_video": output_video.as_posix(),
        "job_dir": job_dir.as_posix(),
        "manifest": manifest.as_posix() if manifest else "",
        "dubbed_audio": dubbed_audio.as_posix(),
        "segments_total": 0,
        "segments_ok": 0,
        "segments_failed": 0,
        "final_duration_seconds": 0.0,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if manifest and manifest.is_file():
        data = read_json(manifest)
        report.update({
            "segments_total": len(data.get("results", data.get("segments", []))),
            "segments_ok": data.get("ok", 0),
            "segments_failed": data.get("error", 0),
        })

    try:
        if result.returncode != 0:
            update_job_stage(job_json_path, "tts", "fail")
            raise RuntimeError(f"pyVideoTrans exited with code {result.returncode}")
        if not manifest or not manifest.is_file():
            update_job_stage(job_json_path, "tts", "fail")
            raise RuntimeError("OpenVoice manifest was not found")
        if report["segments_ok"] < 1:
            update_job_stage(job_json_path, "tts", "fail")
            raise RuntimeError("OpenVoice manifest has no successful segment")
        update_job_stage(job_json_path, "transcription", "pass")
        update_job_stage(job_json_path, "translation", "pass")
        update_job_stage(job_json_path, "tts", "pass")

        # --- Build dubbed audio from manifest ---
        update_job_stage(job_json_path, "audio_build", "running")
        build_audio_report = job_dir / "build_audio_report.json"
        build_cmd = [
            sys.executable, BUILD_AUDIO_SCRIPT.as_posix(),
            "--manifest", stable_manifest.as_posix(),
            "--input-video", input_video.as_posix(),
            "--output-audio", dubbed_audio.as_posix(),
            "--report-json", build_audio_report.as_posix(),
        ]
        if args.allow_partial:
            build_cmd.append("--allow-partial")
        if args.background_volume > 0:
            build_cmd.extend(["--background-volume", str(args.background_volume)])
        if args.voice_volume != 1.0:
            build_cmd.extend(["--voice-volume", str(args.voice_volume)])
        if args.final_gain != 1.0:
            build_cmd.extend(["--final-gain", str(args.final_gain)])
        if args.no_normalize:
            build_cmd.append("--no-normalize")
        if args.vocal_separation:
            build_cmd.append("--vocal-separation")
            build_cmd.extend(["--vocal-separation-method", args.vocal_separation_method])
            if args.allow_demucs_download:
                build_cmd.append("--allow-demucs-download")
        if args.demucs_timeout != 600:
            build_cmd.extend(["--demucs-timeout", str(args.demucs_timeout)])
        if args.ducking:
            build_cmd.append("--ducking")
        if args.target_lufs is not None:
            build_cmd.extend(["--target-lufs", str(args.target_lufs)])
        if args.background_timeout != 300:
            build_cmd.extend(["--background-timeout", str(args.background_timeout)])
        if args.lufs_timeout != 600:
            build_cmd.extend(["--lufs-timeout", str(args.lufs_timeout)])
        if args.fail_if_background_mix_fails:
            build_cmd.append("--fail-if-background-mix-fails")
        build_result = run_subprocess(build_cmd, ROOT, "build-dubbed-audio")
        if build_result.returncode != 0 or not dubbed_audio.is_file():
            update_job_stage(job_json_path, "audio_build", "fail")
            raise RuntimeError(f"dubbed audio build failed: {build_result.stderr[-1500:]}")
        update_job_stage(job_json_path, "audio_build", "pass")

        # --- Post-generation pitch verification (Blocker 6) ---
        # Compare generated segment F0 to source speaker profiles and flag
        # mismatches. Only runs when speaker profiling is active and
        # --verify-pitch is set.
        pitch_verification: dict | None = None
        if args.verify_pitch and speaker_profiles_data and speaker_profiles_path.is_file():
            update_job_stage(job_json_path, "pitch_verification", "running")
            verify_manifest = (
                enriched_manifest_path if enriched_manifest_path.is_file()
                else stable_manifest
            )
            verify_cmd = [
                py_python.as_posix(),
                VERIFY_PITCH_SCRIPT.as_posix(),
                "--manifest", verify_manifest.as_posix(),
                "--speaker-profiles", speaker_profiles_path.as_posix(),
                "--output", pitch_verification_path.as_posix(),
                "--semitone-threshold", str(args.semitone_threshold),
                "--voiced-ratio-threshold", str(args.voiced_ratio_threshold),
                "--duration-ratio-threshold", str(args.duration_ratio_threshold),
            ]
            verify_result = run_subprocess(verify_cmd, ROOT, "verify-pitch")
            if verify_result.returncode == 0 and pitch_verification_path.is_file():
                pitch_verification = read_json(pitch_verification_path)
                summary = pitch_verification.get("summary", {})
                needs_review = summary.get("needs_review", 0)
                rewrite = summary.get("rewrite_shorter", 0)
                print(
                    f"Pitch verification: ok={summary.get('ok',0)} "
                    f"needs_review={needs_review} rewrite_shorter={rewrite} "
                    f"missing={summary.get('missing',0)}"
                )
                if args.fail_on_pitch_mismatch and (needs_review > 0 or rewrite > 0):
                    update_job_stage(job_json_path, "pitch_verification", "fail")
                    raise RuntimeError(
                        f"pitch verification flagged {needs_review + rewrite} "
                        f"segments (use --fail-on-pitch-mismatch=false to allow)"
                    )
                update_job_stage(job_json_path, "pitch_verification", "pass")
            else:
                print(
                    f"WARNING: pitch verification failed: "
                    f"{verify_result.stderr[-800:]}",
                    file=sys.stderr,
                )
                update_job_stage(job_json_path, "pitch_verification", "skip")

        # --- Remux into the job dir final MP4, then sync to user output ---
        # The canonical final video lives in the job dir so regeneration can
        # rebuild it safely. We copy to the user's requested output path after.
        update_job_stage(job_json_path, "remux", "running")
        remux_cmd = [
            sys.executable, REMUX_SCRIPT.as_posix(),
            "--input-video", input_video.as_posix(),
            "--dubbed-audio", dubbed_audio.as_posix(),
            "--output-video", final_video_job.as_posix(),
        ]
        remux_result = run_subprocess(remux_cmd, ROOT, "remux")
        if remux_result.returncode != 0 or not final_video_job.is_file():
            update_job_stage(job_json_path, "remux", "fail")
            raise RuntimeError(f"remux failed: {remux_result.stderr[-1500:]}")
        update_job_stage(job_json_path, "remux", "pass")

        # --- Verify the final MP4 (Blocker 11) ---
        update_job_stage(job_json_path, "verification", "running")
        source_duration = ffprobe_duration(input_video)
        verification = _verify_final_video(final_video_job, source_duration)
        if not verification["verified"]:
            update_job_stage(job_json_path, "verification", "fail")
            raise RuntimeError(
                f"final video verification failed: {verification}"
            )
        update_job_stage(job_json_path, "verification", "pass")

        # Sync to the user's requested output path
        output_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_video_job, output_video)
        final_duration = ffprobe_duration(output_video)

        # --- Write review_segments.json + remux_command.json ---
        # remux_command points to the job dir final video so regeneration
        # rebuilds that, then syncs to the user output path via report.json.
        # Use explicit job-scoped queue file (Blocker 5). No global mtime scan.
        queue_file = openvoice_queue if openvoice_queue.is_file() else None
        # If we have an enriched manifest, pass it to write_review_file so
        # speaker_id and target fields propagate into review_segments.json.
        review_manifest = (
            enriched_manifest_path if enriched_manifest_path.is_file()
            else stable_manifest
        )
        _write_review_file(
            review_path, queue_file, review_manifest, srt_file, job_dir,
            generated_audio, speaker_profiles=speaker_profiles_data,
        )
        # Preserve audio-quality settings so regeneration uses the same
        # flags as the initial dub.
        audio_options = {
            "background_volume": args.background_volume,
            "voice_volume": args.voice_volume,
            "final_gain": args.final_gain,
            "no_normalize": args.no_normalize,
            "vocal_separation": args.vocal_separation,
            "vocal_separation_method": args.vocal_separation_method,
            "ducking": args.ducking,
            "target_lufs": args.target_lufs,
            "background_timeout": args.background_timeout,
            "lufs_timeout": args.lufs_timeout,
            "demucs_timeout": args.demucs_timeout,
            "fail_if_background_mix_fails": args.fail_if_background_mix_fails,
        }
        _write_remux_command(
            remux_path, input_video, dubbed_audio, final_video_job,
            stable_manifest, audio_options=audio_options,
        )

        report["status"] = "pass"
        report["final_duration_seconds"] = final_duration
        report["review_segments"] = review_path.as_posix()
        report["remux_command"] = remux_path.as_posix()
        report["final_video_job"] = final_video_job.as_posix()
        report["output_video"] = output_video.as_posix()
        report["build_audio_report"] = build_audio_report.as_posix()
        report["detected_gender"] = detected_gender
        report["detected_f0_hz"] = detected_f0
        report["reference_voice"] = reference.as_posix()
        report["asr_model"] = args.model
        report["asr_recogn_type"] = recogn_type
        report["job_id"] = job_id
        report["job_json"] = job_json_path.as_posix()
        report["verification"] = verification
        report["tts_engine"] = tts_engine_used
        report["tts_engine_requested"] = args.tts_engine
        report["fallback_tts_engine"] = args.fallback_tts_engine
        report["fallback_attempted"] = fallback_attempted
        report["speaker_profiling"] = bool(speaker_profiles_data)
        if speaker_profiles_data:
            report["speaker_profiles"] = speaker_profiles_path.as_posix()
            report["speakers_detected"] = len(speaker_profiles_data.get("speakers", []))
            report["enriched_manifest"] = enriched_manifest_path.as_posix()
            # Age model provenance (v0.12)
            am = speaker_profiles_data.get("age_model", {})
            if am:
                report["age_model"] = am
        if character_profiles_path.is_file():
            report["character_profiles"] = character_profiles_path.as_posix()
        if pitch_verification is not None:
            report["pitch_verification"] = pitch_verification_path.as_posix()
            report["pitch_verification_summary"] = pitch_verification.get("summary", {})
        # --- Write failed_segments.json (v0.12 review loop) ---
        failed_segments_path = job_dir / "failed_segments.json"
        try:
            failed = _write_failed_segments(
                review_path, stable_manifest,
                pitch_verification_path if pitch_verification_path.is_file() else None,
                failed_segments_path,
            )
            report["failed_segments"] = failed_segments_path.as_posix()
            report["failed_segment_count"] = failed.get("failed_count", 0)
        except Exception as exc:
            print(f"WARNING: failed_segments.json write failed: {exc}", file=sys.stderr)
        # Merge warnings from the build audio report if present
        if build_audio_report.is_file():
            try:
                build_report = read_json(build_audio_report)
                report["warnings"] = build_report.get("warnings", [])
            except Exception:
                report["warnings"] = []
        else:
            report["warnings"] = []
        write_json(report_path, report)
        print()
        print("run_personal_dub: PASS")
        print(f"Final dubbed video: {output_video}")
        print(f"Job video: {final_video_job}")
        print(f"Duration: {final_duration:.3f}s")
        print(f"Segments: {report['segments_ok']} ok / {report['segments_total']} total")
        print(f"Report: {report_path}")
        print(f"Review: {review_path}")
        print(f"Remux command: {remux_path}")
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        write_json(report_path, report)
        # Mark the job as failed in job.json (Blocker 12)
        state = read_job_state(job_json_path)
        if state:
            state["status"] = "failed"
            state["result"] = "fail"
            state["error"] = str(exc)
            write_job_state(job_json_path, state)
        return die(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
