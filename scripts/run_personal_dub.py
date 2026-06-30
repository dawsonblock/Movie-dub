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
VERIFY_PITCH_SCRIPT = ROOT / "scripts" / "verify_segment_pitch.py"
PYVT_CONFIG = PYVIDEOTRANS / "videotrans" / "cfg.json"
PRESERVE_KEY = "openvoice_preserve_dir"
HUGGINGFACE_ASR_PROVIDER = "4"  # HuggingFace ASR for whisper-large-v3-turbo

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
    args = parser.parse_args()

    input_video = Path(args.input).expanduser().resolve()
    output_video = Path(args.output).expanduser().resolve()
    py_python = PYVIDEOTRANS / ".venv" / "bin" / "python"

    # --- Validate inputs ---
    if not input_video.is_file():
        return die(f"input video missing: {input_video}")
    if not py_python.is_file():
        return die(f"pyVideoTrans venv python missing: {py_python}")
    if not (OPENVOICE / ".venv" / "bin" / "python").is_file():
        return die(f"OpenVoice venv python missing: {OPENVOICE / '.venv' / 'bin' / 'python'}")
    if not (OPENVOICE / "checkpoints_v2" / "converter" / "checkpoint.pth").is_file():
        return die("OpenVoice checkpoints missing. Run: make download-openvoice")
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

    # --- Speaker profiling (v0.8: Blocker 1, 2, 3) ---
    # If --speaker-profiling is on, run analyze_speakers.py to produce
    # per-speaker reference WAVs + speaker_profiles.json. If a profile is
    # already provided via --speaker-profile-json, copy it in.
    speaker_profiles_data: dict | None = None
    if args.speaker_profiling or args.speaker_profile_json:
        speakers_dir.mkdir(parents=True, exist_ok=True)
        if args.speaker_profile_json:
            src_profile = Path(args.speaker_profile_json).expanduser().resolve()
            if not src_profile.is_file():
                return die(f"speaker profile JSON not found: {src_profile}")
            shutil.copy2(src_profile, speaker_profiles_path)
            print(f"Speaker profiles (reused): {speaker_profiles_path}")
        else:
            if not ANALYZE_SPEAKERS_SCRIPT.is_file():
                return die(f"analyze_speakers.py missing: {ANALYZE_SPEAKERS_SCRIPT}")
            print("Running speaker profiling...")
            analyze_cmd = [
                py_python.as_posix(),
                ANALYZE_SPEAKERS_SCRIPT.as_posix(),
                "--input", input_video.as_posix(),
                "--output-dir", speakers_dir.as_posix(),
                "--diarization", args.speaker_diarization,
                "--num-speakers", args.num_speakers,
                "--min-clip-seconds", str(args.min_clip_seconds),
                "--max-clip-seconds", str(args.max_clip_seconds),
            ]
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
        try:
            speaker_profiles_data = read_json(speaker_profiles_path)
        except Exception as exc:
            return die(f"failed to read speaker_profiles.json: {exc}")
        speakers = speaker_profiles_data.get("speakers", [])
        if not speakers:
            return die("speaker profiling produced no speakers")
        print(f"  {len(speakers)} speaker(s) profiled")
        for spk in speakers:
            g = spk.get("gender", {})
            a = spk.get("age", {})
            p = spk.get("pitch", {})
            print(
                f"  {spk['speaker_id']}: {g.get('label','?')} "
                f"({g.get('confidence',0):.2f}), {a.get('band','?')}, "
                f"F0={p.get('median_f0_hz',0):.1f}Hz"
            )

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

    # --- Configure OpenVoice settings in pyVideoTrans (Blocker 4) ---
    # Set the reference voice + preserve dir so segment WAVs survive for rebuild.
    # Track the original state of every key we touch so we can restore exactly.
    # If cfg.json does not exist, create it. Never restore missing keys as
    # empty strings — remove them instead.
    PYVT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    if PYVT_CONFIG.is_file():
        try:
            cfg = read_json(PYVT_CONFIG)
        except Exception:
            cfg = {}
    else:
        cfg = {}
    # Track original key state: {key: {"existed": bool, "value": any}}
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
    # When speaker profiling is active, we pre-computed speaker assignments.
    # We do NOT want pyVideoTrans to re-diarize; we inject line_roles so each
    # segment uses the "clone" role and gets routed to the right reference.
    if speaker_profiles_data:
        # line_roles maps segment line number (1-based string) -> role name.
        # We use "clone" for all segments so pyVideoTrans cuts/uses per-segment
        # ref_wav. The per-speaker reference routing is handled by pre-placing
        # clone-{i}.wav files in the cache folder (see below).
        segment_assignments = speaker_profiles_data.get("segment_assignments", [])
        line_roles: dict[str, str] = {}
        for sa in segment_assignments:
            seg_idx = sa.get("segment_index", 0)
            line_roles[str(seg_idx + 1)] = "clone"
        if line_roles:
            cfg["line_roles"] = line_roles
        # Disable pyVideoTrans's own diarization since we pre-computed
        cfg["enable_diariz"] = False
    write_json(PYVT_CONFIG, cfg)
    # Log the active config for debugging
    print(f"Active config: reference={reference.as_posix()} "
          f"preserve_dir={generated_audio.as_posix()} "
          f"cfg_path={PYVT_CONFIG.as_posix()}")

    # --- Determine ASR provider type ---
    # HuggingFace ASR (recogn_type=4) for transformer-based models like
    # whisper-large-v3-turbo; faster-whisper (recogn_type=0) for tiny/base/etc.
    if args.recogn_type:
        recogn_type = args.recogn_type
    elif "/" in args.model or args.model.startswith("whisper-large"):
        # HuggingFace model path (e.g. openai/whisper-large-v3-turbo)
        recogn_type = HUGGINGFACE_ASR_PROVIDER
    else:
        recogn_type = "0"  # faster-whisper

    # --- Run pyVideoTrans VTV with selected TTS engine ---
    # Map --tts-engine to the pyVideoTrans provider index. Falls back to the
    # configured fallback engine if the primary fails (Blocker 4).
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

    # Build environment for subprocess
    pyvt_env = dict(os.environ)
    if args.cpu_only:
        # Force CPU by disabling MPS
        pyvt_env["PYTORCH_MPS_DISABLE"] = "1"
        pyvt_env["CUDA_VISIBLE_DEVICES"] = ""
    update_job_stage(job_json_path, "transcription", "running")
    update_job_stage(job_json_path, "translation", "running")
    update_job_stage(job_json_path, "tts", "running")
    started = time.time()
    cmd = build_pyvt_cmd(tts_provider)
    tts_engine_used = args.tts_engine
    fallback_attempted = False
    try:
        result = run_subprocess(cmd, PYVIDEOTRANS, "pyVideoTrans", env=pyvt_env)
        # Fallback engine retry (Blocker 4): if the primary engine fails and a
        # fallback is configured, re-run with the fallback provider. We only
        # retry on hard failures (nonzero exit), not on partial success.
        if result.returncode != 0 and fallback_provider and fallback_provider != tts_provider:
            print(
                f"Primary TTS engine '{args.tts_engine}' failed (exit "
                f"{result.returncode}). Retrying with fallback "
                f"'{args.fallback_tts_engine}'..."
            )
            fallback_attempted = True
            # Re-write cfg with the fallback engine's default reference if needed.
            # The OpenVoice-specific keys only matter for provider 34; for other
            # engines they are ignored by pyVideoTrans.
            fallback_cmd = build_pyvt_cmd(fallback_provider)
            result = run_subprocess(
                fallback_cmd, PYVIDEOTRANS, "pyVideoTrans-fallback", env=pyvt_env,
            )
            tts_engine_used = args.fallback_tts_engine
    finally:
        # Restore the config so we don't leak preserve_dir into other runs.
        # Restore exactly: if a key existed before, put its original value back;
        # if it did not exist, remove it. Never restore missing keys as empty
        # strings (Blocker 4).
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
    # pyVideoTrans writes SRTs into its cache_folder (tmp/<uuid>/). The uuid is
    # deterministic (md5 of name+size+mtime), so we can predict the cache path.
    # We try the deterministic cache path first, then fall back to scanning
    # newly-created SRTs, then the legacy global scan.
    srt_file: Path | None = None
    job_source_srt = subtitles_dir / "source.srt"
    # Compute the pyVideoTrans cache folder uuid (help_ffmpeg.format_video)
    try:
        import hashlib as _hashlib
        stat = input_video.stat()
        uuid_input = f"{input_video.as_posix()}-{stat.st_size}-{stat.st_mtime}"
        pyvt_uuid = _hashlib.md5(uuid_input.encode()).hexdigest()[:10]
        pyvt_cache = PYVIDEOTRANS / "tmp" / pyvt_uuid
    except Exception:
        pyvt_cache = None

    if pyvt_cache and pyvt_cache.is_dir():
        # Look for source + translated SRTs in the deterministic cache path
        lang_lower = args.target_language.lower().replace("-", "")
        src_lower = args.source_language.lower().replace("-", "")
        # pyVideoTrans names SRTs like: faster-YYYYMMDD-HH_MM_SS.srt or
        # <lang>-dubbing.srt. We take the most recently modified matching file.
        cache_srts = list(pyvt_cache.glob("*.srt"))
        if cache_srts:
            # Try to find a translated/target-language SRT
            lang_matches = [p for p in cache_srts if lang_lower in p.name.lower()]
            if not lang_matches:
                # Fall back to any SRT that's not obviously the source
                lang_matches = [p for p in cache_srts if src_lower not in p.name.lower()]
            chosen = (
                lang_matches[0] if lang_matches
                else max(cache_srts, key=lambda p: p.stat().st_mtime)
            )
            shutil.copy2(chosen, job_srt)
            srt_file = job_srt
            # Also copy a source SRT if present
            src_matches = [p for p in cache_srts if src_lower in p.name.lower()]
            if src_matches:
                shutil.copy2(src_matches[0], job_source_srt)

    if not srt_file and args.legacy_artifact_scan:
        legacy_srt = latest_srt(started)
        if legacy_srt and legacy_srt.is_file():
            shutil.copy2(legacy_srt, job_srt)
            srt_file = job_srt
    elif not srt_file:
        # Last-resort fallback: scan newly created SRTs (less deterministic)
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
        if pitch_verification is not None:
            report["pitch_verification"] = pitch_verification_path.as_posix()
            report["pitch_verification_summary"] = pitch_verification.get("summary", {})
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
