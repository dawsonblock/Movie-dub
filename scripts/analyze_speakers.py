#!/usr/bin/env python3
"""Speaker profiling: diarization + per-speaker reference extraction + age/gender/pitch.

This script runs pyannote speaker diarization on a video's audio track,
extracts a canonical reference WAV per speaker, and estimates apparent
gender / age band / pitch profile per speaker.

Outputs:
  <output-dir>/speaker_profiles.json
  <output-dir>/SPEAKER_00/reference.wav
  <output-dir>/SPEAKER_00/diarization.rttm
  <output-dir>/SPEAKER_01/reference.wav
  ...

The speaker_profiles.json schema matches the v0.8 speaker-profiling spec:
  {
    "speakers": [
      {
        "speaker_id": "SPEAKER_00",
        "reference_audio": "<output-dir>/SPEAKER_00/reference.wav",
        "total_speech_seconds": 142.3,
        "segment_count": 18,
        "gender": {"label": "male", "confidence": 0.82},
        "age": {"band": "adult", "estimated_years": 38.4, "confidence": 0.64},
        "pitch": {
          "median_f0_hz": 118.2,
          "p10_f0_hz": 91.0,
          "p90_f0_hz": 172.5,
          "voiced_ratio": 0.74
        }
      }
    ],
    "diarization": {
      "backend": "pyannote",
      "model": "pyannote/speaker-diarization-3.1",
      "num_speakers_detected": 3,
      "rttm_file": "<output-dir>/diarization.rttm"
    },
    "segment_assignments": [
      {"segment_index": 0, "speaker_id": "SPEAKER_00", "start": 1.2, "end": 4.8}
    ]
  }

Age estimation is a documented pitch-based heuristic, NOT a trained model.
A real age regressor is flagged as future work.

Run inside the pyVideoTrans venv (has pyannote.audio, torchaudio, librosa):
  pyvideotrans-main/.venv/bin/python scripts/analyze_speakers.py \
    --input video.mp4 --output-dir job/speakers --diarization pyannote \
    --hf-token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SUPPORTED_DIARIZATION = ("pyannote", "built", "none")
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def die(msg: str, code: int = 1) -> int:
    print(f"analyze_speakers: FAIL\nReason: {msg}", file=sys.stderr)
    return code


def local_ffmpeg() -> str:
    """Find ffmpeg: prefer PATH, then Homebrew locations."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if Path(p).is_file():
            return p
    raise RuntimeError("ffmpeg not found")


def extract_full_audio(video_path: Path, sample_rate: int = 16000,
                       out_path: Path | None = None) -> Path:
    """Extract the full audio track as a mono WAV at the given sample rate."""
    if out_path is None:
        out_path = Path(tempfile.mktemp(suffix=".wav"))
    cmd = [
        local_ffmpeg(), "-hide_banner", "-nostdin", "-y",
        "-i", video_path.as_posix(),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        out_path.as_posix(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 or not out_path.is_file():
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return out_path


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()[:16]


def resolve_hf_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def run_pyannote_diarization(
    audio_path: Path,
    hf_token: str,
    num_speakers: int | None,
    device: str = "cpu",
) -> tuple[list[dict], str]:
    """Run pyannote speaker-diarization-3.1.

    Returns (turns, rttm_text) where turns is a list of
    {"speaker_id": "SPEAKER_NN", "start": float, "end": float} sorted by start.
    Speakers are reindexed to SPEAKER_00..SPEAKER_NN-1 in order of first appearance.
    """
    import torch
    import pyannote.audio  # noqa: F401 - required for safe_globals registration
    import torchaudio
    from pyannote.audio import Pipeline

    torch.serialization.add_safe_globals([
        torch.torch_version.TorchVersion,
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Problem,
        pyannote.audio.core.task.Resolution,
    ])

    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=hf_token)
    if device != "cpu":
        pipeline.to(torch.device(device))

    waveform, sample_rate = torchaudio.load(audio_path.as_posix())
    if num_speakers and num_speakers > 0:
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )
    else:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # Reindex speakers by first appearance order
    first_seen: dict[str, int] = {}
    turns: list[dict] = []
    rttm_lines: list[str] = []
    for turn, _, raw_label in diarization.itertracks(yield_label=True):
        if raw_label not in first_seen:
            first_seen[raw_label] = len(first_seen)
        idx = first_seen[raw_label]
        speaker_id = f"SPEAKER_{idx:02d}"
        turns.append({
            "speaker_id": speaker_id,
            "start": float(turn.start),
            "end": float(turn.end),
        })
        rttm_lines.append(
            f"SPEAKER {audio_path.name} 1 {turn.start:.3f} "
            f"{(turn.end - turn.start):.3f} <NA> <NA> {speaker_id} <NA> <NA>"
        )
    turns.sort(key=lambda t: (t["start"], t["speaker_id"]))
    return turns, "\n".join(rttm_lines) + "\n"


def run_built_diarization(
    audio_path: Path, language: str, num_speakers: int | None,
) -> tuple[list[dict], str]:
    """Fallback: use pyVideoTrans's built_speakers (sherpa-onnx seg_model).

    Only supports zh/en. Returns turns in the same format as pyannote.
    This is a thin wrapper that calls the upstream function.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyvideotrans-main"))
    from videotrans.process.prepare_audio import built_speakers  # type: ignore

    # built_speakers expects a subtitles_file (list of [start_ms, end_ms]).
    # We don't have subtitles here, so we feed it the full audio as one segment.
    subtitles = [[0, 99999999]]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(subtitles, f)
        subs_file = Path(f.name)
    speak_file = Path(tempfile.mktemp(suffix=".json"))
    try:
        kw = {
            "input_file": audio_path.as_posix(),
            "subtitles_file": subs_file.as_posix(),
            "speak_file": speak_file.as_posix(),
            "num_speakers": -1 if not num_speakers or num_speakers < 1 else num_speakers,
            "language": language,
        }
        ok, err = built_speakers(**kw)
        if not ok or not speak_file.is_file():
            raise RuntimeError(f"built_speakers failed: {err}")
        # speak_file contains a list of "spkN" labels per subtitle segment.
        # Since we passed one segment, this gives us one speaker. Not useful
        # for real diarization — built_speakers is segment-labeling, not
        # turn-taking. We synthesize one turn for the whole audio.
        labels = json.loads(speak_file.read_text(encoding="utf-8"))
        speaker_label = labels[0] if labels else "spk0"
        idx = int(speaker_label.replace("spk", "")) if speaker_label.startswith("spk") else 0
        turns = [{"speaker_id": f"SPEAKER_{idx:02d}", "start": 0.0, "end": 99999.0}]
        rttm = f"SPEAKER {audio_path.name} 1 0.000 99999.000 <NA> <NA> SPEAKER_{idx:02d} <NA> <NA>\n"
        return turns, rttm
    finally:
        subs_file.unlink(missing_ok=True)
        speak_file.unlink(missing_ok=True)


def select_reference_clip(
    audio_path: Path,
    turns: list[dict],
    speaker_id: str,
    min_s: float = 3.0,
    max_s: float = 8.0,
) -> tuple[float, float] | None:
    """Pick the best reference clip for a speaker.

    Strategy: among this speaker's turns, find the longest contiguous run
    (merging adjacent turns with <0.3s gaps). Clip it to [min_s, max_s].
    Prefer a clip from the middle third of the video for stability.
    Returns (start, end) in seconds, or None if no usable clip.
    """
    spk_turns = sorted(
        [t for t in turns if t["speaker_id"] == speaker_id],
        key=lambda t: t["start"],
    )
    if not spk_turns:
        return None

    # Merge adjacent turns
    merged: list[tuple[float, float]] = []
    for t in spk_turns:
        if merged and t["start"] - merged[-1][1] < 0.3:
            merged[-1] = (merged[-1][0], t["end"])
        else:
            merged.append((t["start"], t["end"]))

    # Score each merged run: prefer longer runs, prefer middle of video
    audio_len = spk_turns[-1]["end"] if spk_turns else 0.0
    # Get actual audio length from the longest turn end
    max_end = max(t["end"] for t in turns) if turns else 0.0
    mid = max_end / 2.0

    best: tuple[float, float, float] | None = None  # (score, start, end)
    for start, end in merged:
        duration = end - start
        if duration < min_s:
            continue
        # Prefer clips near the middle, and longer clips
        clip_mid = (start + end) / 2.0
        distance_from_mid = abs(clip_mid - mid) / max(max_end, 1.0)
        usable = min(duration, max_s)
        score = usable - distance_from_mid * 2.0
        if best is None or score > best[0]:
            best = (score, start, start + min(duration, max_s))

    if best is None:
        # Fall back to the longest single turn even if < min_s
        longest = max(merged, key=lambda r: r[1] - r[0])
        if longest[1] - longest[0] >= 1.0:
            return longest
        return None
    return (best[1], best[2])


def cut_reference_wav(
    audio_path: Path, start: float, end: float, out_path: Path,
    sample_rate: int = 16000,
) -> None:
    """Cut [start, end] from audio_path into out_path as mono WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        local_ffmpeg(), "-hide_banner", "-nostdin", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", audio_path.as_posix(),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "wav",
        out_path.as_posix(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not out_path.is_file():
        raise RuntimeError(
            f"ffmpeg reference cut failed ({start:.2f}-{end:.2f}): {result.stderr[-500:]}"
        )


def profile_pitch(audio_clip_path: Path) -> dict:
    """Compute pitch profile from a WAV clip using librosa.pyin.

    Returns {median_f0_hz, p10_f0_hz, p90_f0_hz, voiced_ratio}.
    """
    import numpy as np
    import librosa

    y, sr = librosa.load(audio_clip_path.as_posix(), sr=16000, mono=True)
    if len(y) < sr * 0.3:
        return {"median_f0_hz": 0.0, "p10_f0_hz": 0.0, "p90_f0_hz": 0.0, "voiced_ratio": 0.0}

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=65, fmax=400, sr=sr, frame_length=2048
    )
    voiced = voiced_flag & ~np.isnan(f0)
    voiced_f0 = f0[voiced]
    total_frames = len(f0)
    voiced_ratio = float(np.sum(voiced)) / float(total_frames) if total_frames else 0.0

    if len(voiced_f0) < 5:
        return {"median_f0_hz": 0.0, "p10_f0_hz": 0.0, "p90_f0_hz": 0.0, "voiced_ratio": voiced_ratio}

    return {
        "median_f0_hz": round(float(np.median(voiced_f0)), 2),
        "p10_f0_hz": round(float(np.percentile(voiced_f0, 10)), 2),
        "p90_f0_hz": round(float(np.percentile(voiced_f0, 90)), 2),
        "voiced_ratio": round(voiced_ratio, 3),
    }


def estimate_gender(median_f0: float, voiced_ratio: float) -> tuple[str, float]:
    """Estimate apparent gender from median F0.

    Threshold ~165 Hz is the standard male/female boundary.
    Confidence scales with distance from the threshold and voiced_ratio.
    Returns (label, confidence) where label is male/female/unknown.
    """
    if voiced_ratio < 0.3 or median_f0 <= 0:
        return ("unknown", 0.0)
    threshold = 165.0
    distance = abs(median_f0 - threshold)
    # Map distance to confidence: 0 Hz away -> 0.5, 40+ Hz away -> ~0.95
    conf = min(0.95, 0.5 + distance / 80.0)
    if median_f0 < threshold:
        return ("male", round(conf, 2))
    return ("female", round(conf, 2))


def estimate_age_band(
    median_f0: float, voiced_ratio: float, p10_f0: float, p90_f0: float,
) -> tuple[str, float, float]:
    """Estimate apparent age band from pitch.

    This is a documented heuristic, NOT a trained model. Returns
    (band, estimated_years, confidence).

    Bands: child / teen / adult / senior.
    """
    if voiced_ratio < 0.3 or median_f0 <= 0:
        return ("unknown", 0.0, 0.0)

    f0_range = p90_f0 - p10_f0
    # Child: very high F0
    if median_f0 > 260:
        return ("child", 8.0, 0.4)
    # Teen: high F0 or high variance in the upper range
    if 220 < median_f0 <= 260 or (180 < median_f0 <= 220 and f0_range > 80):
        return ("teen", 15.0, 0.45)
    # Senior: adult-range F0 but low voiced_ratio (breathiness)
    if 85 <= median_f0 <= 180 and voiced_ratio < 0.5:
        return ("senior", 70.0, 0.4)
    # Adult: the default catch-all
    if 85 <= median_f0 <= 220:
        return ("adult", 35.0, 0.6)
    # Anything else
    return ("adult", 35.0, 0.3)


def assign_segments_to_speakers(
    turns: list[dict], segments: list[dict],
) -> list[dict]:
    """Assign each segment to the speaker with the most time overlap.

    segments: list of {"segment_index": int, "start": float, "end": float}
    turns: list of {"speaker_id": str, "start": float, "end": float}

    Returns the input segments with "speaker_id" added. Segments with no
    overlap get the first speaker as a fallback.
    """
    if not turns:
        return [{**s, "speaker_id": "SPEAKER_00"} for s in segments]

    speaker_ids = sorted({t["speaker_id"] for t in turns})
    fallback = speaker_ids[0]

    out: list[dict] = []
    for seg in segments:
        s_start, s_end = seg["start"], seg["end"]
        s_dur = s_end - s_start
        overlaps: dict[str, float] = {}
        for t in turns:
            overlap = max(0.0, min(s_end, t["end"]) - max(s_start, t["start"]))
            if overlap > 0:
                overlaps[t["speaker_id"]] = overlaps.get(t["speaker_id"], 0.0) + overlap
        if not overlaps:
            out.append({**seg, "speaker_id": fallback})
            continue
        if len(overlaps) == 1:
            # Single speaker overlap: require >20% of segment duration
            best = max(overlaps, key=overlaps.get)
            if overlaps[best] > 0.2 * s_dur:
                out.append({**seg, "speaker_id": best})
            else:
                out.append({**seg, "speaker_id": fallback})
        else:
            best = max(overlaps, key=overlaps.get)
            out.append({**seg, "speaker_id": best})
    return out


def write_rttm(rttm_text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rttm_text, encoding="utf-8")


def build_profiles(
    turns: list[dict],
    audio_path: Path,
    output_dir: Path,
    min_clip_s: float,
    max_clip_s: float,
) -> list[dict]:
    """Build per-speaker profiles: reference WAV + pitch + gender + age."""
    speaker_ids = sorted({t["speaker_id"] for t in turns})
    profiles: list[dict] = []
    for sid in speaker_ids:
        spk_turns = [t for t in turns if t["speaker_id"] == sid]
        total_speech = sum(t["end"] - t["start"] for t in spk_turns)
        spk_dir = output_dir / sid
        spk_dir.mkdir(parents=True, exist_ok=True)
        ref_path = spk_dir / "reference.wav"

        clip = select_reference_clip(audio_path, turns, sid, min_clip_s, max_clip_s)
        if clip is None:
            print(f"  WARNING: no usable reference clip for {sid}, skipping profile")
            continue
        cut_reference_wav(audio_path, clip[0], clip[1], ref_path)

        pitch = profile_pitch(ref_path)
        gender_label, gender_conf = estimate_gender(
            pitch["median_f0_hz"], pitch["voiced_ratio"]
        )
        age_band, age_years, age_conf = estimate_age_band(
            pitch["median_f0_hz"], pitch["voiced_ratio"],
            pitch["p10_f0_hz"], pitch["p90_f0_hz"],
        )

        profiles.append({
            "speaker_id": sid,
            "reference_audio": ref_path.as_posix(),
            "total_speech_seconds": round(total_speech, 2),
            "segment_count": len(spk_turns),
            "gender": {"label": gender_label, "confidence": gender_conf},
            "age": {
                "band": age_band,
                "estimated_years": age_years,
                "confidence": age_conf,
            },
            "pitch": pitch,
        })
        print(
            f"  {sid}: {gender_label} ({gender_conf:.2f}), "
            f"{age_band} (~{age_years:.0f}y), "
            f"F0={pitch['median_f0_hz']:.1f}Hz, "
            f"ref={ref_path.name}"
        )
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Speaker profiling: diarization + reference extraction + age/gender/pitch"
    )
    parser.add_argument("--input", required=True, help="path to video or audio file")
    parser.add_argument("--output-dir", required=True,
                        help="directory to write speaker_profiles.json + per-speaker WAVs")
    parser.add_argument("--diarization", choices=SUPPORTED_DIARIZATION,
                        default="pyannote", help="diarization backend")
    parser.add_argument("--hf-token", default="",
                        help="HuggingFace token (or set HF_TOKEN env)")
    parser.add_argument("--num-speakers", default="auto",
                        help="auto or a positive integer")
    parser.add_argument("--min-clip-seconds", type=float, default=3.0,
                        help="minimum reference clip duration")
    parser.add_argument("--max-clip-seconds", type=float, default=8.0,
                        help="maximum reference clip duration")
    parser.add_argument("--device", default="auto",
                        help="torch device: auto/cpu/mps/cuda:0")
    parser.add_argument("--language", default="en",
                        help="language code for built diarization (zh/en)")
    parser.add_argument("--segments-json", default="",
                        help="optional JSON file with [{segment_index,start,end}] "
                             "to pre-compute segment_assignments")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        return die(f"input not found: {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve num_speakers
    num_speakers: int | None = None
    if args.num_speakers != "auto":
        try:
            num_speakers = int(args.num_speakers)
            if num_speakers < 1:
                return die(f"--num-speakers must be auto or >=1, got {num_speakers}")
        except ValueError:
            return die(f"invalid --num-speakers: {args.num_speakers}")

    # Resolve device
    device = args.device
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda:0"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    print(f"Input:      {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Diarization: {args.diarization}")
    print(f"Device:     {device}")

    # Extract full audio
    print("Extracting full audio track...")
    audio_path = output_dir / "_full_audio.wav"
    try:
        extract_full_audio(input_path, sample_rate=16000, out_path=audio_path)
    except Exception as exc:
        return die(f"audio extraction failed: {exc}")

    audio_sha = file_hash(audio_path)
    print(f"Audio SHA:  {audio_sha}")

    # Run diarization
    turns: list[dict] = []
    rttm_text = ""
    backend = args.diarization
    if backend == "none":
        # Treat the whole audio as one speaker
        turns = [{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 999999.0}]
        rttm_text = f"SPEAKER {input_path.name} 1 0.000 999999.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    elif backend == "pyannote":
        hf_token = resolve_hf_token(args.hf_token or None)
        if not hf_token:
            audio_path.unlink(missing_ok=True)
            return die(
                "pyannote diarization requires a HuggingFace token. "
                "Set HF_TOKEN env or pass --hf-token. "
                "Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1 first."
            )
        print(f"Running pyannote diarization ({PYANNOTE_MODEL})...")
        try:
            turns, rttm_text = run_pyannote_diarization(
                audio_path, hf_token, num_speakers, device=device,
            )
        except Exception as exc:
            audio_path.unlink(missing_ok=True)
            return die(f"pyannote diarization failed: {exc}")
    elif backend == "built":
        print(f"Running built diarization (language={args.language})...")
        try:
            turns, rttm_text = run_built_diarization(
                audio_path, args.language, num_speakers,
            )
        except Exception as exc:
            audio_path.unlink(missing_ok=True)
            return die(f"built diarization failed: {exc}")

    if not turns:
        audio_path.unlink(missing_ok=True)
        return die("diarization produced no turns")

    speaker_ids = sorted({t["speaker_id"] for t in turns})
    print(f"Detected {len(speaker_ids)} speaker(s): {', '.join(speaker_ids)}")

    # Write RTTM
    rttm_path = output_dir / "diarization.rttm"
    write_rttm(rttm_text, rttm_path)

    # Build per-speaker profiles
    print("Building per-speaker profiles...")
    profiles = build_profiles(
        turns, audio_path, output_dir, args.min_clip_seconds, args.max_clip_seconds,
    )

    if not profiles:
        audio_path.unlink(missing_ok=True)
        return die("no speaker profiles could be built (no usable reference clips)")

    # Optional segment assignments
    segment_assignments: list[dict] = []
    if args.segments_json:
        seg_path = Path(args.segments_json).expanduser().resolve()
        if not seg_path.is_file():
            audio_path.unlink(missing_ok=True)
            return die(f"segments-json file not found: {seg_path}")
        try:
            segments = json.loads(seg_path.read_text(encoding="utf-8"))
            if not segments:
                audio_path.unlink(missing_ok=True)
                return die(f"segments-json file is empty: {seg_path}")
            segment_assignments = assign_segments_to_speakers(turns, segments)
        except Exception as exc:
            audio_path.unlink(missing_ok=True)
            return die(f"segment assignment failed: {exc}")
        if not segment_assignments:
            audio_path.unlink(missing_ok=True)
            return die("segment assignment produced no results (no turns?)")

    # Write speaker_profiles.json
    result = {
        "speakers": profiles,
        "diarization": {
            "backend": backend,
            "model": PYANNOTE_MODEL if backend == "pyannote" else backend,
            "num_speakers_detected": len(speaker_ids),
            "rttm_file": rttm_path.as_posix(),
            "audio_sha": audio_sha,
        },
        "segment_assignments": segment_assignments,
    }
    profiles_path = output_dir / "speaker_profiles.json"
    profiles_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Clean up the full audio temp file
    audio_path.unlink(missing_ok=True)

    print()
    print("analyze_speakers: PASS")
    print(f"Profiles: {profiles_path}")
    print(f"Speakers: {len(profiles)}")
    print(f"RTTM:     {rttm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
