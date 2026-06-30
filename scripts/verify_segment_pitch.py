#!/usr/bin/env python3
"""Post-generation pitch verification.

After TTS generation, compute F0 for each generated segment WAV and compare
it to the source speaker profile. Flag mismatches:

  - generated median F0 differs by >2 semitones from source -> needs_review
  - generated voiced_ratio < 0.3 -> needs_review
  - generated duration / target duration > 1.35 -> rewrite_shorter

Inputs:
  --manifest        openvoice_manifest.json (or enriched variant)
  --speaker-profiles speaker_profiles.json from analyze_speakers.py
  --output          pitch_verification.json

Output schema:
  {
    "segments": [
      {
        "segment_id": 12,
        "speaker_id": "SPEAKER_01",
        "generated_audio": "/path/to/seg.wav",
        "generated": {"median_f0_hz": 210.0, "voiced_ratio": 0.71, "duration": 2.3},
        "source": {"median_f0_hz": 205.4},
        "pitch_delta_semitones": 0.47,
        "duration_ratio": 1.05,
        "verdict": "ok",
        "flags": []
      }
    ],
    "summary": {"ok": 18, "needs_review": 2, "rewrite_shorter": 1, "missing": 0}
  }

Run inside the pyVideoTrans venv (has librosa + soundfile):
  pyvideotrans-main/.venv/bin/python scripts/verify_segment_pitch.py \
    --manifest job/openvoice_manifest.json \
    --speaker-profiles job/speakers/speaker_profiles.json \
    --output job/pitch_verification.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def die(msg: str, code: int = 1) -> int:
    print(f"verify_segment_pitch: FAIL\nReason: {msg}", file=sys.stderr)
    return code


def load_profiles(path: Path) -> dict[str, dict]:
    """Return {speaker_id: profile_dict} from speaker_profiles.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["speaker_id"]: p for p in data.get("speakers", [])}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_generated_wav(wav_path: Path) -> dict | None:
    """Compute median_f0_hz, voiced_ratio, duration for a generated WAV."""
    try:
        import numpy as np
        import librosa
    except ImportError:
        return None
    try:
        y, sr = librosa.load(wav_path.as_posix(), sr=16000, mono=True)
    except Exception:
        return None
    if len(y) < sr * 0.1:
        return None
    duration = len(y) / float(sr)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=65, fmax=400, sr=sr, frame_length=2048
        )
    except Exception:
        return None
    voiced = voiced_flag & ~np.isnan(f0)
    voiced_f0 = f0[voiced]
    total_frames = len(f0)
    voiced_ratio = float(np.sum(voiced)) / float(total_frames) if total_frames else 0.0
    if len(voiced_f0) < 5:
        return {
            "median_f0_hz": 0.0,
            "voiced_ratio": round(voiced_ratio, 3),
            "duration": round(duration, 3),
        }
    return {
        "median_f0_hz": round(float(np.median(voiced_f0)), 2),
        "voiced_ratio": round(voiced_ratio, 3),
        "duration": round(duration, 3),
    }


def semitone_delta(src_f0: float, gen_f0: float) -> float:
    """Return signed pitch difference in semitones. 0.0 if either is 0."""
    if src_f0 <= 0 or gen_f0 <= 0:
        return 0.0
    return 12.0 * math.log2(gen_f0 / src_f0)


def classify(
    gen: dict | None,
    source_profile: dict | None,
    target_duration: float | None,
    semitone_threshold: float,
    voiced_ratio_threshold: float,
    duration_ratio_threshold: float,
) -> tuple[str, list[str]]:
    """Return (verdict, flags). verdict is ok/needs_review/rewrite_shorter/missing."""
    if gen is None:
        return ("missing", ["generated_audio_unreadable"])
    flags: list[str] = []
    src_f0 = 0.0
    if source_profile:
        src_f0 = source_profile.get("pitch", {}).get("median_f0_hz", 0.0)
    delta = abs(semitone_delta(src_f0, gen["median_f0_hz"]))
    if src_f0 > 0 and gen["median_f0_hz"] > 0 and delta > semitone_threshold:
        flags.append(f"pitch_mismatch_{delta:.1f}_semitones")
    if gen["voiced_ratio"] < voiced_ratio_threshold:
        flags.append(f"low_voiced_ratio_{gen['voiced_ratio']:.2f}")
    duration_ratio = None
    if target_duration and target_duration > 0:
        duration_ratio = gen["duration"] / target_duration
        if duration_ratio > duration_ratio_threshold:
            flags.append(f"duration_too_long_ratio_{duration_ratio:.2f}")
    if any(f.startswith("duration_too_long") for f in flags):
        return ("rewrite_shorter", flags)
    if flags:
        return ("needs_review", flags)
    return ("ok", flags)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-generation pitch verification"
    )
    parser.add_argument("--manifest", required=True, help="openvoice manifest JSON")
    parser.add_argument("--speaker-profiles", required=True,
                        help="speaker_profiles.json from analyze_speakers.py")
    parser.add_argument("--output", required=True, help="output pitch_verification.json")
    parser.add_argument("--semitone-threshold", type=float, default=2.0,
                        help="flag if pitch delta exceeds this many semitones")
    parser.add_argument("--voiced-ratio-threshold", type=float, default=0.3,
                        help="flag if generated voiced_ratio below this")
    parser.add_argument("--duration-ratio-threshold", type=float, default=1.35,
                        help="flag rewrite_shorter if gen/target duration exceeds this")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    profiles_path = Path(args.speaker_profiles).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not manifest_path.is_file():
        return die(f"manifest not found: {manifest_path}")
    if not profiles_path.is_file():
        return die(f"speaker profiles not found: {profiles_path}")

    manifest = load_manifest(manifest_path)
    profiles = load_profiles(profiles_path)
    results = manifest.get("results", [])

    segments_out: list[dict] = []
    summary = {"ok": 0, "needs_review": 0, "rewrite_shorter": 0, "missing": 0}

    for index, item in enumerate(results):
        seg_id = item.get("id", item.get("index", index + 1))
        speaker_id = item.get("speaker_id", "")
        gen_audio = item.get("output_audio", "")
        target_duration = item.get("target_duration") or item.get("duration")
        if gen_audio and Path(gen_audio).is_file():
            gen = profile_generated_wav(Path(gen_audio))
        else:
            gen = None
        source_profile = profiles.get(speaker_id)
        verdict, flags = classify(
            gen, source_profile, target_duration,
            args.semitone_threshold, args.voiced_ratio_threshold,
            args.duration_ratio_threshold,
        )
        summary[verdict] = summary.get(verdict, 0) + 1
        entry = {
            "segment_id": seg_id,
            "speaker_id": speaker_id,
            "generated_audio": gen_audio,
            "generated": gen,
            "source": {
                "median_f0_hz": source_profile.get("pitch", {}).get("median_f0_hz", 0.0)
            } if source_profile else None,
            "pitch_delta_semitones": round(
                semitone_delta(
                    source_profile.get("pitch", {}).get("median_f0_hz", 0.0) if source_profile else 0.0,
                    gen["median_f0_hz"] if gen else 0.0,
                ), 2
            ) if gen and source_profile else None,
            "duration_ratio": round(gen["duration"] / target_duration, 3)
            if gen and target_duration and target_duration > 0 else None,
            "verdict": verdict,
            "flags": flags,
        }
        segments_out.append(entry)
        if verdict != "ok":
            print(f"  seg {seg_id} ({speaker_id}): {verdict} {flags}")

    out = {"segments": segments_out, "summary": summary}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("verify_segment_pitch: PASS")
    print(f"Output: {output_path}")
    print(
        f"Summary: ok={summary['ok']} needs_review={summary['needs_review']} "
        f"rewrite_shorter={summary['rewrite_shorter']} missing={summary['missing']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
