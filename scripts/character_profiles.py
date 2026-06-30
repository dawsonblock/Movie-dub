#!/usr/bin/env python3
"""Build character_profiles.json from speaker_profiles.json.

A *speaker profile* (from analyze_speakers.py) describes a diarized voice:
speaker_id, reference_audio, pitch, gender, age. A *character profile* is the
user-facing upgrade: it ties a diarized speaker to a named, reviewable
character with a stable identity, a locked voice reference, and the list of
segments that character speaks.

Schema (v1.0):

    {
      "schema_version": "1.0",
      "generated_from": "<path to speaker_profiles.json>",
      "tts_engine": "qwen3-local",
      "characters": [
        {
          "character_id": "CHAR_001",
          "speaker_id": "SPEAKER_02",
          "name": "Unknown Woman 1",
          "gender": "female",
          "age_band": "adult",
          "estimated_years": 34.0,
          "median_f0_hz": 205.0,
          "voice_reference": "job/speakers/SPEAKER_02/reference.wav",
          "voice_locked": false,
          "segments": [4, 9, 12, 18],
          "segment_count": 4,
          "total_speech_seconds": 23.4,
          "tts_engine": "qwen3-local",
          "review_status": "approved"
        }
      ]
    }

The builder is idempotent: re-running on the same speaker_profiles.json with
an existing character_profiles.json preserves any user edits (names,
voice_locked, review_status) by matching on speaker_id. New speakers get
fresh CHAR_NNN ids and default names; removed speakers are dropped.

CLI:
    python scripts/character_profiles.py \
        --speaker-profiles job/speakers/speaker_profiles.json \
        --output job/character_profiles.json \
        --tts-engine qwen3-local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCHEMA_VERSION = "1.0"


def _default_name(gender: str, age_band: str, index: int,
                  existing_names: set[str]) -> str:
    """Generate a unique default character name like 'Unknown Woman 1'."""
    g = gender.lower()
    if g == "male":
        label = "Man"
    elif g == "female":
        label = "Woman"
    else:
        label = "Speaker"
    a = age_band.lower()
    if a == "child":
        prefix = "Child"
    elif a == "teen":
        prefix = "Teen"
    elif a == "senior":
        prefix = "Older"
    else:
        prefix = ""
    base = " ".join(p for p in (prefix, label) if p) or "Speaker"
    # Disambiguate against existing names
    n = 1
    while f"{base} {n}" in existing_names:
        n += 1
    return f"{base} {n}"


def build_character_profiles(
    speaker_profiles: dict,
    *,
    tts_engine: str = "",
    existing: dict | None = None,
) -> dict:
    """Build the character_profiles dict from speaker_profiles.

    ``existing`` is a previously-written character_profiles.json (parsed) used
    to preserve user edits. Edits are matched by speaker_id.
    """
    existing_by_sid: dict[str, dict] = {}
    if existing:
        for ch in existing.get("characters", []):
            sid = ch.get("speaker_id", "")
            if sid:
                existing_by_sid[sid] = ch

    used_names: set[str] = {
        ch.get("name", "") for ch in existing_by_sid.values()
        if ch.get("name")
    }
    used_char_ids: set[str] = {
        ch.get("character_id", "") for ch in existing_by_sid.values()
        if ch.get("character_id")
    }

    speakers = speaker_profiles.get("speakers", [])
    seg_assignments = speaker_profiles.get("segment_assignments", [])

    # Map speaker_id -> list of segment_index
    segments_by_speaker: dict[str, list[int]] = {}
    for sa in seg_assignments:
        sid = sa.get("speaker_id", "")
        idx = sa.get("segment_index")
        if sid and idx is not None:
            segments_by_speaker.setdefault(sid, []).append(idx)
    for sid in segments_by_speaker:
        segments_by_speaker[sid].sort()

    characters: list[dict] = []
    for index, spk in enumerate(speakers):
        sid = spk.get("speaker_id", f"SPEAKER_{index:02d}")
        gender = spk.get("gender", {}).get("label", "unknown")
        age = spk.get("age", {})
        age_band = age.get("band", "unknown")
        pitch_median = spk.get("pitch", {}).get("median_f0_hz", 0.0)
        ref = spk.get("reference_audio", "")
        segs = segments_by_speaker.get(sid, [])
        total_speech = spk.get("total_speech_seconds", 0.0)

        prev = existing_by_sid.get(sid, {})

        # Preserve user-set name; otherwise generate a fresh default.
        name = prev.get("name", "")
        if not name:
            name = _default_name(gender, age_band, index, used_names)
        used_names.add(name)

        # Preserve a stable character_id across re-runs.
        char_id = prev.get("character_id", "")
        if not char_id:
            n = 1
            while f"CHAR_{n:03d}" in used_char_ids:
                n += 1
            char_id = f"CHAR_{n:03d}"
        used_char_ids.add(char_id)

        # Preserve user locks; never auto-lock.
        voice_locked = bool(prev.get("voice_locked", False))
        review_status = prev.get("review_status", "pending")
        # Preserve a user-overridden voice reference if locked; otherwise
        # follow the speaker profile's reference_audio.
        if voice_locked and prev.get("voice_reference"):
            voice_ref = prev["voice_reference"]
        else:
            voice_ref = ref

        # Preserve a per-character tts_engine override; default to the
        # pipeline-wide setting.
        char_tts = prev.get("tts_engine", "") or tts_engine

        characters.append({
            "character_id": char_id,
            "speaker_id": sid,
            "name": name,
            "gender": gender,
            "age_band": age_band,
            "estimated_years": age.get("estimated_years", 0.0),
            "median_f0_hz": pitch_median,
            "voice_reference": voice_ref,
            "voice_locked": voice_locked,
            "segments": segs,
            "segment_count": len(segs),
            "total_speech_seconds": total_speech,
            "tts_engine": char_tts,
            "review_status": review_status,
        })

    # Sort by first segment for stable, readable output.
    def _sort_key(ch: dict) -> tuple:
        segs = ch.get("segments", [])
        return (min(segs) if segs else 999999, ch.get("character_id", ""))
    characters.sort(key=_sort_key)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_from": "",
        "tts_engine": tts_engine,
        "characters": characters,
    }


def write_character_profiles(path: Path, data: dict,
                             source: str = "") -> None:
    """Write character_profiles.json, recording the source path."""
    if source:
        data["generated_from"] = source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def load_character_profiles(path: Path) -> dict | None:
    """Read an existing character_profiles.json, or None if missing/invalid."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def rename_character(path: Path, character_id: str, new_name: str) -> int:
    """Rename a character in-place in character_profiles.json. Returns 0/1."""
    data = load_character_profiles(path)
    if data is None:
        print(f"character_profiles.json not found: {path}", file=sys.stderr)
        return 1
    changed = False
    for ch in data.get("characters", []):
        if ch.get("character_id") == character_id:
            ch["name"] = new_name
            changed = True
            break
    if not changed:
        print(f"character_id not found: {character_id}", file=sys.stderr)
        return 1
    write_character_profiles(path, data)
    print(f"Renamed {character_id} -> {new_name}")
    return 0


def lock_voice(path: Path, character_id: str, reference_wav: str,
               locked: bool = True) -> int:
    """Lock/unlock a character's voice reference in-place. Returns 0/1."""
    data = load_character_profiles(path)
    if data is None:
        print(f"character_profiles.json not found: {path}", file=sys.stderr)
        return 1
    changed = False
    for ch in data.get("characters", []):
        if ch.get("character_id") == character_id:
            if reference_wav:
                ch["voice_reference"] = reference_wav
            ch["voice_locked"] = locked
            changed = True
            break
    if not changed:
        print(f"character_id not found: {character_id}", file=sys.stderr)
        return 1
    write_character_profiles(path, data)
    state = "locked" if locked else "unlocked"
    print(f"{state} voice for {character_id}")
    return 0


def set_review_status(path: Path, character_id: str, status: str) -> int:
    """Set review_status (approved/rejected/pending) for a character."""
    data = load_character_profiles(path)
    if data is None:
        print(f"character_profiles.json not found: {path}", file=sys.stderr)
        return 1
    changed = False
    for ch in data.get("characters", []):
        if ch.get("character_id") == character_id:
            ch["review_status"] = status
            changed = True
            break
    if not changed:
        print(f"character_id not found: {character_id}", file=sys.stderr)
        return 1
    write_character_profiles(path, data)
    print(f"Set {character_id} review_status={status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build character_profiles.json from speaker_profiles.json"
    )
    parser.add_argument("--speaker-profiles", required=True,
                        help="speaker_profiles.json from analyze_speakers.py")
    parser.add_argument("--output", required=True,
                        help="output character_profiles.json path")
    parser.add_argument("--tts-engine", default="",
                        help="TTS engine to record per character (default: '')")
    parser.add_argument("--rename", nargs=2, metavar=("CHAR_ID", "NAME"),
                        help="rename a character in the output file")
    parser.add_argument("--lock-voice", nargs=2, metavar=("CHAR_ID", "WAV"),
                        help="lock a character's voice reference to WAV")
    parser.add_argument("--unlock-voice", metavar="CHAR_ID",
                        help="unlock a character's voice reference")
    parser.add_argument("--set-review", nargs=2, metavar=("CHAR_ID", "STATUS"),
                        help="set review_status (approved/rejected/pending)")
    args = parser.parse_args()

    out_path = Path(args.output).expanduser().resolve()
    sp_path = Path(args.speaker_profiles).expanduser().resolve()

    # Edit operations on an existing file
    if args.rename:
        return rename_character(out_path, args.rename[0], args.rename[1])
    if args.lock_voice:
        return lock_voice(out_path, args.lock_voice[0], args.lock_voice[1], True)
    if args.unlock_voice:
        return lock_voice(out_path, args.unlock_voice, "", False)
    if args.set_review:
        return set_review_status(out_path, args.set_review[0], args.set_review[1])

    # Build operation
    if not sp_path.is_file():
        print(f"speaker_profiles.json not found: {sp_path}", file=sys.stderr)
        return 1
    speaker_profiles = json.loads(sp_path.read_text(encoding="utf-8"))
    existing = load_character_profiles(out_path)
    data = build_character_profiles(
        speaker_profiles, tts_engine=args.tts_engine, existing=existing,
    )
    write_character_profiles(out_path, data, source=sp_path.as_posix())

    print("character_profiles: PASS")
    print(f"Output: {out_path}")
    print(f"Characters: {len(data['characters'])}")
    for ch in data["characters"]:
        lock = " [locked]" if ch["voice_locked"] else ""
        print(
            f"  {ch['character_id']} ({ch['speaker_id']}): {ch['name']} "
            f"- {ch['gender']}/{ch['age_band']}, "
            f"{ch['segment_count']} segments{lock}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
