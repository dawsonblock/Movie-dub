#!/usr/bin/env python3
"""Segment review/regeneration loop helpers.

This module provides:

  * :func:`write_failed_segments` — extract failed/skipped/needs-review
    segments from a manifest + pitch verification into ``failed_segments.json``.
  * :func:`apply_speaker_change` — re-assign a segment (or all segments of a
    speaker) to a different speaker / reference voice in ``review_segments.json``.
  * :func:`apply_reference_change` — point a segment at a different reference
    WAV without changing the speaker identity.
  * :func:`shorten_text` — truncate a translated line to fit a target duration
    at a rough speech rate, used by ``regenerate_segment.py --shorten``.

These are pure-stdlib helpers so they can be imported from the bare
interpreter. The actual TTS regeneration is done by
``regenerate_segment.py`` which calls into the bridge scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# Rough English speech rate: ~2.7 words/sec. Used only for --shorten
# truncation; not a precise model.
WORDS_PER_SECOND = 2.7


def _read(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def write_failed_segments(
    review_path: Path,
    manifest_path: Path | None,
    pitch_verification_path: Path | None,
    output_path: Path,
) -> dict:
    """Write failed_segments.json listing every segment needing attention.

    A segment is "failed" if any of:
      - manifest status is not "ok" (error / skipped_empty_text / etc.)
      - review status is "needs_review" or any non-ok status
      - pitch verification verdict is "needs_review" / "rewrite_shorter" /
        "missing"

    Returns the written dict.
    """
    failed: list[dict] = []
    seen_ids: set[str] = set()

    # 1. From the manifest
    manifest: dict = {}
    if manifest_path and manifest_path.is_file():
        try:
            manifest = _read(manifest_path)
        except Exception:
            manifest = {}
    for item in manifest.get("results", []):
        status = item.get("status", "ok")
        if status and status != "ok":
            sid = str(item.get("id", item.get("index", "")))
            failed.append({
                "segment_id": sid,
                "source": "manifest",
                "status": status,
                "reason": item.get("error", item.get("reason", "")),
                "output_audio": item.get("output_audio", ""),
            })
            seen_ids.add(sid)

    # 2. From review_segments.json
    if review_path.is_file():
        try:
            review = _read(review_path)
            for item in review:
                status = item.get("status", "ok")
                if status and status not in ("ok", "regenerated"):
                    sid = str(item.get("id", ""))
                    if sid in seen_ids:
                        # Merge reason into existing entry
                        for f in failed:
                            if f["segment_id"] == sid:
                                f.setdefault("review_status", status)
                                f.setdefault("review_reason", item.get("reason", ""))
                                break
                        continue
                    failed.append({
                        "segment_id": sid,
                        "source": "review",
                        "status": status,
                        "reason": item.get("reason", ""),
                        "output_audio": item.get("output_audio", ""),
                    })
                    seen_ids.add(sid)
        except Exception:
            pass

    # 3. From pitch verification
    if pitch_verification_path and pitch_verification_path.is_file():
        try:
            pv = _read(pitch_verification_path)
            for seg in pv.get("segments", []):
                verdict = seg.get("verdict", "ok")
                if verdict in ("needs_review", "rewrite_shorter", "missing"):
                    sid = str(seg.get("segment_id", ""))
                    if sid in seen_ids:
                        for f in failed:
                            if f["segment_id"] == sid:
                                f.setdefault("pitch_verdict", verdict)
                                f.setdefault("pitch_flags", seg.get("flags", []))
                                break
                        continue
                    failed.append({
                        "segment_id": sid,
                        "source": "pitch_verification",
                        "status": verdict,
                        "reason": "; ".join(seg.get("flags", [])),
                        "output_audio": seg.get("generated_audio", ""),
                    })
                    seen_ids.add(sid)
        except Exception:
            pass

    result = {
        "schema_version": "1.0",
        "failed_count": len(failed),
        "segments": failed,
    }
    _write(output_path, result)
    return result


def apply_speaker_change(
    review_path: Path,
    speaker_profiles_path: Path,
    *,
    segment_id: str | None = None,
    from_speaker_id: str | None = None,
    to_speaker_id: str,
) -> int:
    """Re-assign a segment (or all segments of from_speaker_id) to a new speaker.

    Updates each affected review entry's ``speaker_id`` and ``ref_wav`` to the
    new speaker's reference_audio from speaker_profiles.json. Returns the
    number of entries changed.
    """
    if not review_path.is_file():
        print(f"review_segments.json not found: {review_path}", file=sys.stderr)
        return 0
    review = _read(review_path)
    profiles = _read(speaker_profiles_path)
    by_id = {p["speaker_id"]: p for p in profiles.get("speakers", [])}
    new_profile = by_id.get(to_speaker_id)
    if not new_profile:
        print(f"target speaker not found in profiles: {to_speaker_id}",
              file=sys.stderr)
        return 0
    new_ref = new_profile.get("reference_audio", "")
    changed = 0
    for item in review:
        sid = str(item.get("id", ""))
        cur_speaker = item.get("speaker_id", "")
        match = False
        if segment_id is not None and sid == str(segment_id):
            match = True
        elif from_speaker_id is not None and cur_speaker == from_speaker_id:
            match = True
        if match:
            item["speaker_id"] = to_speaker_id
            if new_ref:
                item["ref_wav"] = new_ref
            item["status"] = "pending_regen"
            changed += 1
    _write(review_path, review)
    return changed


def apply_reference_change(
    review_path: Path,
    segment_id: str,
    reference_wav: str,
) -> int:
    """Point a single segment at a different reference WAV. Returns 0/1."""
    if not review_path.is_file():
        return 0
    review = _read(review_path)
    for item in review:
        if str(item.get("id", "")) == str(segment_id):
            item["ref_wav"] = reference_wav
            item["status"] = "pending_regen"
            _write(review_path, review)
            return 1
    return 0


def shorten_text(text: str, target_duration_s: float,
                 words_per_second: float = WORDS_PER_SECOND) -> str:
    """Truncate a line to fit a target duration at a rough speech rate.

    Tries to cut on a sentence/clause boundary first, then on a word
    boundary, preserving trailing punctuation when sensible.
    """
    if target_duration_s <= 0 or not text:
        return text
    max_words = max(1, int(target_duration_s * words_per_second))
    words = text.split()
    if len(words) <= max_words:
        return text
    shortened = " ".join(words[:max_words])
    # Try to end on a sentence boundary if one is nearby (within 2 words)
    for end in (".", "!", "?", ",", ";"):
        idx = shortened.rfind(end)
        if idx > len(shortened) * 0.6:
            return shortened[: idx + 1]
    return shortened + " …"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Review-loop helpers: failed_segments + speaker/ref changes"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fail = sub.add_parser("failed", help="write failed_segments.json")
    p_fail.add_argument("--review", required=True)
    p_fail.add_argument("--manifest", default="")
    p_fail.add_argument("--pitch-verification", default="")
    p_fail.add_argument("--output", required=True)

    p_speaker = sub.add_parser("change-speaker",
                               help="re-assign segment(s) to a new speaker")
    p_speaker.add_argument("--review", required=True)
    p_speaker.add_argument("--speaker-profiles", required=True)
    p_speaker.add_argument("--to-speaker", required=True)
    g = p_speaker.add_mutually_exclusive_group(required=True)
    g.add_argument("--segment-id")
    g.add_argument("--from-speaker")

    p_ref = sub.add_parser("change-reference",
                           help="point a segment at a different reference WAV")
    p_ref.add_argument("--review", required=True)
    p_ref.add_argument("--segment-id", required=True)
    p_ref.add_argument("--reference", required=True)

    args = parser.parse_args()

    if args.cmd == "failed":
        result = write_failed_segments(
            Path(args.review),
            Path(args.manifest) if args.manifest else None,
            Path(args.pitch_verification) if args.pitch_verification else None,
            Path(args.output),
        )
        print(f"failed_segments: {result['failed_count']} segment(s)")
        print(f"Output: {args.output}")
        return 0
    if args.cmd == "change-speaker":
        n = apply_speaker_change(
            Path(args.review), Path(args.speaker_profiles),
            segment_id=args.segment_id, from_speaker_id=args.from_speaker,
            to_speaker_id=args.to_speaker,
        )
        print(f"Changed {n} segment(s) to {args.to_speaker}")
        return 0 if n else 1
    if args.cmd == "change-reference":
        n = apply_reference_change(
            Path(args.review), args.segment_id, args.reference,
        )
        print(f"Updated {n} segment(s)")
        return 0 if n else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
