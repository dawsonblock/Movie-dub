#!/usr/bin/env python3
"""Regenerate one OpenVoice segment from a review job directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPENVOICE = ROOT / "OpenVoice-main"


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_segment(review: list[dict], segment_id: str) -> dict:
    for item in review:
        if str(item.get("id", item.get("segment_id", ""))) == segment_id:
            return item
    raise RuntimeError(f"segment id not found in review file: {segment_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate one OpenVoice segment from review_segments.json")
    parser.add_argument("--job", help="job directory containing review_segments.json (alias for --job-dir)")
    parser.add_argument("--job-dir", dest="job", help="job directory containing review_segments.json")
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--text", default="", help="replacement spoken text; otherwise edited_text/translated_text is used")
    parser.add_argument("--remux", action="store_true", help="rebuild dubbed audio and remux final video after regeneration")
    args = parser.parse_args()

    if not args.job:
        parser.error("one of --job or --job-dir is required")
    job = Path(args.job).expanduser().resolve()
    review_file = job / "review_segments.json"
    manifest_file = job / "openvoice_manifest.json"
    if not review_file.is_file():
        print(f"Missing review file: {review_file}", file=sys.stderr)
        return 1

    try:
        review = read_json(review_file)
        if not isinstance(review, list):
            raise RuntimeError("review_segments.json must contain a JSON list")
        segment = find_segment(review, str(args.segment_id))
        text = args.text or segment.get("edited_text") or segment.get("translated_text") or segment.get("text")
        if not text:
            raise RuntimeError("no text available for regeneration")

        generated_dir = job / "generated_audio"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.segment_id)).strip("_") or "segment"
        output_audio = Path(segment.get("output_audio") or generated_dir / f"{safe_id}.wav")
        if not output_audio.is_absolute():
            output_audio = job / output_audio

        # Track the old segment path for proof (Blocker 10)
        old_segment_path = output_audio if output_audio.is_file() else None

        stamp = int(time.time())
        queue_file = job / f"regenerate-{args.segment_id}-{stamp}.json"
        regen_manifest_file = job / f"regenerate-{args.segment_id}-{stamp}-manifest.json"
        segment_queue = {
            "id": segment.get("id", args.segment_id),
            "line": segment.get("id", args.segment_id),
            "text": text,
            "filename": output_audio.as_posix(),
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "role": segment.get("role", "clone"),
            "ref_wav": segment.get("ref_wav") or (ROOT / "voices" / "openvoice_default_reference.wav").as_posix(),
        }
        write_json(queue_file, [segment_queue])

        cmd = [
            (OPENVOICE / ".venv" / "bin" / "python").as_posix(),
            (ROOT / "bridge" / "openvoice_segment_tts.py").as_posix(),
            "--queue-tts-file",
            queue_file.as_posix(),
            "--manifest-file",
            regen_manifest_file.as_posix(),
            "--work-dir",
            (job / "regenerate_work").as_posix(),
            "--openvoice-repo",
            OPENVOICE.as_posix(),
            "--checkpoint-dir",
            (OPENVOICE / "checkpoints_v2").as_posix(),
            "--default-reference",
            (ROOT / "voices" / "openvoice_default_reference.wav").as_posix(),
            "--language",
            segment.get("language", "EN"),
            "--device",
            segment.get("device", "auto"),
            "--preserve-dir",
            (job / "generated_audio").as_posix(),
        ]
        result = subprocess.run(cmd, cwd=OPENVOICE, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"bridge exited with code {result.returncode}: {result.stderr[-1200:]}")

        if manifest_file.is_file() and regen_manifest_file.is_file():
            manifest = read_json(manifest_file)
            regen_manifest = read_json(regen_manifest_file)
            if isinstance(manifest, dict) and isinstance(regen_manifest, dict):
                new_results = regen_manifest.get("results", [])
                if new_results:
                    replacement = new_results[0]
                    replaced = False
                    results = manifest.setdefault("results", [])
                    for index, existing in enumerate(results):
                        if str(existing.get("id", existing.get("index", ""))) == str(args.segment_id):
                            results[index] = replacement
                            replaced = True
                            break
                    if not replaced:
                        results.append(replacement)
                    # Keep segments mirror in sync with results.
                    manifest["segments"] = list(results)
                    manifest["ok"] = len([item for item in results if item.get("status") == "ok"])
                    manifest["error"] = len([item for item in results if item.get("status") == "error"])
                    manifest["skipped"] = len([item for item in results if item.get("status") == "skipped_empty_text"])
                    manifest["skipped_empty_text"] = manifest["skipped"]
                    manifest["last_regenerated_segment"] = args.segment_id
                    write_json(manifest_file, manifest)
                    preserved = replacement.get("preserved_audio", "")
                    if preserved:
                        output_audio = Path(preserved)

        segment["edited_text"] = text
        segment["output_audio"] = output_audio.as_posix()
        segment["status"] = "regenerated"
        write_json(review_file, review)

        # Track old segment hash for proof (Blocker 10)
        old_hash = ""
        if old_segment_path and Path(old_segment_path).is_file():
            old_hash = hashlib.sha256(
                Path(old_segment_path).read_bytes()
            ).hexdigest()[:16]

        audio_rebuilt = False
        video_remuxed = False
        output_synced = False
        synced_output = ""

        if args.remux:
            remux_file = job / "remux_command.json"
            if not remux_file.is_file():
                raise RuntimeError(f"--remux requested but remux command is missing: {remux_file}")
            remux_cmd = read_json(remux_file)
            # Support both the new dict format and the legacy flat-list format.
            if isinstance(remux_cmd, dict):
                build_cmd = remux_cmd.get("build_audio_command")
                mux_cmd = remux_cmd.get("command")
                if build_cmd:
                    subprocess.run([str(part) for part in build_cmd], cwd=ROOT, check=True)
                    audio_rebuilt = True
                if mux_cmd:
                    subprocess.run([str(part) for part in mux_cmd], cwd=ROOT, check=True)
                    video_remuxed = True
            elif isinstance(remux_cmd, list):
                subprocess.run([str(part) for part in remux_cmd], cwd=job, check=True)
                audio_rebuilt = True
                video_remuxed = True
            else:
                raise RuntimeError("remux_command.json must contain a command list or dict")

            # Sync the rebuilt job video to the user's requested output path
            # so the user sees the updated final MP4 without hunting in the job dir.
            report_file = job / "report.json"
            if report_file.is_file():
                report_data = read_json(report_file)
                user_output = report_data.get("output_video", "")
                job_video = report_data.get("final_video_job", str(job / "final_dubbed.mp4"))
                if user_output and Path(job_video).is_file():
                    shutil.copy2(job_video, user_output)
                    synced_output = user_output
                    output_synced = True

        # Compute new segment hash for proof (Blocker 10)
        new_hash = ""
        if output_audio.is_file():
            new_hash = hashlib.sha256(output_audio.read_bytes()).hexdigest()[:16]

        # Write regeneration proof report (Blocker 10)
        regen_report = {
            "regenerated_segment_id": args.segment_id,
            "old_segment_path": str(old_segment_path) if old_segment_path else "",
            "new_segment_path": output_audio.as_posix(),
            "old_segment_hash": old_hash,
            "new_segment_hash": new_hash,
            "hash_changed": old_hash != new_hash if old_hash else True,
            "manifest_updated": manifest_file.is_file(),
            "audio_rebuilt": audio_rebuilt,
            "video_remuxed": video_remuxed,
            "output_synced": output_synced,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        regen_report_path = job / "regeneration_report.json"
        write_json(regen_report_path, regen_report)

        print("Regenerate segment: PASS")
        print(f"Segment: {args.segment_id}")
        print(f"Audio: {output_audio}")
        print(f"Manifest: {manifest_file}")
        if args.remux:
            print(f"Updated audio: {job / 'dubbed_audio.wav'}")
            print(f"Updated video: {job / 'final_dubbed.mp4'}")
            if synced_output:
                print(f"Synced output: {synced_output}")
        print(f"Report: {regen_report_path}")
        return 0
    except Exception as exc:
        print("Regenerate segment: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
