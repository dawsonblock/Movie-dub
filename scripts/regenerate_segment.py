#!/usr/bin/env python3
"""Regenerate one OpenVoice segment from a review job directory."""

from __future__ import annotations

import argparse
import json
import re
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
    parser.add_argument("--job", required=True, help="job directory containing review_segments.json")
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--text", default="", help="replacement spoken text; otherwise edited_text/translated_text is used")
    parser.add_argument("--remux", action="store_true", help="run job/remux_command.json command after regeneration")
    args = parser.parse_args()

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
                    manifest["ok"] = len([item for item in results if item.get("status") == "ok"])
                    manifest["error"] = len([item for item in results if item.get("status") == "error"])
                    manifest["last_regenerated_segment"] = args.segment_id
                    write_json(manifest_file, manifest)

        segment["edited_text"] = text
        segment["output_audio"] = output_audio.as_posix()
        segment["status"] = "regenerated"
        write_json(review_file, review)

        if args.remux:
            remux_file = job / "remux_command.json"
            if not remux_file.is_file():
                raise RuntimeError(f"--remux requested but remux command is missing: {remux_file}")
            remux_cmd = read_json(remux_file)
            if not isinstance(remux_cmd, list):
                raise RuntimeError("remux_command.json must contain a command list")
            subprocess.run([str(part) for part in remux_cmd], cwd=job, check=True)

        print("Regenerate segment: PASS")
        print(f"Segment: {args.segment_id}")
        print(f"Audio: {output_audio}")
        print(f"Manifest: {manifest_file}")
        return 0
    except Exception as exc:
        print("Regenerate segment: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
