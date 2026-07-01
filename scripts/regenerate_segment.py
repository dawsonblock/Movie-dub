#!/usr/bin/env python3
"""Regenerate one TTS segment from a review job directory.

Supports the split-pipeline engines: qwen3-local and omnivoice.
"""

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
PYVIDEOTRANS = ROOT / "pyvideotrans-main"

# Bridge scripts per engine. The split pipeline supports qwen3-local
# and omnivoice.
BRIDGE_SCRIPTS = {
    "qwen3-local": ROOT / "bridge" / "qwen3_segment_tts.py",
    "omnivoice": ROOT / "bridge" / "omnivoice_segment_tts.py",
}
BRIDGE_VENVS = {
    "qwen3-local": PYVIDEOTRANS / ".venv" / "bin" / "python",
    "omnivoice": Path(sys.executable),
}
BRIDGE_CWDS = {
    "qwen3-local": ROOT,
    "omnivoice": ROOT,
}


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


def _default_reference() -> Path:
    ref = ROOT / "voices" / "reference.wav"
    if ref.is_file():
        return ref
    male_ref = ROOT / "voices" / "male_reference.wav"
    if male_ref.is_file():
        return male_ref
    return ref


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate one segment from review_segments.json")
    parser.add_argument("--job", help="job directory containing review_segments.json (alias for --job-dir)")
    parser.add_argument("--job-dir", dest="job", help="job directory containing review_segments.json")
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--text", default="", help="replacement spoken text; otherwise edited_text/translated_text is used")
    parser.add_argument("--remux", action="store_true", help="rebuild dubbed audio and remux final video after regeneration")
    # --- Review-loop enhancements (v0.12) ---
    parser.add_argument("--tts-engine", choices=list(BRIDGE_SCRIPTS.keys()),
                        default="qwen3-local",
                        help="TTS bridge to use for regeneration (default: qwen3-local). "
                             "Must match the engine used for the original dub.")
    parser.add_argument("--change-speaker", default="",
                        help="re-assign this segment to a different speaker_id "
                             "(reads the new speaker's reference_audio from "
                             "speakers/speaker_profiles.json)")
    parser.add_argument("--change-reference", default="",
                        help="use this WAV as the clone reference for this segment "
                             "instead of the segment's current ref_wav")
    parser.add_argument("--shorten", action="store_true",
                        help="truncate the translated line to fit the segment "
                             "duration at a rough speech rate before regenerating")
    parser.add_argument("--qwen3-model", default="",
                        help="Qwen3-TTS model path (only used with --tts-engine qwen3-local)")
    parser.add_argument("--omnivoice-url", default="",
                        help="OmniVoice server URL (only used with --tts-engine omnivoice)")
    args = parser.parse_args()

    if not args.job:
        parser.error("one of --job or --job-dir is required")
    job = Path(args.job).expanduser().resolve()
    review_file = job / "review_segments.json"
    manifest_file = job / "tts_manifest.json"
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

        # --- Apply review-loop changes before building the queue ---
        # 1. Change speaker: pull the new speaker's reference_audio from
        #    speaker_profiles.json and update the segment's ref_wav + speaker_id.
        if args.change_speaker:
            profiles_path = job / "speakers" / "speaker_profiles.json"
            if not profiles_path.is_file():
                raise RuntimeError(
                    f"--change-speaker requires speaker_profiles.json: {profiles_path}"
                )
            profiles = read_json(profiles_path)
            by_id = {p["speaker_id"]: p for p in profiles.get("speakers", [])}
            new_spk = by_id.get(args.change_speaker)
            if not new_spk:
                raise RuntimeError(
                    f"speaker '{args.change_speaker}' not found in {profiles_path}. "
                    f"Available: {', '.join(sorted(by_id))}"
                )
            segment["speaker_id"] = args.change_speaker
            if new_spk.get("reference_audio"):
                segment["ref_wav"] = new_spk["reference_audio"]
            print(f"Changed segment {args.segment_id} -> speaker {args.change_speaker}")
        # 2. Change reference: override ref_wav directly.
        if args.change_reference:
            ref_path = Path(args.change_reference).expanduser().resolve()
            if not ref_path.is_file():
                raise RuntimeError(f"--change-reference WAV not found: {ref_path}")
            segment["ref_wav"] = ref_path.as_posix()
            print(f"Changed segment {args.segment_id} reference -> {ref_path}")
        # 3. Shorten: truncate the line to fit the segment duration.
        if args.shorten:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from review_loop import shorten_text  # noqa: E402
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            dur = max(0.0, end - start)
            if dur > 0:
                new_text = shorten_text(text, dur)
                if new_text != text:
                    print(f"Shortened: {len(text)} -> {len(new_text)} chars")
                    text = new_text
        # Persist any speaker/reference/shorten edits back to review_segments.json
        # so the change is durable across re-runs.
        segment["edited_text"] = text
        write_json(review_file, review)

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
            "ref_wav": segment.get("ref_wav") or _default_reference().as_posix(),
        }
        write_json(queue_file, [segment_queue])

        # --- Dispatch to the correct bridge for the chosen engine ---
        engine = args.tts_engine
        bridge_script = BRIDGE_SCRIPTS[engine]
        bridge_python = BRIDGE_VENVS[engine]
        bridge_cwd = BRIDGE_CWDS[engine]
        if not bridge_script.is_file():
            raise RuntimeError(f"bridge script not found for {engine}: {bridge_script}")
        if not bridge_python.is_file():
            raise RuntimeError(f"venv python not found for {engine}: {bridge_python}")

        cmd = [
            bridge_python.as_posix(),
            bridge_script.as_posix(),
            "--queue-tts-file",
            queue_file.as_posix(),
            "--manifest-file",
            regen_manifest_file.as_posix(),
            "--work-dir",
            (job / "regenerate_work").as_posix(),
            "--language",
            segment.get("language", "EN"),
            "--logs-file",
            (job / "regenerate_work" / "bridge.log").as_posix(),
            "--preserve-dir",
            (job / "generated_audio").as_posix(),
        ]
        if engine == "qwen3-local":
            model = args.qwen3_model or ""
            if model:
                cmd.extend(["--model-path", model])
        else:  # omnivoice
            if not args.omnivoice_url:
                raise RuntimeError("--omnivoice-url is required with --tts-engine omnivoice")
            cmd.extend([
                "--api-url", args.omnivoice_url,
                "--default-reference",
                _default_reference().as_posix(),
            ])
        result = subprocess.run(cmd, cwd=bridge_cwd, text=True, capture_output=True)
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
