#!/usr/bin/env python3
"""Run a one-segment OpenVoice bridge smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPENVOICE = ROOT / "OpenVoice-main"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one cloned WAV through the OpenVoice bridge")
    parser.add_argument("--text", default="This is a local OpenVoice bridge smoke test.")
    parser.add_argument("--language", default="EN")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    openvoice_python = OPENVOICE / ".venv" / "bin" / "python"
    bridge = ROOT / "bridge" / "openvoice_segment_tts.py"
    checkpoint_dir = OPENVOICE / "checkpoints_v2"
    reference = ROOT / "voices" / "openvoice_default_reference.wav"

    require_file(openvoice_python, "OpenVoice venv python")
    require_file(bridge, "OpenVoice bridge")
    require_file(checkpoint_dir / "converter" / "config.json", "OpenVoice converter config")
    require_file(checkpoint_dir / "converter" / "checkpoint.pth", "OpenVoice converter checkpoint")
    require_file(reference, "default reference WAV")

    speaker_dir = checkpoint_dir / "base_speakers" / "ses"
    if not speaker_dir.is_dir() or not list(speaker_dir.glob("*.pth")):
        raise SystemExit(f"Missing OpenVoice base speaker embeddings: {speaker_dir}")

    work_root = ROOT / "pyvideotrans-main" / "tmp" / "openvoice-smoke"
    stamp = str(int(time.time()))
    work_dir = work_root / stamp
    work_dir.mkdir(parents=True, exist_ok=True)
    output_wav = work_dir / "openvoice-smoke.wav"
    queue_file = work_dir / "queue.json"
    manifest_file = work_dir / "manifest.json"

    queue = [
        {
            "id": "smoke-1",
            "line": 1,
            "text": args.text,
            "filename": output_wav.as_posix(),
            "start_time": 0,
            "end_time": 3000,
            "role": "clone",
            "ref_wav": reference.as_posix(),
        }
    ]
    queue_file.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        openvoice_python.as_posix(),
        bridge.as_posix(),
        "--queue-tts-file",
        queue_file.as_posix(),
        "--manifest-file",
        manifest_file.as_posix(),
        "--work-dir",
        work_dir.as_posix(),
        "--openvoice-repo",
        OPENVOICE.as_posix(),
        "--checkpoint-dir",
        checkpoint_dir.as_posix(),
        "--default-reference",
        reference.as_posix(),
        "--language",
        args.language,
        "--device",
        args.device,
    ]
    result = subprocess.run(cmd, cwd=OPENVOICE, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"OpenVoice bridge smoke failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("ok") != 1 or not output_wav.is_file():
        print(f"OpenVoice bridge smoke did not produce a successful WAV. Manifest: {manifest_file}", file=sys.stderr)
        return 1
    timing = manifest["results"][0].get("timing_status", "unknown")
    print(f"PASS OpenVoice bridge smoke: {output_wav}")
    print(f"Manifest: {manifest_file}")
    print(f"Timing status: {timing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
