#!/usr/bin/env python3
"""Smoke test: generate one real TTS segment with Qwen3-TTS (MLX).

This proves the Qwen3-local engine end-to-end on this Mac:
  - mlx-audio loads the model
  - voice cloning from a reference WAV works
  - a real 24 kHz WAV is written
  - the WAV is non-empty and has a sane duration

It does NOT touch pyVideoTrans, diarization, or the full dub pipeline.
It only proves the TTS bridge can produce audio.

Run with the wrapper python (the one that runs run_personal_dub.py):
  python3 scripts/smoke_qwen3.py
  python3 scripts/smoke_qwen3.py --model-path models/Qwen3-TTS-12Hz-0.6B-Base-bf16

Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "bridge" / "qwen3_segment_tts.py"
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_MODEL_REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
DEFAULT_REFERENCE = ROOT / "voices" / "reference.wav"
SAMPLE_TEXT = "I told you not to open that door."


def _resolve_model(arg: str) -> str:
    if arg and arg != DEFAULT_MODEL_REPO:
        return arg
    if DEFAULT_MODEL_DIR.is_dir():
        return DEFAULT_MODEL_DIR.as_posix()
    return DEFAULT_MODEL_REPO


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-TTS one-segment smoke test")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_REPO,
                        help="local model dir or HF repo id")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE.as_posix(),
                        help="reference WAV for voice cloning")
    parser.add_argument("--text", default=SAMPLE_TEXT,
                        help="text to synthesize")
    parser.add_argument("--language", default="en")
    parser.add_argument("--work-dir", default="",
                        help="output dir (default: a temp dir)")
    args = parser.parse_args()

    # Pre-flight: required imports on this python.
    missing = []
    for mod in ("mlx_audio", "soundfile", "librosa"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        print(f"FAIL: missing imports on {sys.executable}: {', '.join(missing)}")
        print("Run: make setup-qwen3")
        return 1
    if not BRIDGE.is_file():
        print(f"FAIL: bridge script missing: {BRIDGE}")
        return 1
    ref = Path(args.reference).expanduser().resolve()
    if not ref.is_file():
        male_ref = ROOT / "voices" / "male_reference.wav"
        if male_ref.is_file():
            ref = male_ref
        else:
            print(f"FAIL: reference WAV missing: {ref}")
            return 1

    model_path = _resolve_model(args.model_path)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else \
        Path(tempfile.mkdtemp(prefix="qwen3-smoke-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    out_wav = work_dir / "smoke_segment.wav"
    queue_path = work_dir / "queue.json"
    manifest_path = work_dir / "manifest.json"

    # Build a one-item queue in the same format the split pipeline uses.
    queue = [{
        "line": 1,
        "text": args.text,
        "role": "clone",
        "ref_wav": ref.as_posix(),
        "voice_reference": ref.as_posix(),
        "start_time": 0,
        "end_time": 4000,
        "rate": "+0%",
        "volume": "+0%",
        "pitch": "+0Hz",
        "tts_type": 1,
        "filename": out_wav.as_posix(),
    }]
    queue_path.write_text(json.dumps(queue, ensure_ascii=False), encoding="utf-8")

    print(f"Python:     {sys.executable}")
    print(f"Model:      {model_path}")
    print(f"Reference:  {ref}")
    print(f"Text:       {args.text!r}")
    print(f"Output:     {out_wav}")
    print("Generating (this loads the model; first run may take a while) ...")

    cmd = [
        sys.executable, BRIDGE.as_posix(),
        "--queue-tts-file", queue_path.as_posix(),
        "--manifest-file", manifest_path.as_posix(),
        "--work-dir", work_dir.as_posix(),
        "--model-path", model_path,
        "--language", args.language,
        "--logs-file", (work_dir / "bridge.log").as_posix(),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"FAIL: bridge exited {result.returncode}")
        print("--- stderr (tail) ---")
        print(result.stderr[-2000:])
        print("--- stdout (tail) ---")
        print(result.stdout[-2000:])
        return 1

    if not out_wav.is_file():
        print(f"FAIL: bridge exited 0 but no output WAV at {out_wav}")
        print(result.stdout[-2000:])
        return 1

    # Validate the WAV: non-empty, real audio, sane duration.
    try:
        import soundfile as sf
        data, sr = sf.read(out_wav.as_posix())
    except Exception as exc:
        print(f"FAIL: could not read output WAV: {exc}")
        return 1
    duration = len(data) / float(sr) if sr else 0.0
    size = out_wav.stat().st_size
    print(f"WAV: {size} bytes, {sr} Hz, {duration:.2f}s, {len(data)} samples")

    if size < 1000:
        print(f"FAIL: output WAV too small ({size} bytes)")
        return 1
    if duration < 0.3:
        print(f"FAIL: output WAV too short ({duration:.2f}s)")
        return 1
    # Check it's not pure silence.
    try:
        peak = max(abs(float(x)) for x in data[: min(len(data), 48000)])
    except Exception:
        peak = 0.0
    if peak < 1e-4:
        print(f"FAIL: output WAV appears silent (peak {peak:.6f})")
        return 1

    print(f"PASS: Qwen3-TTS generated a real {duration:.2f}s segment "
          f"(peak {peak:.4f})")
    print(f"  -> {out_wav}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
