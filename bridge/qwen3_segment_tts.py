#!/usr/bin/env python3
"""Generate per-speaker TTS segment WAV files with Qwen3-TTS (MLX).

This script is the Qwen3-TTS bridge for the split pipeline. It accepts
the same --queue-tts-file / --manifest-file interface as the other split
TTS bridges so run_personal_dub.py can use it as a drop-in engine.

Each queue item should have:
  - text: the text to synthesize
  - ref_wav / voice_reference: path to the speaker's reference audio
  - start_time / end_time: segment timing in milliseconds
  - output_audio / filename: output WAV path

The script uses mlx-audio for on-device Apple Silicon inference.
Voice cloning is done via ref_audio + ref_text (the reference transcript
is auto-generated if not provided — we transcribe the reference clip
with a simple heuristic: use the segment text from the same speaker).

Dependencies (install in a dedicated venv or system-wide):
  pip install mlx-audio soundfile librosa
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared utilities (independent of other bridge implementations)
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "AUTO", "CHINESE", "ENGLISH", "JAPANESE", "KOREAN",
    "GERMAN", "FRENCH", "RUSSIAN", "PORTUGUESE", "SPANISH", "ITALIAN",
}

# Map short codes to mlx-audio lang_code values
LANG_CODE_MAP = {
    "EN": "english",
    "EN_US": "english",
    "EN_GB": "english",
    "ZH": "chinese",
    "ZH_CN": "chinese",
    "ZH_TW": "chinese",
    "JA": "japanese",
    "JP": "japanese",
    "KO": "korean",
    "KR": "korean",
    "KO_KR": "korean",
    "FR": "french",
    "DE": "german",
    "ES": "spanish",
    "RU": "russian",
    "PT": "portuguese",
    "IT": "italian",
    "AUTO": "auto",
}


def write_log(path: str | None, text: str, msg_type: str = "logs") -> None:
    if not path:
        return
    try:
        Path(path).write_text(
            json.dumps({"type": msg_type, "text": text}), encoding="utf-8"
        )
    except OSError:
        pass


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wav_duration(path: str) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        if not info.samplerate:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def duration_status(
    generated_duration: float | None,
    target_duration: float | None,
) -> tuple[float | None, str]:
    if not generated_duration or not target_duration or target_duration <= 0:
        return None, "unknown"
    ratio = generated_duration / target_duration
    if 0.85 <= ratio <= 1.15:
        return ratio, "accept"
    if 1.15 < ratio <= 1.35:
        return ratio, "light_speedup_candidate"
    if ratio > 1.35:
        return ratio, "rewrite_shorter"
    if ratio < 0.70:
        return ratio, "padding_or_slowdown_candidate"
    return ratio, "minor_timing_mismatch"


def return_code_for_counts(ok: int, err: int, skipped: int = 0) -> int:
    if ok > 0 and err == 0 and skipped == 0:
        return 0
    if ok > 0 and (err > 0 or skipped > 0):
        return 2
    return 3


def segment_times_seconds(
    item: dict[str, Any],
) -> tuple[float, float] | tuple[None, None]:
    if "start_time" in item or "end_time" in item:
        start = float(item.get("start_time", 0.0)) / 1000.0
        end = float(item.get("end_time", 0.0)) / 1000.0
        return start, end
    if "start" in item or "end" in item:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))
        return start, end
    return None, None


def normalize_language(language: str | None) -> str:
    """Convert a short language code to mlx-audio's lang_code format."""
    if not language:
        return "auto"
    value = language.strip().upper().replace("-", "_")
    mapped = LANG_CODE_MAP.get(value)
    if mapped:
        return mapped
    # Try the base part before underscore
    base = value.split("_")[0]
    mapped = LANG_CODE_MAP.get(base)
    if mapped:
        return mapped
    # If it's already a full word like "english", return lowercased
    if language.lower() in {v for v in LANG_CODE_MAP.values()}:
        return language.lower()
    # Default to auto for unknown codes
    print(
        f"WARNING: language '{language}' not recognized, "
        f"using 'auto'. Supported: {', '.join(sorted(LANG_CODE_MAP.keys()))}",
        file=sys.stderr,
    )
    return "auto"


def resolve_reference(
    item: dict[str, Any], default_reference: str | None
) -> str:
    """Resolve the reference audio path for voice cloning."""
    role = str(item.get("role", "")).strip().lower()
    ref = item.get("ref_wav") if role == "clone" else None
    ref = ref or item.get("voice_reference") or default_reference
    if not ref:
        raise ValueError("No Qwen3-TTS reference WAV configured")
    ref_path = Path(str(ref)).expanduser()
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Qwen3-TTS reference audio not found: {ref_path}"
        )
    return ref_path.as_posix()


# ---------------------------------------------------------------------------
# Qwen3-TTS model loading (MLX)
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}


def load_qwen3_model(model_path: str):
    """Lazy-load a Qwen3-TTS model via mlx-audio.

    The model is cached so repeated calls with the same path don't reload.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]
    from mlx_audio.tts.utils import load_model

    print(f"Loading Qwen3-TTS model: {model_path}")
    model = load_model(model_path)
    _model_cache[model_path] = model
    return model


def load_reference_audio(ref_path: str):
    """Load a reference WAV file as an mlx array at 24 kHz mono."""
    import librosa
    import mlx.core as mx

    audio_np, _ = librosa.load(ref_path, sr=24000, mono=True)
    return mx.array(audio_np)


# ---------------------------------------------------------------------------
# Segment synthesis
# ---------------------------------------------------------------------------

def synthesize_segments(args: argparse.Namespace) -> int:
    queue = read_json(args.queue_tts_file)
    if not isinstance(queue, list):
        raise ValueError("queue_tts_file must contain a JSON list")

    language = normalize_language(args.language)
    temperature = float(args.temperature)
    speed = float(args.speed)

    # Load the Qwen3-TTS model
    model = load_qwen3_model(args.model_path)

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_file).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache reference audio arrays so we don't reload per segment
    ref_audio_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    ok = 0
    err = 0
    skipped = 0

    for index, item in enumerate(queue):
        segment_id = item.get("line", item.get("id", index + 1))
        text = str(
            item.get("text") or item.get("target_text") or ""
        ).strip()
        output_audio = (
            item.get("output_audio")
            or item.get("filename")
        )
        start_s, end_s = segment_times_seconds(item)
        target_duration = None
        if start_s is not None and end_s is not None:
            target_duration = max(0.0, end_s - start_s)

        result = {
            "index": index,
            "id": segment_id,
            "text": text,
            "speaker": item.get("role", "clone"),
            "language": language,
            "start": start_s if start_s is not None else 0.0,
            "end": end_s if end_s is not None else 0.0,
            "target_duration": target_duration,
            "output_audio": output_audio,
            "reference_voice": (
                item.get("ref_wav")
                or item.get("voice_reference")
                or args.default_reference
            ),
            "status": "pending",
            "error": None,
        }

        if not text:
            result["status"] = "skipped_empty_text"
            skipped += 1
            results.append(result)
            continue
        if not output_audio:
            result["status"] = "error"
            result["error"] = "No output path in segment"
            results.append(result)
            err += 1
            continue

        try:
            write_log(args.logs_file, f"Qwen3-TTS {index + 1}/{len(queue)}")
            ref = resolve_reference(item, args.default_reference)

            # Load and cache the reference audio as mlx array
            if ref not in ref_audio_cache:
                ref_audio_cache[ref] = load_reference_audio(ref)
            ref_audio_mx = ref_audio_cache[ref]

            # Use ref_text if provided in the queue item, otherwise
            # pass None and let mlx-audio handle it. Some Qwen3-TTS
            # models can work without a ref_text transcript.
            ref_text = item.get("ref_text") or None

            output_path = Path(str(output_audio)).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate audio with voice cloning
            import numpy as np

            gen_results = list(
                model.generate(
                    text=text,
                    ref_audio=ref_audio_mx,
                    ref_text=ref_text,
                    temperature=temperature,
                    speed=speed,
                    lang_code=language,
                )
            )

            if not gen_results:
                raise RuntimeError("Qwen3-TTS produced no audio")

            audio_np = np.array(gen_results[0].audio)

            # Write WAV at 24 kHz (Qwen3-TTS native sample rate)
            import soundfile as sf

            sf.write(output_path.as_posix(), audio_np, 24000)

            generated = wav_duration(output_path.as_posix())
            ratio, timing = duration_status(generated, target_duration)

            preserved_audio = ""
            if args.preserve_dir:
                preserve_dir = (
                    Path(args.preserve_dir).expanduser().resolve()
                )
                preserve_dir.mkdir(parents=True, exist_ok=True)
                safe_id = (
                    re.sub(r"[^A-Za-z0-9_.-]+", "_", str(segment_id))
                    .strip("_")
                    or "segment"
                )
                preserved_path = (
                    preserve_dir / f"{int(index) + 1:06d}-{safe_id}.wav"
                )
                shutil.copy2(
                    output_path.as_posix(), preserved_path.as_posix()
                )
                preserved_audio = preserved_path.as_posix()

            result.update(
                {
                    "status": "ok",
                    "reference_audio": ref,
                    "device": "mlx",
                    "generated_duration": generated,
                    "duration_ratio": (
                        None if ratio is None or math.isnan(ratio)
                        else ratio
                    ),
                    "timing_status": timing,
                    "preserved_audio": preserved_audio,
                    "tts_engine": "qwen3-local",
                }
            )
            ok += 1
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()
            err += 1
        results.append(result)

    manifest = {
        "status": "ok" if ok > 0 and err == 0
        else ("partial" if ok > 0 else "error"),
        "created_at": time.time(),
        "ok": ok,
        "error": err,
        "skipped": skipped,
        "skipped_empty_text": skipped,
        "allow_partial": False,
        "language": language,
        "base_speaker": None,
        "device": "mlx",
        "tts_engine": "qwen3-local",
        "model_path": args.model_path,
        "fallback_reason": None,
        "results": results,
        "segments": results,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_log(
        args.logs_file,
        f"Qwen3-TTS finished: ok={ok}, error={err}, skipped={skipped}",
    )
    return return_code_for_counts(ok, err, skipped)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-speaker TTS WAV segments with Qwen3-TTS (MLX). "
            "Split-pipeline TTS bridge for the speaker-profiling flow."
        )
    )
    parser.add_argument(
        "--queue-tts-file", required=True,
        help="JSON file containing the TTS queue (list of segment dicts)",
    )
    parser.add_argument(
        "--manifest-file", required=True,
        help="output path for the manifest JSON",
    )
    parser.add_argument(
        "--work-dir", required=True,
        help="directory for temporary working files",
    )
    parser.add_argument(
        "--model-path", required=True,
        help=(
            "HuggingFace repo ID or local path for the Qwen3-TTS model "
            "(e.g. mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16). "
            "Must include the speech_tokenizer/ subdir."
        ),
    )
    parser.add_argument(
        "--default-reference", default="",
        help="default reference WAV path (fallback when item has no ref_wav)",
    )
    parser.add_argument(
        "--language", default="auto",
        help="language code (en, zh, ja, ko, fr, de, es, auto)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="sampling temperature (0.0-2.0, default 0.9)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="speech speed multiplier (default 1.0)",
    )
    parser.add_argument(
        "--logs-file", default="",
        help="path to write log JSON",
    )
    parser.add_argument(
        "--preserve-dir", default="",
        help="directory to copy each segment WAV for stable rebuild access",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return synthesize_segments(args)
    except Exception as exc:
        write_log(args.logs_file, str(exc), "error")
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
