#!/usr/bin/env python3
"""Generate pyVideoTrans TTS segment WAV files with OpenVoice V2.

This script intentionally has no pyVideoTrans imports. It is designed to run
inside the OpenVoice virtual environment while pyVideoTrans remains in its own
environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SUPPORTED_LANGUAGES = {"EN", "ES", "FR", "ZH", "JP", "KR", "EN_NEWEST"}


def write_log(path: str | None, text: str, msg_type: str = "logs") -> None:
    if not path:
        return
    try:
        Path(path).write_text(json.dumps({"type": msg_type, "text": text}), encoding="utf-8")
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


def duration_status(generated_duration: float | None, target_duration: float | None) -> tuple[float | None, str]:
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


def normalize_language(language: str | None) -> str:
    if not language:
        return "EN"
    value = language.strip().upper().replace("-", "_")
    language_map = {
        "EN_US": "EN",
        "EN_GB": "EN",
        "EN_AU": "EN",
        "EN_IN": "EN",
        "ZH_CN": "ZH",
        "ZH_TW": "ZH",
        "JA": "JP",
        "KO": "KR",
        "KO_KR": "KR",
    }
    value = language_map.get(value, value.split("_")[0])
    if value not in SUPPORTED_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"OpenVoice V2 language '{language}' is not supported. Supported: {supported}")
    return value


def resolve_reference(item: dict[str, Any], default_reference: str | None) -> str:
    role = str(item.get("role", "")).strip().lower()
    ref = item.get("ref_wav") if role == "clone" else None
    ref = ref or item.get("voice_reference") or default_reference
    if not ref:
        raise ValueError("No OpenVoice reference WAV configured")
    ref_path = Path(str(ref)).expanduser()
    if not ref_path.is_file():
        raise FileNotFoundError(f"OpenVoice reference audio not found: {ref_path}")
    return ref_path.as_posix()


def load_openvoice(checkpoint_dir: str, device: str):
    import torch
    from openvoice.api import ToneColorConverter

    converter_dir = validate_checkpoint_dir(checkpoint_dir)
    checkpoint_path = converter_dir / "checkpoint.pth"

    converter = ToneColorConverter((converter_dir / "config.json").as_posix(), device=device)
    converter.watermark_model = None
    converter.load_ckpt(checkpoint_path.as_posix())
    return converter, torch


def validate_checkpoint_dir(checkpoint_dir: str) -> Path:
    ckpt_dir = Path(checkpoint_dir).expanduser()
    converter_dir = ckpt_dir / "converter"
    config_path = converter_dir / "config.json"
    checkpoint_path = converter_dir / "checkpoint.pth"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "OpenVoice V2 converter checkpoint is missing. Expected "
            f"{config_path} and {checkpoint_path}"
        )
    return converter_dir


def synthesize_segments(args: argparse.Namespace) -> int:
    queue = read_json(args.queue_tts_file)
    if not isinstance(queue, list):
        raise ValueError("queue_tts_file must contain a JSON list")

    openvoice_repo = Path(args.openvoice_repo).expanduser().resolve()
    if openvoice_repo.as_posix() not in sys.path:
        sys.path.insert(0, openvoice_repo.as_posix())

    language = normalize_language(args.language)
    base_speaker = args.base_speaker.strip() or None
    speed = float(args.speed)
    validate_checkpoint_dir(args.checkpoint_dir)

    import torch
    from melo.api import TTS

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    converter, torch_mod = load_openvoice(args.checkpoint_dir, device)
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    if not speaker_ids:
        raise RuntimeError(f"No MeloTTS speakers available for language {language}")
    if not base_speaker:
        base_speaker = next(iter(speaker_ids.keys()))
    if base_speaker not in speaker_ids:
        raise ValueError(
            f"Base speaker '{base_speaker}' is not available for {language}. "
            f"Available: {', '.join(speaker_ids.keys())}"
        )

    speaker_id = speaker_ids[base_speaker]
    speaker_key = base_speaker.lower().replace("_", "-")
    source_se_path = Path(args.checkpoint_dir).expanduser() / "base_speakers" / "ses" / f"{speaker_key}.pth"
    if not source_se_path.is_file():
        raise FileNotFoundError(f"OpenVoice source speaker embedding not found: {source_se_path}")
    source_se = torch_mod.load(source_se_path.as_posix(), map_location=device)

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_file).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    target_se_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    ok = 0
    err = 0

    for index, item in enumerate(queue):
        segment_id = item.get("line", item.get("id", index + 1))
        text = str(item.get("text") or item.get("target_text") or "").strip()
        output_audio = item.get("openvoice_output") or item.get("output_audio") or item.get("filename")
        result = {
            "index": index,
            "id": segment_id,
            "output_audio": output_audio,
            "status": "pending",
        }
        if not text:
            result["status"] = "skipped_empty_text"
            results.append(result)
            continue
        if not output_audio:
            result["status"] = "error"
            result["error"] = "No output path in segment"
            results.append(result)
            err += 1
            continue

        try:
            write_log(args.logs_file, f"OpenVoice {index + 1}/{len(queue)}")
            ref = resolve_reference(item, args.default_reference)
            if ref not in target_se_cache:
                target_se_cache[ref] = converter.extract_se(ref)
            target_se = target_se_cache[ref]

            output_path = Path(str(output_audio)).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = work_dir / f"openvoice-src-{index:06d}.wav"
            model.tts_to_file(text, speaker_id, tmp_path.as_posix(), speed=speed)
            converter.convert(
                audio_src_path=tmp_path.as_posix(),
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path.as_posix(),
                message=args.watermark,
            )

            generated = wav_duration(output_path.as_posix())
            start = item.get("start_time", item.get("start"))
            end = item.get("end_time", item.get("end"))
            target_duration = None
            if start is not None and end is not None:
                target_duration = max(0.0, (float(end) - float(start)) / (1000.0 if float(end) > 1000 else 1.0))
            ratio, timing = duration_status(generated, target_duration)

            result.update(
                {
                    "status": "ok",
                    "reference_audio": ref,
                    "generated_duration": generated,
                    "target_duration": target_duration,
                    "duration_ratio": None if ratio is None or math.isnan(ratio) else ratio,
                    "timing_status": timing,
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
        "created_at": time.time(),
        "ok": ok,
        "error": err,
        "language": language,
        "base_speaker": base_speaker,
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_log(args.logs_file, f"OpenVoice finished: ok={ok}, error={err}")
    return 0 if ok > 0 and err == 0 else 2 if ok > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate OpenVoice WAVs for pyVideoTrans TTS segments")
    parser.add_argument("--queue-tts-file", required=True)
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--openvoice-repo", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--default-reference", default="")
    parser.add_argument("--language", default="EN")
    parser.add_argument("--base-speaker", default="")
    parser.add_argument("--speed", default="1.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logs-file", default="")
    parser.add_argument("--watermark", default="@MyShell")
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
