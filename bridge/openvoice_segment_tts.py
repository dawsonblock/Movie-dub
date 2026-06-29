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
import re
import shutil
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


def return_code_for_counts(ok: int, err: int, skipped: int = 0) -> int:
    if ok > 0 and err == 0 and skipped == 0:
        return 0
    if ok > 0 and (err > 0 or skipped > 0):
        return 2
    return 3


def segment_times_seconds(item: dict[str, Any]) -> tuple[float, float] | tuple[None, None]:
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


def resolve_device(device: str):
    import torch

    requested = (device or "auto").strip().lower()
    if requested != "auto":
        return device, torch
    if torch.cuda.is_available():
        return "cuda:0", torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch
    return "cpu", torch


def load_openvoice(checkpoint_dir: str, device: str):
    import torch
    from openvoice.api import ToneColorConverter

    converter_dir = validate_checkpoint_dir(checkpoint_dir)
    checkpoint_path = converter_dir / "checkpoint.pth"

    converter = ToneColorConverter((converter_dir / "config.json").as_posix(), device=device)
    converter.watermark_model = None
    converter.load_ckpt(checkpoint_path.as_posix())
    return converter, torch


def load_runtime(args: argparse.Namespace, device: str, language: str, base_speaker: str | None):
    from melo.api import TTS

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
    return converter, torch_mod, model, speaker_id, source_se, base_speaker


def validate_checkpoint_dir(checkpoint_dir: str) -> Path:
    """Validate OpenVoice V2 checkpoint directory (Blocker 5).

    Checks that:
    - converter/config.json exists and is valid JSON
    - converter/checkpoint.pth exists, is not a Git LFS pointer, and is
      large enough to be a real checkpoint
    - base_speakers/ses/ has at least one .pth embedding file

    Raises FileNotFoundError with a descriptive message if any check fails.
    """
    ckpt_dir = Path(checkpoint_dir).expanduser()
    converter_dir = ckpt_dir / "converter"
    config_path = converter_dir / "config.json"
    checkpoint_path = converter_dir / "checkpoint.pth"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"OpenVoice V2 converter config is missing: {config_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"OpenVoice V2 converter checkpoint is missing: {checkpoint_path}"
        )
    # Reject Git LFS / Xet pointer files
    MIN_CHECKPOINT_BYTES = 1_000_000  # 1 MB
    raw = checkpoint_path.read_bytes()[:512]
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise FileNotFoundError(
            f"OpenVoice checkpoint is a Git LFS pointer, not a real file: "
            f"{checkpoint_path}. Run: make download-openvoice"
        )
    if raw.startswith(b"https://huggingface.co/xet"):
        raise FileNotFoundError(
            f"OpenVoice checkpoint is an Xet pointer, not a real file: "
            f"{checkpoint_path}. Run: make download-openvoice"
        )
    ckpt_size = checkpoint_path.stat().st_size
    if ckpt_size < MIN_CHECKPOINT_BYTES:
        raise FileNotFoundError(
            f"OpenVoice checkpoint is too small ({ckpt_size} bytes), "
            f"likely corrupted or a stub: {checkpoint_path}. "
            f"Run: make download-openvoice"
        )
    # Validate config.json is parseable JSON
    try:
        json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FileNotFoundError(
            f"OpenVoice config.json is not valid JSON: {config_path}: {exc}"
        )
    # Check base speaker embeddings exist
    ses_dir = ckpt_dir / "base_speakers" / "ses"
    if ses_dir.is_dir():
        pth_files = list(ses_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(
                f"No base speaker embedding .pth files in {ses_dir}. "
                f"Run: make download-openvoice"
            )
        for pth in pth_files:
            pth_raw = pth.read_bytes()[:512]
            if pth_raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise FileNotFoundError(
                    f"Base speaker embedding is a Git LFS pointer: {pth}. "
                    f"Run: make download-openvoice"
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

    device, _ = resolve_device(args.device)
    fallback_reason = ""
    try:
        converter, torch_mod, model, speaker_id, source_se, base_speaker = load_runtime(
            args, device, language, base_speaker
        )
    except Exception as exc:
        if device != "mps":
            raise
        fallback_reason = f"MPS runtime setup failed, retried on CPU: {exc}"
        write_log(args.logs_file, fallback_reason, "error")
        device = "cpu"
        converter, torch_mod, model, speaker_id, source_se, base_speaker = load_runtime(
            args, device, language, base_speaker
        )

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_file).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    target_se_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    ok = 0
    err = 0
    skipped = 0

    for index, item in enumerate(queue):
        segment_id = item.get("line", item.get("id", index + 1))
        text = str(item.get("text") or item.get("target_text") or "").strip()
        output_audio = item.get("openvoice_output") or item.get("output_audio") or item.get("filename")
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
            "reference_voice": item.get("ref_wav") or item.get("voice_reference") or args.default_reference,
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
            write_log(args.logs_file, f"OpenVoice {index + 1}/{len(queue)}")
            ref = resolve_reference(item, args.default_reference)
            if ref not in target_se_cache:
                target_se_cache[ref] = converter.extract_se(ref)
            target_se = target_se_cache[ref]

            output_path = Path(str(output_audio)).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = work_dir / f"openvoice-src-{index:06d}.wav"
            model.tts_to_file(text, speaker_id, tmp_path.as_posix(), speed=speed)
            try:
                converter.convert(
                    audio_src_path=tmp_path.as_posix(),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=output_path.as_posix(),
                    message=args.watermark,
                )
            except Exception as exc:
                if device != "mps":
                    raise
                fallback_reason = f"MPS segment generation failed, retried on CPU: {exc}"
                write_log(args.logs_file, fallback_reason, "error")
                device = "cpu"
                converter, torch_mod, model, speaker_id, source_se, base_speaker = load_runtime(
                    args, device, language, base_speaker
                )
                target_se_cache.clear()
                target_se = converter.extract_se(ref)
                target_se_cache[ref] = target_se
                model.tts_to_file(text, speaker_id, tmp_path.as_posix(), speed=speed)
                converter.convert(
                    audio_src_path=tmp_path.as_posix(),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=output_path.as_posix(),
                    message=args.watermark,
                )

            generated = wav_duration(output_path.as_posix())
            ratio, timing = duration_status(generated, target_duration)

            preserved_audio = ""
            if args.preserve_dir:
                preserve_dir = Path(args.preserve_dir).expanduser().resolve()
                preserve_dir.mkdir(parents=True, exist_ok=True)
                safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(segment_id)).strip("_") or "segment"
                preserved_path = preserve_dir / f"{int(index) + 1:06d}-{safe_id}.wav"
                shutil.copy2(output_path.as_posix(), preserved_path.as_posix())
                preserved_audio = preserved_path.as_posix()

            result.update(
                {
                    "status": "ok",
                    "reference_audio": ref,
                    "device": device,
                    "generated_duration": generated,
                    "duration_ratio": None if ratio is None or math.isnan(ratio) else ratio,
                    "timing_status": timing,
                    "preserved_audio": preserved_audio,
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
        "status": "ok" if ok > 0 and err == 0 else ("partial" if ok > 0 else "error"),
        "created_at": time.time(),
        "ok": ok,
        "error": err,
        "skipped": skipped,
        "skipped_empty_text": skipped,
        "allow_partial": False,
        "language": language,
        "base_speaker": base_speaker,
        "device": device,
        "fallback_reason": fallback_reason or None,
        "results": results,
        "segments": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_log(args.logs_file, f"OpenVoice finished: ok={ok}, error={err}, skipped={skipped}")
    return return_code_for_counts(ok, err, skipped)


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
    parser.add_argument("--preserve-dir", default="", help="directory to copy each segment WAV for stable rebuild access")
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
