#!/usr/bin/env python3
"""Generate per-speaker TTS segment WAV files with OmniVoice.

This script is the OmniVoice equivalent of openvoice_segment_tts.py and
qwen3_segment_tts.py. It accepts the same --queue-tts-file / --manifest-file
interface so the split pipeline in run_personal_dub.py can use it as a
drop-in replacement.

OmniVoice is a remote/server TTS engine. Two modes are supported:

  1. OmniVoice-studio (default port 3900): direct HTTP POST to
     ``{api_url}/generate`` with multipart form data. Requires only
     ``requests``.

  2. Gradio web UI: uses ``gradio_client.Client.predict(..., api_name="/_clone_fn")``.
     Requires ``gradio_client`` and ``requests``.

The studio mode is auto-detected when --api-url ends with ``:3900`` or when
--studio is passed. Otherwise the Gradio client path is used.

Dependencies:
    pip install gradio_client requests

Example:
    python bridge/omnivoice_segment_tts.py \
        --queue-tts-file tmp/job/openvoice_queue.json \
        --manifest-file tmp/job/openvoice_manifest.json \
        --work-dir tmp/job/openvoice_work \
        --api-url http://localhost:3900 \
        --language en
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


# Map short language codes to OmniVoice language names (used in both modes).
LANG_CODE_MAP = {
    "auto": "Auto",
    "en": "English",
    "en_us": "English",
    "en_gb": "English",
    "zh": "Chinese",
    "zh_cn": "Chinese",
    "zh_tw": "Min Nan Chinese",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "ko_kr": "Korean",
    "ru": "Russian",
    "es": "Spanish",
    "th": "Thai",
    "it": "Italian",
    "pt": "Portuguese",
    "vi": "Vietnamese",
    "ar": "Standard Arabic",
    "tr": "Turkish",
    "hi": "Hindi",
    "hu": "Hungarian",
    "uk": "Ukrainian",
    "id": "Indonesian",
    "ms": "Malay",
    "kk": "Kazakh",
    "cs": "Czech",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "he": "Hebrew",
    "bn": "Bengali",
    "fil": "Filipino",
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "az": "Azerbaijani",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan",
    "hr": "Croatian",
    "da": "Danish",
    "et": "Estonian",
    "fi": "Finnish",
    "gl": "Galician",
    "ka": "Georgian",
    "el": "Greek",
    "gu": "Gujarati",
    "is": "Icelandic",
    "iu": "Inuktitut",
    "ga": "Irish",
    "jv": "Javanese",
    "kn": "Kannada",
    "km": "Khmer",
    "lo": "Lao",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mr": "Marathi",
    "mn": "Mongolian",
    "ne": "Nepali",
    "nb": "Norwegian Bokmål",
    "ps": "Pashto",
    "fa": "Persian",
    "ro": "Romanian",
    "sr": "Serbian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "su": "Sudanese Arabic",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "uz": "Uzbek",
    "cy": "Welsh",
    "zu": "Zulu",
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
        return "Auto"
    key = language.strip().lower().replace("-", "_")
    return LANG_CODE_MAP.get(key, "Auto")


def resolve_reference(item: dict[str, Any], default_reference: str | None) -> str:
    role = str(item.get("role", "")).strip().lower()
    ref = item.get("ref_wav") if role == "clone" else None
    ref = ref or item.get("voice_reference") or default_reference
    if not ref:
        raise ValueError("No OmniVoice reference WAV configured")
    ref_path = Path(str(ref)).expanduser()
    if not ref_path.is_file():
        raise FileNotFoundError(f"OmniVoice reference audio not found: {ref_path}")
    return ref_path.as_posix()


def _is_studio_url(api_url: str, studio: bool) -> bool:
    """Return True if the URL should be treated as OmniVoice-studio."""
    if studio:
        return True
    url = api_url.rstrip("/")
    return url.endswith(":3900")


def _synthesize_studio(
    api_url: str,
    text: str,
    lang: str,
    ref_wav: str | None,
    ref_text: str | None,
    num_steps: int,
    guidance_scale: float,
    speed: float,
    denoise: bool,
    postprocess: bool,
    timeout: float,
    output_path: Path,
) -> None:
    """Studio-mode HTTP POST to /generate."""
    import requests

    data = {
        "text": text,
        "language": lang,
        "num_step": num_steps,
        "guidance_scale": guidance_scale,
        "speed": speed,
        "denoise": "true" if denoise else "false",
        "postprocess_output": "true" if postprocess else "false",
    }
    if ref_text:
        data["ref_text"] = ref_text

    files = None
    try:
        if ref_wav:
            files = {"ref_audio": open(ref_wav, "rb")}
        response = requests.post(
            f"{api_url.rstrip('/')}/generate",
            data=data,
            files=files,
            timeout=timeout,
            proxies={"https": "", "http": ""},
        )
    finally:
        if files:
            files["ref_audio"].close()

    if not response.ok:
        raise RuntimeError(f"OmniVoice studio returned {response.status_code}: {response.text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.wav")
    tmp_path.write_bytes(response.content)
    # Studio returns a WAV directly, but normalize to the requested path.
    shutil.move(tmp_path.as_posix(), output_path.as_posix())


def _synthesize_gradio(
    api_url: str,
    text: str,
    lang: str,
    ref_wav: str | None,
    ref_text: str | None,
    num_steps: int,
    guidance_scale: float,
    speed: float,
    denoise: bool,
    postprocess: bool,
    timeout: float,
    output_path: Path,
) -> None:
    """Gradio-client path to an OmniVoice Gradio server."""
    from gradio_client import handle_file
    from gradio_client import Client

    kwargs = {
        "text": text,
        "lang": lang,
        "ref_aud": handle_file(ref_wav) if ref_wav else None,
        "ref_text": ref_text or "",
        "instruct": "",
        "ns": num_steps,
        "gs": guidance_scale,
        "dn": denoise,
        "sp": speed,
        "du": 0,
        "pp": postprocess,
        "po": False,
        "api_name": "/_clone_fn",
    }

    client = Client(api_url, httpx_kwargs={"timeout": timeout}, ssl_verify=False)
    result = client.predict(**kwargs)
    wav_file = result[0] if isinstance(result, (list, tuple)) and result else result
    if isinstance(wav_file, dict) and "value" in wav_file:
        wav_file = wav_file["value"]
    if not isinstance(wav_file, str) or not Path(wav_file).is_file():
        raise RuntimeError(f"OmniVoice Gradio returned unexpected result: {result!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav_file, output_path.as_posix())


def _synthesize(
    args: argparse.Namespace,
    text: str,
    lang: str,
    ref_wav: str | None,
    ref_text: str | None,
    output_path: Path,
) -> None:
    """Dispatch to the correct OmniVoice API mode."""
    studio = _is_studio_url(args.api_url, args.studio)
    if studio:
        _synthesize_studio(
            api_url=args.api_url,
            text=text,
            lang=lang,
            ref_wav=ref_wav,
            ref_text=ref_text,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            speed=args.speed,
            denoise=args.denoise,
            postprocess=args.postprocess,
            timeout=args.timeout,
            output_path=output_path,
        )
    else:
        _synthesize_gradio(
            api_url=args.api_url,
            text=text,
            lang=lang,
            ref_wav=ref_wav,
            ref_text=ref_text,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            speed=args.speed,
            denoise=args.denoise,
            postprocess=args.postprocess,
            timeout=args.timeout,
            output_path=output_path,
        )


def synthesize_segments(args: argparse.Namespace) -> int:
    queue = read_json(args.queue_tts_file)
    if not isinstance(queue, list):
        raise ValueError("queue_tts_file must contain a JSON list")

    language = normalize_language(args.language)

    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_file).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

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
            item.get("openvoice_output")
            or item.get("output_audio")
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
            write_log(args.logs_file, f"OmniVoice {index + 1}/{len(queue)}")
            ref = resolve_reference(item, args.default_reference)
            ref_text = item.get("ref_text") or None

            output_path = Path(str(output_audio)).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            _synthesize(
                args,
                text=text,
                lang=language,
                ref_wav=ref,
                ref_text=ref_text,
                output_path=output_path,
            )

            generated = wav_duration(output_path.as_posix())
            ratio, timing = duration_status(generated, target_duration)

            preserved_audio = ""
            if args.preserve_dir:
                preserve_dir = Path(args.preserve_dir).expanduser().resolve()
                preserve_dir.mkdir(parents=True, exist_ok=True)
                safe_id = (
                    re.sub(r"[^A-Za-z0-9_.-]+", "_", str(segment_id))
                    .strip("_")
                    or "segment"
                )
                preserved_path = preserve_dir / f"{int(index) + 1:06d}-{safe_id}.wav"
                shutil.copy2(output_path.as_posix(), preserved_path.as_posix())
                preserved_audio = preserved_path.as_posix()

            result.update(
                {
                    "status": "ok",
                    "reference_audio": ref,
                    "device": "api",
                    "generated_duration": generated,
                    "duration_ratio": (
                        None if ratio is None or math.isnan(ratio)
                        else ratio
                    ),
                    "timing_status": timing,
                    "preserved_audio": preserved_audio,
                    "tts_engine": "omnivoice",
                    "api_url": args.api_url,
                    "studio": _is_studio_url(args.api_url, args.studio),
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
        "device": "api",
        "tts_engine": "omnivoice",
        "api_url": args.api_url,
        "studio": _is_studio_url(args.api_url, args.studio),
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
        f"OmniVoice finished: ok={ok}, error={err}, skipped={skipped}",
    )
    return return_code_for_counts(ok, err, skipped)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-speaker TTS WAV segments with OmniVoice."
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
        "--api-url", required=True,
        help="OmniVoice server URL (e.g. http://localhost:3900 or "
             "http://localhost:7860 for a Gradio UI)",
    )
    parser.add_argument(
        "--default-reference", default="",
        help="default reference WAV path (fallback when item has no ref_wav)",
    )
    parser.add_argument(
        "--language", default="auto",
        help="language code (en, zh, ja, ko, etc.)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="speech speed multiplier (default 1.0)",
    )
    parser.add_argument(
        "--num-steps", type=int, default=32,
        help="number of diffusion steps (default 32)",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=2.0,
        help="guidance scale (default 2.0)",
    )
    parser.add_argument(
        "--denoise", action="store_true",
        help="enable denoising",
    )
    parser.add_argument(
        "--postprocess", action="store_true", default=True,
        help="enable post-processing (default True)",
    )
    parser.add_argument(
        "--no-postprocess", action="store_true",
        help="disable post-processing",
    )
    parser.add_argument(
        "--studio", action="store_true",
        help="force OmniVoice-studio mode (direct HTTP /generate)",
    )
    parser.add_argument(
        "--timeout", type=float, default=3600,
        help="API request timeout in seconds (default 3600)",
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
    parser = build_parser()
    args = parser.parse_args()
    if args.no_postprocess:
        args.postprocess = False
    try:
        return synthesize_segments(args)
    except Exception as exc:
        write_log(args.logs_file, str(exc), "error")
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
