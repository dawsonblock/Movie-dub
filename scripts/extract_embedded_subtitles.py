#!/usr/bin/env python3
"""Extract embedded subtitles from a video as the text+timing source.

Embedded subtitles give us WHAT was said and WHEN — but never WHO said
it. Speaker identity comes from audio diarization downstream
(analyze_speakers.py). This module is only responsible for turning an
embedded subtitle stream into a source SRT.

Pipeline role (Step 0 of the split dub pipeline):

  video
    -> list subtitle streams        (ffprobe -select_streams s)
    -> pick best text-based stream  (prefer source-language match)
    -> ffmpeg -map 0:s:k -> source.srt
    -> if only image-based subs (PGS/VobSub): OCR, else signal ASR fallback
    -> if no subtitle stream at all: signal ASR fallback

Return contract:
  extract_embedded_subtitles(...) -> ExtractResult(
      srt_path=Path | None,        # None => run ASR fallback
      source="embedded_text" | "embedded_ocr" | None,
      stream_index=int | None,
      codec_name=str | None,
      language=str | None,
      note=str,                    # human-readable explanation
  )

Image-based OCR is best-effort. We try `pgs-to-srt` (pip) for PGS and
`vobsub2srt` for VobSub. If the tool is missing or OCR fails, we return
srt_path=None so the caller falls back to whisper ASR. We never fail
hard — embedded subs are an optimization, not a requirement.

Run with the system python (only needs ffprobe/ffmpeg on PATH):
  python scripts/extract_embedded_subtitles.py \
    --input video.mp4 --output-dir job/subtitles --source-language en
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# Text-based subtitle codecs ffmpeg can transcode to SRT directly.
TEXT_CODECS = {
    "subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "web_vtt",
    "stl", "subviewer", "subviewer1", "jacosub", "microdvd",
}
# Image-based subtitle codecs that require OCR.
IMAGE_CODECS = {
    "hdmv_pgs_subtitle", "pgs_subtitle", "dvd_subtitle", "vobsub",
    "dvb_subtitle", "dvbsub", "xsub", "hdmv_text_subtitle",
}


@dataclass
class ExtractResult:
    """Result of an embedded-subtitle extraction attempt."""
    srt_path: str | None
    source: str | None          # "embedded_text" | "embedded_ocr" | None
    stream_index: int | None
    codec_name: str | None
    language: str | None
    title: str | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for cand in ("/opt/homebrew/bin", "/usr/local/bin"):
        p = Path(cand) / name
        if p.is_file():
            return p.as_posix()
    return None


def list_subtitle_streams(video: Path, ffprobe: str | None = None) -> list[dict]:
    """List subtitle streams via ffprobe.

    Returns a list of dicts: {index, codec_name, language, title}.
    `language` is the raw tag (may be empty/und). `title` is the stream
    title tag (may be empty).
    """
    ffprobe = ffprobe or _find_binary("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_name:stream_tags=language,title",
        "-of", "json", video.as_posix(),
    ]
    result = subprocess.run(
        cmd, text=True, capture_output=True, check=True, timeout=60,
    )
    info = json.loads(result.stdout or "{}")
    streams: list[dict] = []
    for s in info.get("streams", []):
        tags = s.get("tags", {}) or {}
        streams.append({
            "index": int(s.get("index", -1)),
            "codec_name": s.get("codec_name", "") or "",
            "language": (tags.get("language") or "").strip().lower(),
            "title": (tags.get("title") or "").strip(),
        })
    return streams


def _lang_matches(stream_lang: str, source_language: str) -> bool:
    """Loose language match: 'en' matches 'en', 'eng', 'en-US', 'en-us'."""
    if not stream_lang or not source_language:
        return False
    src = source_language.lower().replace("-", "").replace("_", "")
    sl = stream_lang.lower().replace("-", "").replace("_", "")
    # ISO 639-1 (en) vs ISO 639-2/B (eng): match first 2 chars either way.
    return sl.startswith(src[:2]) or src.startswith(sl[:2])


def pick_subtitle_stream(
    streams: list[dict],
    source_language: str = "",
) -> tuple[dict | None, list[dict]]:
    """Pick the best subtitle stream.

    Preference order:
      1. Text-based stream whose language matches --source-language.
      2. Any text-based stream (language-agnostic).
      3. Image-based stream whose language matches --source-language.
      4. Any image-based stream.

    Returns (chosen_stream_or_None, image_based_candidates).
    `image_based_candidates` is the list of image-based streams (for OCR
    attempt) when no text-based stream was chosen.
    """
    text_streams = [s for s in streams if s["codec_name"] in TEXT_CODECS]
    image_streams = [s for s in streams if s["codec_name"] in IMAGE_CODECS]
    # Unknown codec: treat as text-tryable only if not in either set — be
    # conservative and let ffmpeg attempt a transcode. We classify unknowns
    # as text-attemptable but separate from the known-text list.
    unknown_streams = [
        s for s in streams
        if s["codec_name"] not in TEXT_CODECS and s["codec_name"] not in IMAGE_CODECS
    ]

    # 1. Text + language match.
    for s in text_streams:
        if _lang_matches(s["language"], source_language):
            return s, []
    # 2. Any text.
    if text_streams:
        return text_streams[0], []
    # Unknown codecs: try as text (ffmpeg will fail gracefully if not).
    for s in unknown_streams:
        if _lang_matches(s["language"], source_language):
            return s, image_streams
    if unknown_streams:
        return unknown_streams[0], image_streams
    # 3/4. Image-based only — return None here; caller decides OCR.
    return None, image_streams


def extract_text_subtitle(
    video: Path,
    stream_index: int,
    out_srt: Path,
    ffmpeg: str | None = None,
    timeout: int = 300,
) -> bool:
    """Extract a text subtitle stream to SRT via ffmpeg.

    Returns True on success (out_srt exists and is non-empty), False otherwise.
    """
    ffmpeg = ffmpeg or _find_binary("ffmpeg")
    if not ffmpeg:
        return False
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-v", "error",
        "-i", video.as_posix(),
        "-map", f"0:{stream_index}",
        "-c:s", "srt",
        "-y", out_srt.as_posix(),
    ]
    try:
        result = subprocess.run(
            cmd, text=True, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    return out_srt.is_file() and out_srt.stat().st_size > 0


def _ocr_pgs(video: Path, stream_index: int, out_srt: Path) -> bool:
    """OCR a PGS subtitle stream to SRT using pgs-to-srt (pip)."""
    # pgs-to-srt is a CLI; check it's importable as a command.
    pgs_cli = _find_binary("pgs-to-srt")
    if not pgs_cli:
        return False
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    # pgs-to-srt -i <video> -o <out> -si <stream_index>
    cmd = [
        pgs_cli,
        "-i", video.as_posix(),
        "-o", out_srt.with_suffix("").as_posix(),
        "-si", str(stream_index),
    ]
    try:
        result = subprocess.run(
            cmd, text=True, capture_output=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    # pgs-to-srt may write .srt or .sup.srt next to the -o base.
    candidates = [
        out_srt,
        out_srt.with_suffix(".srt"),
        Path(str(out_srt.with_suffix("")) + ".srt"),
    ]
    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 0:
            if cand != out_srt:
                shutil.move(cand.as_posix(), out_srt.as_posix())
            return True
    return False


def _ocr_vobsub(video: Path, stream_index: int, out_srt: Path) -> bool:
    """OCR a VobSub stream to SRT using vobsub2srt (Homebrew package)."""
    tool = _find_binary("vobsub2srt")
    if not tool:
        return False
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    # vobsub2srt works on .idx/.sub files, not directly on a video stream.
    # We first extract the vobsub to .idx/.sub with ffmpeg, then OCR.
    base = out_srt.with_suffix("")
    sub = Path(str(base) + ".sub")
    idx = Path(str(base) + ".idx")
    ffmpeg = _find_binary("ffmpeg")
    if not ffmpeg:
        return False
    extract = [
        ffmpeg, "-v", "error", "-i", video.as_posix(),
        "-map", f"0:{stream_index}", "-c:s", "dvd_subtitle",
        "-y", sub.as_posix(),
    ]
    try:
        r = subprocess.run(extract, text=True, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        _cleanup_vobsub_intermediates(sub, idx)
        return False
    if r.returncode != 0 or not sub.is_file():
        _cleanup_vobsub_intermediates(sub, idx)
        return False
    # vobsub2srt <basename> writes <basename>.srt
    cmd = [tool, base.as_posix()]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        _cleanup_vobsub_intermediates(sub, idx)
        return False
    srt_out = Path(str(base) + ".srt")
    ok = (
        result.returncode == 0
        and srt_out.is_file()
        and srt_out.stat().st_size > 0
    )
    if ok:
        shutil.move(srt_out.as_posix(), out_srt.as_posix())
    # Always clean up the intermediate .idx/.sub files (can be large).
    _cleanup_vobsub_intermediates(sub, idx)
    return ok if ok else False


def _cleanup_vobsub_intermediates(*paths: Path) -> None:
    """Remove intermediate VobSub extraction files."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def ocr_image_subtitle(
    video: Path,
    stream: dict,
    out_srt: Path,
) -> bool:
    """OCR an image-based subtitle stream to SRT. Best-effort.

    Returns True on success. If the required OCR tool is missing or OCR
    fails, returns False so the caller falls back to ASR.
    """
    codec = stream.get("codec_name", "")
    idx = int(stream.get("index", -1))
    if idx < 0:
        return False
    if codec in ("hdmv_pgs_subtitle", "pgs_subtitle"):
        return _ocr_pgs(video, idx, out_srt)
    if codec in ("dvd_subtitle", "vobsub"):
        return _ocr_vobsub(video, idx, out_srt)
    # Other image codecs (dvb_subtitle, xsub): no OCR path implemented.
    return False


def extract_embedded_subtitles(
    video: Path,
    output_dir: Path,
    source_language: str = "",
    ocr_image_subs: bool = True,
    ffprobe: str | None = None,
    ffmpeg: str | None = None,
) -> ExtractResult:
    """Extract an embedded subtitle stream to <output_dir>/source.srt.

    Returns an ExtractResult. srt_path is None when no usable embedded
    subtitle was found (caller should run ASR fallback). This function
    never raises on "no subtitles" — it only raises on tooling errors
    (ffprobe missing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_srt = output_dir / "source.srt"

    try:
        streams = list_subtitle_streams(video, ffprobe=ffprobe)
    except Exception as exc:
        return ExtractResult(
            srt_path=None, source=None, stream_index=None,
            codec_name=None, language=None, title=None,
            note=f"ffprobe failed: {exc}",
        )

    if not streams:
        return ExtractResult(
            srt_path=None, source=None, stream_index=None,
            codec_name=None, language=None, title=None,
            note="no subtitle streams found; ASR fallback required",
        )

    chosen, image_candidates = pick_subtitle_stream(streams, source_language)

    # Case 1: a text-based (or unknown-but-tryable) stream was chosen.
    if chosen is not None:
        ok = extract_text_subtitle(
            video, int(chosen["index"]), out_srt, ffmpeg=ffmpeg,
        )
        if ok:
            return ExtractResult(
                srt_path=out_srt.as_posix(),
                source="embedded_text",
                stream_index=int(chosen["index"]),
                codec_name=chosen["codec_name"] or None,
                language=chosen["language"] or None,
                title=chosen["title"] or None,
                note=(
                    f"extracted text subtitle stream "
                    f"{chosen['index']} ({chosen['codec_name']}, "
                    f"lang={chosen['language'] or 'und'})"
                ),
            )
        # Text extraction failed (unknown codec not actually text). Try
        # OCR if this stream is image-based; else fall through to image
        # candidates, then ASR.
        if chosen["codec_name"] in IMAGE_CODECS:
            image_candidates = [chosen] + image_candidates

    # Case 2: only image-based subtitles available (or text extraction
    # failed and the chosen stream was image-based). Try OCR.
    if ocr_image_subs and image_candidates:
        for stream in image_candidates:
            ok = ocr_image_subtitle(video, stream, out_srt)
            if ok:
                return ExtractResult(
                    srt_path=out_srt.as_posix(),
                    source="embedded_ocr",
                    stream_index=int(stream["index"]),
                    codec_name=stream["codec_name"] or None,
                    language=stream["language"] or None,
                    title=stream["title"] or None,
                    note=(
                        f"OCR'd image subtitle stream "
                        f"{stream['index']} ({stream['codec_name']}, "
                        f"lang={stream['language'] or 'und'})"
                    ),
                )
        return ExtractResult(
            srt_path=None, source=None, stream_index=None,
            codec_name=None, language=None, title=None,
            note=(
                f"found {len(image_candidates)} image-based subtitle "
                f"stream(s) but OCR failed or tooling missing "
                f"(need pgs-to-srt and/or vobsub2srt); ASR fallback "
                f"required"
            ),
        )

    # Case 3: image-based subs present but OCR disabled.
    if image_candidates and not ocr_image_subs:
        return ExtractResult(
            srt_path=None, source=None, stream_index=None,
            codec_name=None, language=None, title=None,
            note=(
                f"found {len(image_candidates)} image-based subtitle "
                f"stream(s); OCR disabled (--no-ocr-image-subs); ASR "
                f"fallback required"
            ),
        )

    # Case 4: text extraction failed and no image candidates.
    return ExtractResult(
        srt_path=None, source=None, stream_index=None,
        codec_name=None, language=None, title=None,
        note="subtitle extraction failed; ASR fallback required",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract embedded subtitles as a source SRT.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-language", default="")
    parser.add_argument("--no-ocr-image-subs", action="store_true",
                        help="skip OCR for image-based subs (force ASR fallback)")
    parser.add_argument("--json", action="store_true",
                        help="print result as JSON")
    args = parser.parse_args()

    video = args.input.expanduser().resolve()
    if not video.is_file():
        print(f"input not found: {video}", file=sys.stderr)
        return 1

    result = extract_embedded_subtitles(
        video=video,
        output_dir=args.output_dir.expanduser().resolve(),
        source_language=args.source_language,
        ocr_image_subs=not args.no_ocr_image_subs,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        if result.srt_path:
            print(f"OK: {result.note}")
            print(f"     -> {result.srt_path}")
        else:
            print(f"SKIP: {result.note}")
    return 0 if result.srt_path else 2


if __name__ == "__main__":
    sys.exit(main())
