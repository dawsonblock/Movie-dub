#!/usr/bin/env python3
"""Build a single dubbed audio track from an OpenVoice manifest.

Reads the manifest, places every successful segment WAV at its correct
timestamp on a silent canvas the length of the input video, mixes the
segments together, normalizes, and writes dubbed_audio.wav.

Designed to run in either the OpenVoice venv (soundfile + numpy) or a
plain Python with only the standard library (wave + array fallback).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


def local_ffmpeg() -> str:
    """Find ffmpeg: prefer bundled, then PATH, then common Homebrew locations."""
    candidates = [
        Path(__file__).resolve().parents[1] / "pyvideotrans-main" / "ffmpeg" / "ffmpeg",
        Path(shutil.which("ffmpeg") or ""),
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    raise RuntimeError("ffmpeg not found")


SUPPORTED_SAMPLE_RATES = (44100, 48000, 24000, 22050, 16000)
DEFAULT_SAMPLE_RATE = 44100


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def local_ffprobe() -> str:
    bundled = Path(__file__).resolve().parents[1] / "pyvideotrans-main" / "ffmpeg" / "ffprobe"
    return bundled.as_posix() if bundled.is_file() else "ffprobe"


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            local_ffprobe(),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            path.as_posix(),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return float(result.stdout.strip())


def manifest_segments(manifest: dict) -> list[dict]:
    """Return the segment list from either the new 'segments' key or legacy 'results'."""
    segments = manifest.get("segments")
    if isinstance(segments, list):
        return segments
    results = manifest.get("results")
    if isinstance(results, list):
        return results
    return []


def segment_start(seg: dict) -> float:
    if "start" in seg:
        return float(seg["start"])
    if "start_time" in seg:
        return float(seg["start_time"]) / 1000.0
    return 0.0


def segment_end(seg: dict) -> float:
    if "end" in seg:
        return float(seg["end"])
    if "end_time" in seg:
        return float(seg["end_time"]) / 1000.0
    start = segment_start(seg)
    target = seg.get("target_duration")
    if target is not None:
        return start + float(target)
    generated = seg.get("generated_duration")
    if generated is not None:
        return start + float(generated)
    return start


def resolve_output_audio(seg: dict, manifest_dir: Path) -> Path | None:
    # Prefer the preserved (stable) copy when the original temp WAV is gone.
    raw = (
        seg.get("preserved_audio")
        or seg.get("output_audio")
        or seg.get("filename")
        or seg.get("openvoice_output")
    )
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    # Try relative to cwd first, then relative to the manifest directory.
    if path.is_file():
        return path.resolve()
    return (manifest_dir / path).resolve()


# --- Audio I/O with soundfile/numpy preferred, stdlib fallback ---

_HAVE_SOUNDFILE: bool | None = None


def _have_soundfile() -> bool:
    global _HAVE_SOUNDFILE
    if _HAVE_SOUNDFILE is None:
        try:
            import soundfile  # noqa: F401
            import numpy  # noqa: F401
            _HAVE_SOUNDFILE = True
        except Exception:
            _HAVE_SOUNDFILE = False
    return _HAVE_SOUNDFILE


def load_audio_samples(path: Path, target_sr: int) -> tuple[list, int] | "object":
    """Load audio and resample (nearest) to target_sr. Returns (mono_samples_list, sr)."""
    if _have_soundfile():
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(path.as_posix(), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            ratio = target_sr / sr
            n_out = int(round(len(data) * ratio))
            idx = np.linspace(0, len(data) - 1, n_out)
            data = np.interp(idx, np.arange(len(data)), data).astype("float32")
        return data.tolist(), target_sr

    # stdlib fallback: PCM WAV only
    import array
    import wave

    with wave.open(path.as_posix(), "rb") as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sampwidth == 2:
        a = array.array("h")
        a.frombytes(frames)
    elif sampwidth == 1:
        a = array.array("B")
        a.frombytes(frames)
        a = array.array("h", [int((v - 128) * 257) for v in a])
    elif sampwidth == 4:
        a = array.array("i")
        a.frombytes(frames)
    else:
        raise RuntimeError(f"unsupported sample width {sampwidth} for {path}")
    if n_channels > 1:
        mono = [sum(a[i:i + n_channels]) // n_channels for i in range(0, len(a), n_channels)]
        a = array.array("h" if sampwidth <= 2 else "i", mono)
    # normalize to float
    scale = 32768.0 if a.typecode in ("h", "H") else 2147483648.0
    data = [v / scale for v in a]
    if sr != target_sr:
        ratio = target_sr / sr
        n_out = int(round(len(data) * ratio))
        out = []
        for i in range(n_out):
            src = i / ratio
            lo = int(src)
            hi = min(lo + 1, len(data) - 1)
            frac = src - lo
            out.append(data[lo] * (1 - frac) + data[hi] * frac)
        data = out
    return data, target_sr


def write_audio_samples(path: Path, samples, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _have_soundfile():
        import numpy as np
        import soundfile as sf

        arr = np.asarray(samples, dtype="float32")
        sf.write(path.as_posix(), arr, sr, subtype="PCM_16")
        return

    import array
    import wave

    arr = array.array("h")
    for v in samples:
        v = max(-1.0, min(1.0, float(v)))
        arr.append(int(v * 32767))
    with wave.open(path.as_posix(), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(arr.tobytes())


def normalize(samples, target_peak: float = 0.97):
    """Normalize peak to target_peak. Returns the same type as input (list or ndarray)."""
    if _have_soundfile():
        import numpy as np

        arr = np.asarray(samples, dtype="float32")
        peak = float(np.abs(arr).max()) if arr.size else 0.0
        if peak <= 0:
            return arr
        gain = target_peak / peak
        return arr * gain

    peak = 0.0
    for v in samples:
        a = abs(float(v))
        if a > peak:
            peak = a
    if peak <= 0:
        return list(samples)
    gain = target_peak / peak
    return [float(v) * gain for v in samples]


def _mix_background_audio(
    canvas, input_video: Path, sample_rate: int, total_samples: int, volume: float,
    vocal_separation: bool = False,
    ducking: bool = False,
    speech_regions: list[tuple[int, int]] | None = None,
) -> bool:
    """Extract original audio from the input video and mix it under the speech canvas.

    If vocal_separation is True, removes center-channel vocals from the
    background before mixing (karaoke-style center removal via FFmpeg).
    If ducking is True, lowers the background volume where speech is present
    using a smooth envelope derived from speech_regions.
    Returns True if mixing succeeded.
    """
    import subprocess
    import tempfile

    bg_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            bg_path = Path(tmp.name)

        if vocal_separation:
            # Remove center-channel vocals using FFmpeg's pan filter.
            # L-R and R-L cancel out centered (mono) vocals, leaving
            # mostly instrumental content. Downmix to mono afterward.
            cmd = [
                local_ffmpeg(), "-hide_banner", "-nostdin", "-y",
                "-i", input_video.as_posix(),
                "-vn",
                "-af", "pan=stereo|c0=c0-c1|c1=c1-c0",
                "-ac", "1",
                "-ar", str(sample_rate),
                "-f", "wav",
                bg_path.as_posix(),
            ]
        else:
            cmd = [
                local_ffmpeg(), "-hide_banner", "-nostdin", "-y",
                "-i", input_video.as_posix(),
                "-vn",
                "-ac", "1",
                "-ar", str(sample_rate),
                "-f", "wav",
                bg_path.as_posix(),
            ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not bg_path.is_file():
            return False

        bg_samples, _ = load_audio_samples(bg_path, sample_rate)

        if _have_soundfile():
            import numpy as np
            bg_arr = np.asarray(bg_samples, dtype="float32") * volume

            if ducking and speech_regions:
                # Create a ducking envelope: 1.0 where no speech,
                # duck_level (0.3) where speech is present, with smooth
                # transitions to avoid clicks.
                duck_level = 0.3
                fade_samples = int(0.05 * sample_rate)  # 50ms fade
                envelope = np.ones(total_samples, dtype="float32")
                for start_idx, end_idx in speech_regions:
                    s = max(0, start_idx - fade_samples)
                    e = min(total_samples, end_idx + fade_samples)
                    envelope[start_idx:end_idx] = duck_level
                    # Smooth fade in
                    if start_idx > 0:
                        fade_len = min(fade_samples, start_idx - s)
                        if fade_len > 0:
                            envelope[s:start_idx] = np.linspace(
                                1.0, duck_level, fade_len
                            )
                    # Smooth fade out
                    if end_idx < total_samples:
                        fade_len = min(fade_samples, e - end_idx)
                        if fade_len > 0:
                            envelope[end_idx:e] = np.linspace(
                                duck_level, 1.0, fade_len
                            )
                n = min(len(bg_arr), total_samples)
                bg_arr[:n] *= envelope[:n]

            n = min(len(bg_arr), total_samples)
            canvas[:n] += bg_arr[:n]
        else:
            n = min(len(bg_samples), total_samples)
            for i in range(n):
                canvas[i] += float(bg_samples[i]) * volume
        return True
    except Exception:
        return False
    finally:
        if bg_path is not None:
            bg_path.unlink(missing_ok=True)


def _apply_lufs_normalization(
    input_wav: Path, target_lufs: float = -16.0
) -> bool:
    """Apply EBU R128 LUFS normalization via FFmpeg loudnorm filter.

    Normalizes input_wav in-place. Returns True if normalization succeeded.
    target_lufs: -16.0 is typical for web/streaming, -23.0 for EBU broadcast.
    """
    import subprocess
    import tempfile

    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_path = Path(tmp_name)

        cmd = [
            local_ffmpeg(), "-hide_banner", "-nostdin", "-y",
            "-i", input_wav.as_posix(),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", str(DEFAULT_SAMPLE_RATE),
            "-f", "wav",
            tmp_path.as_posix(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not tmp_path.is_file():
            return False
        # Replace the original with the normalized version
        shutil.move(tmp_path.as_posix(), input_wav.as_posix())
        tmp_path = None  # consumed by move
        return True
    except Exception:
        return False
    finally:
        if tmp_path is not None and tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)


def build_dubbed_audio(
    manifest_path: Path,
    input_video: Path,
    output_audio: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    allow_partial: bool = False,
    background_volume: float = 0.0,
    voice_volume: float = 1.0,
    final_gain: float = 1.0,
    no_normalize: bool = False,
    vocal_separation: bool = False,
    ducking: bool = False,
    target_lufs: float | None = None,
) -> dict:
    manifest = read_json(manifest_path)
    segments = manifest_segments(manifest)
    manifest_dir = manifest_path.parent

    video_duration = ffprobe_duration(input_video)
    total_samples = int(math.ceil(video_duration * sample_rate))

    # Start with a silent canvas.
    if _have_soundfile():
        import numpy as np
        canvas = np.zeros(total_samples, dtype="float32")
    else:
        canvas = [0.0] * total_samples

    placed = 0
    skipped = 0
    failed = 0
    errors: list[str] = []
    speech_regions: list[tuple[int, int]] = []  # (start_idx, end_idx) for ducking

    for seg in segments:
        status = seg.get("status", "ok")
        if status not in {"ok", "regenerated"}:
            # skipped, skipped_empty_text, and error all require --allow-partial
            # to prevent silent missing lines in the final video
            msg = f"segment {seg.get('id')} has status '{status}'"
            if status in {"skipped", "skipped_empty_text"}:
                skipped += 1
            else:
                failed += 1
            if allow_partial:
                errors.append(msg)
                continue
            raise RuntimeError(
                f"{msg}; use --allow-partial to skip failed/skipped segments"
            )

        wav_path = resolve_output_audio(seg, manifest_dir)
        if not wav_path or not wav_path.is_file():
            msg = f"segment {seg.get('id')}: output audio missing: {wav_path}"
            if allow_partial:
                failed += 1
                errors.append(msg)
                continue
            raise RuntimeError(msg)

        samples, sr = load_audio_samples(wav_path, sample_rate)
        start = segment_start(seg)
        start_index = int(round(start * sample_rate))
        end_index = min(start_index + len(samples), total_samples)
        if start_index < 0:
            # clip leading samples that fall before the canvas
            offset = -start_index
            samples = samples[offset:]
            start_index = 0
        if start_index >= total_samples:
            continue

        if _have_soundfile():
            import numpy as np
            seg_arr = np.asarray(samples, dtype="float32") * voice_volume
            canvas[start_index:end_index] += seg_arr[: end_index - start_index]
        else:
            for i, v in enumerate(samples):
                idx = start_index + i
                if idx >= total_samples:
                    break
                canvas[idx] += float(v) * voice_volume
        speech_regions.append((start_index, end_index))
        placed += 1

    if placed == 0:
        raise RuntimeError("no segments were placed on the audio canvas")

    # Optionally mix original audio (background music, ambience) under the speech.
    background_mix = False
    if background_volume > 0:
        background_mix = _mix_background_audio(
            canvas, input_video, sample_rate, total_samples, background_volume,
            vocal_separation=vocal_separation,
            ducking=ducking,
            speech_regions=speech_regions if ducking else None,
        )

    if no_normalize:
        # Skip peak normalization — apply final_gain directly
        if _have_soundfile():
            import numpy as np
            final = np.asarray(canvas, dtype="float32") * final_gain
        else:
            final = [float(v) * final_gain for v in canvas]
    else:
        normalized = normalize(canvas)
        if _have_soundfile():
            import numpy as np
            final = np.asarray(normalized, dtype="float32") * final_gain
        else:
            final = [float(v) * final_gain for v in normalized]
    write_audio_samples(output_audio, final, sample_rate)

    # Optionally apply EBU R128 LUFS normalization as a post-processing step
    lufs_applied = False
    if target_lufs is not None:
        lufs_applied = _apply_lufs_normalization(output_audio, target_lufs)

    return {
        "status": "ok",
        "input_video": input_video.as_posix(),
        "output_audio": output_audio.as_posix(),
        "manifest": manifest_path.as_posix(),
        "video_duration": video_duration,
        "sample_rate": sample_rate,
        "segments_total": len(segments),
        "segments_placed": placed,
        "segments_skipped": skipped,
        "segments_failed": failed,
        "errors": errors,
        "background_volume": background_volume,
        "background_mix": background_mix,
        "voice_volume": voice_volume,
        "final_gain": final_gain,
        "no_normalize": no_normalize,
        "vocal_separation": vocal_separation,
        "ducking": ducking,
        "target_lufs": target_lufs,
        "lufs_applied": lufs_applied,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dubbed audio from an OpenVoice manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-audio", required=True)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--background-volume", type=float, default=0.0,
                        help="mix original audio at this volume "
                             "(0.0=off, 0.15=quiet bg, 1.0=full)")
    parser.add_argument("--voice-volume", type=float, default=1.0,
                        help="gain on dubbed speech "
                             "(1.0=normal, 1.5=louder, 0.5=quieter)")
    parser.add_argument("--final-gain", type=float, default=1.0,
                        help="gain applied to the final mix "
                             "after normalization (1.0=no change)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="skip peak normalization "
                             "(use with --final-gain for manual control)")
    parser.add_argument("--vocal-separation", action="store_true",
                        help="remove center-channel vocals "
                             "from background audio (karaoke-style)")
    parser.add_argument("--ducking", action="store_true",
                        help="lower background volume "
                             "where dubbed speech is present")
    parser.add_argument("--target-lufs", type=float, default=None,
                        help="apply EBU R128 LUFS normalization (-16=web, -23=broadcast)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    input_video = Path(args.input_video).expanduser().resolve()
    output_audio = Path(args.output_audio).expanduser().resolve()
    if args.work_dir:
        Path(args.work_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    try:
        report = build_dubbed_audio(
            manifest_path, input_video, output_audio, args.sample_rate, args.allow_partial,
            background_volume=args.background_volume,
            voice_volume=args.voice_volume,
            final_gain=args.final_gain,
            no_normalize=args.no_normalize,
            vocal_separation=args.vocal_separation,
            ducking=args.ducking,
            target_lufs=args.target_lufs,
        )
        print("Build dubbed audio: PASS")
        print(f"Output: {output_audio}")
        print(f"Placed {report['segments_placed']}/{report['segments_total']} segments")
        print(f"Duration: {report['video_duration']:.3f}s @ {report['sample_rate']}Hz")
        if args.background_volume > 0:
            print(f"Background volume: {args.background_volume}")
        if args.voice_volume != 1.0:
            print(f"Voice volume: {args.voice_volume}")
        if args.final_gain != 1.0:
            print(f"Final gain: {args.final_gain}")
        if args.no_normalize:
            print("Normalize: off")
        if args.vocal_separation:
            print("Vocal separation: on (center removal)")
        if args.ducking:
            print("Ducking: on")
        if args.target_lufs is not None:
            status = "applied" if report.get("lufs_applied") else "failed"
            print(f"LUFS: {args.target_lufs} ({status})")
        return 0
    except Exception as exc:
        print("Build dubbed audio: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
