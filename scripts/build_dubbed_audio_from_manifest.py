#!/usr/bin/env python3
"""Build a single dubbed audio track from a TTS manifest.

Reads the manifest, places every successful segment WAV at its correct
timestamp on a silent canvas the length of the input video, mixes the
segments together, normalizes, and writes dubbed_audio.wav.

Designed to run in a plain Python with soundfile + numpy, or with only the
standard library (wave + array fallback).
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


def local_ffprobe() -> str:
    """Find ffprobe: prefer bundled, then PATH, then common Homebrew locations."""
    candidates = [
        Path(__file__).resolve().parents[1] / "pyvideotrans-main" / "ffmpeg" / "ffprobe",
        Path(shutil.which("ffprobe") or ""),
        Path("/opt/homebrew/bin/ffprobe"),
        Path("/usr/local/bin/ffprobe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    raise RuntimeError("ffprobe not found")


SUPPORTED_SAMPLE_RATES = (44100, 48000, 24000, 22050, 16000)
DEFAULT_SAMPLE_RATE = 44100


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _demucs_vocal_separation(
    input_video: Path, output_dir: Path, timeout: int = 600,
    allow_download: bool = False,
) -> Path | None:
    """Use Demucs (AI source separation) to extract instrumental/no-vocals track.

    Demucs separates audio into stems: vocals, drums, bass, other.
    We combine drums+bass+other into a single "no vocals" WAV.

    Returns path to the no-vocals WAV, or None if separation failed.

    If allow_download is False (the default), refuses to trigger a runtime
    model download. If the model is not already cached, returns None so the
    caller falls back to ffmpeg.
    """
    # First extract audio from video
    raw_audio = output_dir / "demucs_input.wav"
    cmd = [
        local_ffmpeg(), "-hide_banner", "-nostdin", "-y",
        "-i", input_video.as_posix(),
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        "-f", "wav",
        raw_audio.as_posix(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or not raw_audio.is_file():
        return None

    # Run Demucs separation
    # demucs --two-stems vocals --name demucs_run -o <output_dir> <input>
    # This creates: output_dir/demucs_run/<filename>/no_vocals.wav and vocals.wav
    try:
        import demucs  # noqa: F401 - just checking if demucs is available
    except ImportError:
        # Fall back to CLI
        demucs_cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "-o", output_dir.as_posix(),
            raw_audio.as_posix(),
        ]
        result = subprocess.run(demucs_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
    else:
        # Use Python API
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Stronger cache preflight (Blocker 8): check the local Demucs cache
        # directory BEFORE calling get_model(), which may itself trigger a
        # network resolution/download depending on Demucs internals. If the
        # model is not cached and downloads are refused, fall back to ffmpeg
        # without ever calling get_model().
        def _demucs_cache_present() -> bool:
            """Check if the htdemucs model is already in the torch cache."""
            cache_dirs: list[Path] = []
            # torch.hub cache
            hub_dir = Path(torch.hub.get_dir()) if hasattr(torch.hub, "get_dir") else None
            if hub_dir:
                cache_dirs.append(hub_dir / "checkpoints")
            # macOS CoreAudio/torch cache fallback
            home = Path.home()
            cache_dirs.append(home / "Library" / "Caches" / "torch" / "hub" / "checkpoints")
            cache_dirs.append(home / ".cache" / "torch" / "hub" / "checkpoints")
            for d in cache_dirs:
                if d.is_dir():
                    for p in d.iterdir():
                        if "htdemucs" in p.name.lower() and p.stat().st_size > 1_000_000:
                            return True
            return False

        if not allow_download and not _demucs_cache_present():
            print(
                "WARNING: Demucs htdemucs model not found in local cache and "
                "--allow-demucs-download not set. Falling back to ffmpeg "
                "without calling get_model().",
                file=sys.stderr,
            )
            raw_audio.unlink(missing_ok=True)
            return None

        # Model is cached or download is allowed; safe to call get_model
        try:
            model = get_model("htdemucs")
        except Exception as exc:
            if not allow_download:
                print(
                    f"WARNING: Demucs model not cached and --allow-demucs-download "
                    f"not set. Falling back to ffmpeg. Error: {exc}",
                    file=sys.stderr,
                )
                raw_audio.unlink(missing_ok=True)
                return None
            # Re-raise to let the caller's exception boundary handle it
            raise
        model.to(device)

        import torchaudio
        wav, sr = torchaudio.load(raw_audio.as_posix())
        wav = wav.unsqueeze(0).to(device)  # (1, channels, samples)
        with torch.no_grad():
            sources = apply_model(model, wav, split=True, overlap=0.25)
        # sources shape: (1, n_sources, channels, samples)
        # htdemucs: [drums, bass, other, vocals]
        # no_vocals = drums + bass + other
        no_vocals = sources[0, 0] + sources[0, 1] + sources[0, 2]
        no_vocals_path = output_dir / "no_vocals.wav"
        torchaudio.save(no_vocals_path.as_posix(), no_vocals.cpu(), sr)
        raw_audio.unlink(missing_ok=True)
        return no_vocals_path

    # Find the no_vocals output from CLI
    demucs_out = output_dir / "htdemucs" / "demucs_input"
    no_vocals = demucs_out / "no_vocals.wav"
    if no_vocals.is_file():
        raw_audio.unlink(missing_ok=True)
        return no_vocals
    return None


def _mix_background_audio(
    canvas, input_video: Path, sample_rate: int, total_samples: int, volume: float,
    vocal_separation: bool = False,
    vocal_separation_method: str = "ffmpeg",
    ducking: bool = False,
    speech_regions: list[tuple[int, int]] | None = None,
    timeout: int = 300,
    demucs_timeout: int = 600,
    allow_demucs_download: bool = False,
) -> dict:
    """Extract original audio from the input video and mix it under the speech canvas.

    If vocal_separation is True, removes vocals from the background before mixing.
    vocal_separation_method controls the technique:
      - "ffmpeg": center-channel removal via pan filter (fast, lower quality)
      - "demucs": AI-based source separation (slower, much better quality)
    If ducking is True, lowers the background volume where speech is present
    using a smooth envelope derived from speech_regions.

    Returns a dict with:
      - ``background_mix_applied``: bool
      - ``demucs_requested``: bool
      - ``demucs_used``: bool
      - ``demucs_error``: str (empty if no error)
      - ``ffmpeg_fallback_used``: bool
      - ``warnings``: list of warning dicts
    """
    import subprocess
    import tempfile

    result = {
        "background_mix_applied": False,
        "demucs_requested": vocal_separation and vocal_separation_method == "demucs",
        "demucs_used": False,
        "demucs_error": "",
        "ffmpeg_fallback_used": False,
        "warnings": [],
    }

    bg_path: Path | None = None
    demucs_dir: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            bg_path = Path(tmp.name)

        # --- Demucs path with controlled exception boundary (Blocker 2) ---
        if result["demucs_requested"]:
            demucs_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
            try:
                no_vocals = _demucs_vocal_separation(
                    input_video, demucs_dir, timeout=demucs_timeout,
                    allow_download=allow_demucs_download,
                )
                if no_vocals and no_vocals.is_file():
                    # Convert no_vocals to the target sample rate and mono
                    cmd = [
                        local_ffmpeg(), "-hide_banner", "-nostdin", "-y",
                        "-i", no_vocals.as_posix(),
                        "-ac", "1",
                        "-ar", str(sample_rate),
                        "-f", "wav",
                        bg_path.as_posix(),
                    ]
                    conv = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=timeout
                    )
                    if conv.returncode == 0 and bg_path.is_file():
                        result["demucs_used"] = True
                if not result["demucs_used"]:
                    raise RuntimeError("Demucs produced no usable output")
            except Exception as exc:
                # Demucs failed (import error, inference crash, etc.).
                # Warn and fall back to ffmpeg. Never kill the job unless
                # the user explicitly requests fail-hard behavior.
                result["demucs_error"] = str(exc)
                result["warnings"].append({
                    "stage": "demucs",
                    "message": str(exc),
                    "fallback": "ffmpeg",
                })
                print(
                    f"WARNING: Demucs failed, falling back to ffmpeg: {exc}",
                    file=sys.stderr,
                )

        # --- ffmpeg fallback (always attempted if Demucs didn't produce output) ---
        if not result["demucs_used"]:
            result["ffmpeg_fallback_used"] = True
            if vocal_separation:
                # Remove center-channel vocals using FFmpeg's pan filter.
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
            ff_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if ff_result.returncode != 0 or not bg_path.is_file():
                return result  # background_mix_applied stays False

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
        result["background_mix_applied"] = True
        return result
    except Exception as exc:
        result["warnings"].append({
            "stage": "background_mix",
            "message": str(exc),
            "fallback": "none",
        })
        return result
    finally:
        if bg_path is not None:
            bg_path.unlink(missing_ok=True)
        if demucs_dir is not None:
            shutil.rmtree(demucs_dir, ignore_errors=True)


def _apply_lufs_normalization(
    input_wav: Path, target_lufs: float = -16.0, timeout: int = 600,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bool:
    """Apply EBU R128 LUFS normalization via FFmpeg loudnorm filter.

    Normalizes input_wav in-place. Returns True if normalization succeeded.
    target_lufs: -16.0 is typical for web/streaming, -23.0 for EBU broadcast.
    sample_rate: the output sample rate (Blocker 6 — must preserve the
    selected sample rate, not default to 44100).
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
            "-ar", str(sample_rate),
            "-f", "wav",
            tmp_path.as_posix(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
    vocal_separation_method: str = "ffmpeg",
    ducking: bool = False,
    target_lufs: float | None = None,
    background_timeout: int = 300,
    lufs_timeout: int = 600,
    fail_if_background_mix_fails: bool = False,
    demucs_timeout: int = 600,
    allow_demucs_download: bool = False,
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

    # Optionally mix original audio (background music, ambience) under
    # the speech.
    background_mix = False
    background_warning = ""
    bg_report = {
        "background_mix_requested": background_volume > 0,
        "demucs_requested": vocal_separation and vocal_separation_method == "demucs",
        "demucs_used": False,
        "demucs_error": "",
        "ffmpeg_fallback_used": False,
        "background_mix_applied": False,
        "warnings": [],
    }
    if background_volume > 0:
        bg_result = _mix_background_audio(
            canvas, input_video, sample_rate, total_samples,
            background_volume,
            vocal_separation=vocal_separation,
            vocal_separation_method=vocal_separation_method,
            ducking=ducking,
            speech_regions=speech_regions if ducking else None,
            timeout=background_timeout,
            demucs_timeout=demucs_timeout,
            allow_demucs_download=allow_demucs_download,
        )
        bg_report.update(bg_result)
        background_mix = bg_result["background_mix_applied"]
        if not background_mix:
            background_warning = (
                "background audio mix failed (ffmpeg timeout or error); "
                "output will have no background audio"
            )
            if fail_if_background_mix_fails:
                raise RuntimeError(background_warning)
            print(f"WARNING: {background_warning}", file=sys.stderr)

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
    lufs_warning = ""
    if target_lufs is not None:
        lufs_applied = _apply_lufs_normalization(
            output_audio, target_lufs, timeout=lufs_timeout,
            sample_rate=sample_rate,
        )
        if not lufs_applied:
            lufs_warning = (
                "LUFS normalization failed (ffmpeg timeout or error); "
                "output keeps pre-LUFS levels"
            )
            print(f"WARNING: {lufs_warning}", file=sys.stderr)

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
        "background_mix_requested": bg_report["background_mix_requested"],
        "demucs_requested": bg_report["demucs_requested"],
        "demucs_used": bg_report["demucs_used"],
        "demucs_error": bg_report["demucs_error"],
        "ffmpeg_fallback_used": bg_report["ffmpeg_fallback_used"],
        "background_mix_applied": bg_report["background_mix_applied"],
        "voice_volume": voice_volume,
        "final_gain": final_gain,
        "no_normalize": no_normalize,
        "vocal_separation": vocal_separation,
        "vocal_separation_method": vocal_separation_method,
        "ducking": ducking,
        "target_lufs": target_lufs,
        "lufs_applied": lufs_applied,
        "warnings": [w for w in [background_warning, lufs_warning] if w]
                     + bg_report["warnings"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dubbed audio from a TTS manifest")
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
                        help="remove vocals from background audio "
                             "(uses --vocal-separation-method)")
    parser.add_argument("--vocal-separation-method",
                        choices=["ffmpeg", "demucs"], default="ffmpeg",
                        help="vocal removal technique: "
                             "ffmpeg=center-channel pan (fast), "
                             "demucs=AI source separation (better quality)")
    parser.add_argument("--demucs-timeout", type=int, default=600,
                        help="timeout for Demucs vocal separation "
                             "in seconds (default 600)")
    parser.add_argument("--allow-demucs-download", action="store_true",
                        help="allow Demucs to download its model at runtime "
                             "(default: refuse, fall back to ffmpeg)")
    parser.add_argument("--ducking", action="store_true",
                        help="lower background volume "
                             "where dubbed speech is present")
    parser.add_argument("--target-lufs", type=float, default=None,
                        help="apply EBU R128 LUFS normalization "
                             "(-16=web, -23=broadcast)")
    parser.add_argument("--background-timeout", type=int, default=300,
                        help="ffmpeg timeout for background audio "
                             "extraction in seconds (default 300)")
    parser.add_argument("--lufs-timeout", type=int, default=600,
                        help="ffmpeg timeout for LUFS normalization "
                             "in seconds (default 600)")
    parser.add_argument("--fail-if-background-mix-fails",
                        action="store_true",
                        help="raise an error if background audio "
                             "mixing fails instead of silently "
                             "producing speech-only output")
    parser.add_argument("--report-json", default="",
                        help="write the build report as JSON to this path")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    input_video = Path(args.input_video).expanduser().resolve()
    output_audio = Path(args.output_audio).expanduser().resolve()
    if args.work_dir:
        Path(args.work_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    try:
        report = build_dubbed_audio(
            manifest_path, input_video, output_audio,
            args.sample_rate, args.allow_partial,
            background_volume=args.background_volume,
            voice_volume=args.voice_volume,
            final_gain=args.final_gain,
            no_normalize=args.no_normalize,
            vocal_separation=args.vocal_separation,
            vocal_separation_method=args.vocal_separation_method,
            ducking=args.ducking,
            target_lufs=args.target_lufs,
            background_timeout=args.background_timeout,
            lufs_timeout=args.lufs_timeout,
            fail_if_background_mix_fails=args.fail_if_background_mix_fails,
            demucs_timeout=args.demucs_timeout,
            allow_demucs_download=args.allow_demucs_download,
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
            print(f"Vocal separation: on ({args.vocal_separation_method})")
        if args.ducking:
            print("Ducking: on")
        if args.target_lufs is not None:
            status = "applied" if report.get("lufs_applied") else "failed"
            print(f"LUFS: {args.target_lufs} ({status})")
        if args.report_json:
            report_path = Path(args.report_json).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Report: {report_path}")
        return 0
    except Exception as exc:
        print("Build dubbed audio: FAIL", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
