#!/usr/bin/env python3
"""Optional age-regression plugin with pitch-heuristic fallback.

This module provides a single entry point, :func:`estimate_age`, that returns
an age estimate for a speaker reference WAV. It tries the real trained age
regressor first and falls back to the existing pitch-based heuristic when the
model is unavailable.

Real model (optional plugin):
    ``griko/age_reg_ann_ecapa_librosa_combined`` — a SpeechBrain ECAPA-TDNN
    embedding + ANN regressor trained on VoxCeleb2 + TIMIT. Test MAE ~6.93
    years. Installed via::

        pip install "voice-age-regressor[full] @ git+https://github.com/griko/voice-age-regression.git"

    Usage::

        from voice_age_regressor import AgeRegressionPipeline
        regressor = AgeRegressionPipeline.from_pretrained(
            "griko/age_reg_ann_ecapa_librosa_combined"
        )
        result = regressor("path/to/audio.wav")  # -> list[float]

Fallback (always available, no extra deps beyond librosa):
    The pitch-based heuristic from ``analyze_speakers.estimate_age_band``.

The plugin is loaded lazily and only once per process. If the import or model
load fails for any reason (missing package, no network, wrong torch version,
corrupted download), the fallback is used silently and the failure is recorded
so callers can report it.

This module imports no third-party packages at module load time, so it is safe
to import from the bare system interpreter (doctor.py, run_personal_dub.py).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

DEFAULT_AGE_MODEL_REPO = "griko/age_reg_ann_ecapa_librosa_combined"
# Default local model dir (populated by `make setup-age` / setup_age_model.sh).
# Preferred over the HF repo id when present: avoids a runtime download.
DEFAULT_AGE_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "models"
    / "age_reg_ann_ecapa_librosa_combined"
)

# Module-level singleton state. The age regressor is expensive to load
# (downloads ECAPA-TDNN weights + the ANN), so we cache it per process.
# The cache is keyed by the source string (repo id or local path) so a
# caller can force a different source by resetting first.
_lock = threading.Lock()
_regressor: Any | None = None
_loaded_source: str = ""
_load_attempted: bool = False
_load_error: str = ""


class AgeEstimate(dict):
    """Typed dict-ish result. Use plain dict access in callers; this subclass
    only exists so ``isinstance`` checks and ``__repr__`` are friendlier."""

    __slots__ = ()


def _resolve_source(model_path: str | None, repo: str) -> str:
    """Pick the model source: a local dir if provided/present, else the repo id."""
    if model_path:
        return model_path
    if DEFAULT_AGE_MODEL_DIR.is_dir():
        return DEFAULT_AGE_MODEL_DIR.as_posix()
    return repo


def _try_load_regressor(source: str) -> Any | None:
    """Attempt to import and load the age regressor. Returns None on failure."""
    global _load_attempted, _load_error, _loaded_source
    try:
        from voice_age_regressor import AgeRegressionPipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001 - any import failure is recoverable
        _load_error = f"voice_age_regressor import failed: {exc}"
        return None
    try:
        reg = AgeRegressionPipeline.from_pretrained(source)
        _loaded_source = source
        return reg
    except Exception as exc:  # noqa: BLE001
        _load_error = f"voice_age_regressor load failed ({source}): {exc}"
        return None


def get_regressor(
    repo: str = DEFAULT_AGE_MODEL_REPO,
    model_path: str | None = None,
) -> Any | None:
    """Return the cached age regressor, loading it on first call.

    ``model_path`` (a local dir) is preferred over ``repo`` (a HF repo id)
    when present. Returns None if the plugin is unavailable. The load is
    attempted exactly once per process per source; subsequent calls return
    the cached result (or None). Thread-safe.
    """
    global _regressor, _load_attempted
    source = _resolve_source(model_path, repo)
    with _lock:
        if not _load_attempted or _loaded_source != source:
            _load_attempted = True
            _regressor = _try_load_regressor(source)
        return _regressor


def reset_for_tests() -> None:
    """Reset the singleton so a fresh load can be attempted (tests only)."""
    global _regressor, _load_attempted, _load_error, _loaded_source
    with _lock:
        _regressor = None
        _load_attempted = False
        _load_error = ""
        _loaded_source = ""


def last_load_error() -> str:
    """Return the error from the most recent load attempt, or '' if none."""
    return _load_error


def loaded_source() -> str:
    """Return the source string the cached regressor was loaded from."""
    return _loaded_source


def is_available(
    repo: str = DEFAULT_AGE_MODEL_REPO,
    model_path: str | None = None,
) -> bool:
    """True if the age regressor plugin is loaded and ready."""
    return get_regressor(repo=repo, model_path=model_path) is not None


def _years_to_band(years: float) -> str:
    """Map a numeric age to a coarse band.

    Bands: child (<13) / teen (<20) / young_adult (<35) / adult (<60) /
    senior (60+). Matches estimate_speaker_age.age_band so the standalone
    wrapper and the pipeline produce identical bands.
    """
    if years <= 0:
        return "unknown"
    if years < 13:
        return "child"
    if years < 20:
        return "teen"
    if years < 35:
        return "young_adult"
    if years < 60:
        return "adult"
    return "senior"


def _heuristic_age(
    median_f0: float,
    voiced_ratio: float,
    p10_f0: float,
    p90_f0: float,
) -> tuple[str, float, float]:
    """Pitch-based fallback. Returns (band, estimated_years, confidence).

    Mirrors analyze_speakers.estimate_age_band so the fallback is identical to
    the pre-plugin behavior. Kept here to avoid a circular import
    (analyze_speakers imports this module).
    """
    if voiced_ratio < 0.3 or median_f0 <= 0:
        return ("unknown", 0.0, 0.0)
    f0_range = p90_f0 - p10_f0
    if median_f0 > 260:
        return ("child", 8.0, 0.4)
    if 220 < median_f0 <= 260 or (180 < median_f0 <= 220 and f0_range > 80):
        return ("teen", 15.0, 0.45)
    if 85 <= median_f0 <= 180 and voiced_ratio < 0.5:
        return ("senior", 70.0, 0.4)
    if 85 <= median_f0 <= 220:
        return ("adult", 35.0, 0.6)
    return ("adult", 35.0, 0.3)


def estimate_age(
    reference_wav: Path,
    pitch: dict,
    *,
    use_model: str = "auto",
    repo: str = DEFAULT_AGE_MODEL_REPO,
    model_path: str | None = None,
) -> dict:
    """Estimate apparent age for a speaker.

    Parameters
    ----------
    reference_wav
        Path to the speaker's reference WAV (16 kHz mono preferred).
    pitch
        Pitch profile dict from ``analyze_speakers.profile_pitch`` with keys
        ``median_f0_hz``, ``p10_f0_hz``, ``p90_f0_hz``, ``voiced_ratio``.
        Used for the fallback and for confidence blending.
    use_model
        ``"auto"``  -> try the model, fall back to heuristic on any failure.
        ``"on"``    -> require the model; raise if unavailable.
        ``"off"``   -> skip the model, use the heuristic directly.
    repo
        HuggingFace repo id for the age regressor (used when model_path is
        absent and no local dir exists).
    model_path
        Local model directory (e.g.
        ``models/age_reg_ann_ecapa_librosa_combined``). Preferred over the
        repo id when present — avoids a runtime download.

    Returns
    -------
    dict with keys:
        ``band``             - child/teen/young_adult/adult/senior/unknown
        ``estimated_years``  - float
        ``confidence``       - float in [0, 1]
        ``source``           - "model" | "heuristic"
        ``method``           - "griko/age_reg_ann_ecapa_librosa_combined"
                               (model) or "pitch_heuristic" (fallback)
        ``model_error``      - str, only present when source == "heuristic"
                               and a model load was attempted but failed
        ``note``             - str, only present when source == "model"
    """
    median_f0 = float(pitch.get("median_f0_hz", 0.0))
    voiced_ratio = float(pitch.get("voiced_ratio", 0.0))
    p10 = float(pitch.get("p10_f0_hz", 0.0))
    p90 = float(pitch.get("p90_f0_hz", 0.0))

    if use_model == "off":
        band, years, conf = _heuristic_age(median_f0, voiced_ratio, p10, p90)
        return {
            "band": band,
            "estimated_years": years,
            "confidence": conf,
            "source": "heuristic",
            "method": "pitch_heuristic",
        }

    # auto or on: try the model
    regressor = get_regressor(repo=repo, model_path=model_path)
    if regressor is None:
        if use_model == "on":
            raise RuntimeError(
                f"age model requested (--age-model on) but unavailable: "
                f"{last_load_error() or 'voice_age_regressor not installed'}. "
                f"Run `make setup-age` to install the plugin and download "
                f"the model into models/ (gitignored)."
            )
        band, years, conf = _heuristic_age(median_f0, voiced_ratio, p10, p90)
        result: dict = {
            "band": band,
            "estimated_years": years,
            "confidence": conf,
            "source": "heuristic",
            "method": "pitch_heuristic",
        }
        err = last_load_error()
        if err:
            result["model_error"] = err
        return result

    # Model available: predict. The regressor returns a list of ages.
    try:
        wav_path = Path(reference_wav)
        if not wav_path.is_file():
            raise RuntimeError(f"reference WAV not found: {wav_path}")
        results = regressor(wav_path.as_posix())
        if not results:
            raise RuntimeError("age regressor returned empty result")
        years = float(results[0])
    except Exception as exc:  # noqa: BLE001 - prediction failure is recoverable
        if use_model == "on":
            raise
        band, years_h, conf = _heuristic_age(median_f0, voiced_ratio, p10, p90)
        return {
            "band": band,
            "estimated_years": years_h,
            "confidence": conf,
            "source": "heuristic",
            "method": "pitch_heuristic",
            "model_error": f"prediction failed: {exc}",
        }

    band = _years_to_band(years)
    # Confidence: the model's MAE is ~6.93 years on the combined test set.
    # Map a generous uncertainty band to a confidence in [0.5, 0.9].
    # We don't have per-prediction uncertainty, so use voiced_ratio as a
    # proxy for signal quality (more voiced frames -> more reliable embedding).
    conf = round(min(0.9, 0.6 + voiced_ratio * 0.3), 2)
    return {
        "band": band,
        "estimated_years": round(years, 1),
        "confidence": conf,
        "source": "model",
        "method": loaded_source() or DEFAULT_AGE_MODEL_REPO,
        "note": "Apparent vocal age estimate; not exact biological age.",
    }


def cli_status() -> int:
    """Print plugin status and exit 0/1 (1 = unavailable). Used by doctor."""
    avail = is_available()
    print(f"age_model: {'available' if avail else 'unavailable'}")
    print(f"  repo: {DEFAULT_AGE_MODEL_REPO}")
    local = DEFAULT_AGE_MODEL_DIR
    print(f"  local dir: {local} ({'present' if local.is_dir() else 'missing'})")
    if avail:
        print(f"  loaded from: {loaded_source() or DEFAULT_AGE_MODEL_REPO}")
    else:
        err = last_load_error()
        if err:
            print(f"  last error: {err}")
        print("  fallback: pitch heuristic (always available)")
        print("  setup:    make setup-age")
    return 0 if avail else 1


if __name__ == "__main__":
    import sys
    sys.exit(cli_status())
