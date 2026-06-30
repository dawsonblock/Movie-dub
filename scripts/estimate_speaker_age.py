#!/usr/bin/env python3
"""Standalone age-estimation CLI for a single speaker reference WAV.

Wraps the optional griko/age_reg_ann_ecapa_librosa_combined regressor. Run
inside the .venv-age venv (created by ``make setup-age``):

    source .venv-age/bin/activate
    python scripts/estimate_speaker_age.py \
        --audio voices/male_reference.wav \
        --model-path models/age_reg_ann_ecapa_librosa_combined \
        --output tmp/age_test.json

The model predicts apparent VOCAL age from audio (SpeechBrain ECAPA-TDNN
embedding + Librosa acoustic features + ANN regressor). Test MAE ~6.93
years on the combined VoxCeleb2 + TIMIT set. Treat the output as an
approximate apparent age, not an exact biological age.

Run only on clean per-speaker reference clips — never on the whole movie,
background audio, mixed music/effects, overlapping dialogue, or
TTS-generated audio used as the source profile.

Output JSON:
    {
      "estimated_years": 42.7,
      "band": "adult",
      "method": "griko/age_reg_ann_ecapa_librosa_combined",
      "confidence": null,
      "note": "Apparent vocal age estimate; not exact biological age."
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the shared plugin logic so the load is cached and the method string
# stays consistent with analyze_speakers.py / age_model.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from age_model import DEFAULT_AGE_MODEL_REPO  # noqa: E402


def age_band(age: float) -> str:
    """Map a numeric age to a coarse band."""
    if age < 13:
        return "child"
    if age < 20:
        return "teen"
    if age < 35:
        return "young_adult"
    if age < 60:
        return "adult"
    return "senior"


def estimate_age(audio_path: Path, model_path: str | None = None) -> dict:
    """Run the trained regressor on one WAV. Raises on failure (no fallback)."""
    from age_regressor import AgeRegressionPipeline

    source = model_path or DEFAULT_AGE_MODEL_REPO
    regressor = AgeRegressionPipeline.from_pretrained(source)
    result = regressor(str(audio_path))
    age = float(result[0])
    return {
        "estimated_years": round(age, 1),
        "band": age_band(age),
        "method": DEFAULT_AGE_MODEL_REPO if model_path is None else model_path,
        "confidence": None,
        "note": "Apparent vocal age estimate; not exact biological age.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate apparent vocal age from a speaker reference WAV"
    )
    parser.add_argument("--audio", required=True,
                        help="Speaker reference WAV (16 kHz mono preferred)")
    parser.add_argument(
        "--model-path", default=None,
        help="Local model dir, e.g. models/age_reg_ann_ecapa_librosa_combined "
             "(default: download from HuggingFace by repo id)",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        print(f"Missing audio file: {audio_path}", file=sys.stderr)
        return 1

    try:
        result = estimate_age(audio_path, args.model_path)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"age estimation failed: {exc}", file=sys.stderr)
        print("Did you run `make setup-age` and activate .venv-age?",
              file=sys.stderr)
        return 1

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
