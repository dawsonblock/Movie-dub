#!/usr/bin/env python3
"""Run age_model.estimate_age in the .venv-age interpreter.

This helper is executed by age_model.py when the calling interpreter cannot
import ``voice_age_regressor`` (e.g. the pyVideoTrans venv). It reads the
reference WAV + pitch profile from CLI arguments and prints the JSON result.

The script is intentionally minimal: it depends only on the project-local
scripts/age_model.py and the .venv-age site-packages (which provide
voice_age_regressor + torch/tensorflow/speechbrain).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from age_model import (  # noqa: E402
    DEFAULT_AGE_MODEL_REPO,
    estimate_age,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Age model subprocess runner")
    parser.add_argument("--reference-wav", required=True,
                        help="Path to the speaker reference WAV")
    parser.add_argument("--pitch-json", required=True,
                        help="JSON object with median_f0_hz, voiced_ratio, "
                             "p10_f0_hz, p90_f0_hz")
    parser.add_argument("--model-path", default=None,
                        help="Local model directory (default: models/)")
    parser.add_argument("--repo", default=DEFAULT_AGE_MODEL_REPO,
                        help="HuggingFace repo id (default: "
                             f"{DEFAULT_AGE_MODEL_REPO})")
    args = parser.parse_args()

    ref_path = Path(args.reference_wav)
    try:
        pitch = json.loads(args.pitch_json)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid pitch JSON: {exc}"}), file=sys.stderr)
        return 1

    try:
        result = estimate_age(
            ref_path,
            pitch,
            use_model="on",
            repo=args.repo,
            model_path=args.model_path,
        )
    except Exception as exc:  # noqa: BLE001 - subprocess boundary
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    # Emit the JSON result on stdout for the caller to parse.
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
