#!/usr/bin/env python3
"""Generate male and female reference voice samples using MeloTTS.

These are used as OpenVoice reference voices when gender is detected.
The reference voice determines the timbre/character of the dubbed voice.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICES_DIR = ROOT / "voices"


def generate_reference_voices() -> int:
    """Generate male and female reference WAVs using MeloTTS."""
    from melo.api import TTS

    # EN-US tends to produce a male-sounding voice
    # EN-Default tends to produce a female-sounding voice
    # EN-BR is another option
    samples = [
        ("male_reference.wav", "EN-US",
         "Hello, this is a reference voice sample for dubbing. "
         "The quick brown fox jumps over the lazy dog. "
         "Thank you for watching this video."),
        ("female_reference.wav", "EN-Default",
         "Hello, this is a reference voice sample for dubbing. "
         "The quick brown fox jumps over the lazy dog. "
         "Thank you for watching this video."),
    ]

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    model = TTS(language="EN", device="auto")

    for filename, speaker_id, text in samples:
        output_path = VOICES_DIR / filename
        if output_path.exists():
            print(f"  already exists: {output_path}")
            continue
        speaker = model.hps.data.spk2id[speaker_id]
        model.tts_to_file(text, speaker, output_path.as_posix(), speed=1.0)
        print(f"  generated: {output_path}")

    return 0


def main() -> int:
    print(f"Generating reference voices in {VOICES_DIR} ...")
    try:
        return generate_reference_voices()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
