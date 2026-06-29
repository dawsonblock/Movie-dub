# Upstream Projects

This repository bundles the following upstream projects. Each has its own
license, contributing guidelines, and maintainers. Please respect their
terms when using or redistributing this bundle.

## pyVideoTrans

- **Repository:** https://github.com/jianchang512/pyvideotrans
- **License:** GPLv3 (see `pyvideotrans-main/LICENSE`)
- **Purpose:** Video transcription, translation, subtitle generation,
  timing, mixing, and export orchestration.
- **Local path:** `pyvideotrans-main/`

## OpenVoice

- **Repository:** https://github.com/myshell-ai/OpenVoice
- **License:** Apache 2.0 (see `OpenVoice-main/LICENSE`)
- **Purpose:** Local voice cloning and tone color conversion using
  OpenVoice V2. Also includes MeloTTS for base speech synthesis.
- **Local path:** `OpenVoice-main/`

## OpenVoice V2 Checkpoints

- **Repository:** https://huggingface.co/rsxdalv/OpenVoiceV2
- **License:** Refer to the upstream model card.
- **Purpose:** Pre-trained OpenVoice V2 converter and base speaker
  embeddings.
- **Local path:** `OpenVoice-main/checkpoints_v2/` (not committed; downloaded
  via `make download-openvoice`)

## ffmpeg / ffprobe

- **Repository:** https://ffmpeg.org/
- **License:** LGPL or GPL depending on build configuration.
- **Purpose:** Audio/video encoding, decoding, and remuxing.
- **Local path:** `pyvideotrans-main/ffmpeg/` (bundled binary, macOS)

## How to update upstreams

To update a bundled upstream project, replace the contents of its local
directory with a fresh checkout from the upstream repository. Then run
`make doctor` and `make proof` to verify the bundle still works together.
