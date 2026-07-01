.PHONY: doctor doctor-strict doctor-demucs doctor-checkpoints doctor-qwen3 setup-ffmpeg setup-pyvt setup-openvoice setup-qwen3 setup-omnivoice setup-checkpoints download-openvoice smoke-openvoice smoke-qwen3 smoke-e2e test test-unit test-verbose setup-dev test-openvoice test-pyvt dub proof proof-report generate-voices analyze-speakers verify-pitch clean

doctor:
	python3 scripts/doctor.py

# Remove generated reports and proof artifacts. Does NOT delete venvs,
# checkpoints, or job directories (use --force-overwrite-job-dir for those).
clean:
	rm -rf reports
	rm -f /tmp/movie-dub-doctor.out /tmp/movie-dub-proof/checks.json
	rm -rf /tmp/movie-dub-proof

doctor-strict:
	python3 scripts/doctor.py --strict

doctor-demucs:
	python3 scripts/doctor.py --need-demucs

doctor-checkpoints:
	python3 scripts/doctor.py --checkpoints-only

doctor-qwen3:
	python3 scripts/doctor.py --qwen3-only

# Run all four pass gates in sequence. All must pass.
# Note: doctor output is redirected to a file first, then tailed,
# so the doctor exit code is not masked by the pipe to tail.
# A machine-readable proof_report.json is always written (Blocker 7).
proof:
	@rm -rf reports && mkdir -p reports /tmp/movie-dub-proof
	@echo "=== 1/4 doctor ==="
	@python3 scripts/doctor.py > /tmp/movie-dub-doctor.out 2>&1 || { \
		cat /tmp/movie-dub-doctor.out; \
		python3 scripts/write_proof_report.py --result fail \
			--failed-check doctor \
			--message "doctor failed; see /tmp/movie-dub-doctor.out" \
			--output reports/proof_report.json; \
		exit 1; \
	}
	@tail -3 /tmp/movie-dub-doctor.out
	@echo '{"doctor": "pass"}' > /tmp/movie-dub-proof/checks.json
	@echo "=== 2/4 smoke-openvoice ===" && python3 scripts/smoke_openvoice_bridge.py || { \
		python3 scripts/write_proof_report.py --result fail \
			--failed-check openvoice_smoke \
			--message "smoke_openvoice_bridge failed" \
			--checks-json /tmp/movie-dub-proof/checks.json \
			--output reports/proof_report.json; \
		exit 1; \
	}
	@python3 -c "import json; d=json.load(open('/tmp/movie-dub-proof/checks.json')); d['openvoice_smoke']='pass'; json.dump(d, open('/tmp/movie-dub-proof/checks.json','w'))"
	@echo "=== 3/4 smoke-e2e ===" && python3 scripts/smoke_e2e_short_clip.py || { \
		python3 scripts/write_proof_report.py --result fail \
			--failed-check audio_builder_smoke \
			--message "smoke_e2e_short_clip failed" \
			--checks-json /tmp/movie-dub-proof/checks.json \
			--output reports/proof_report.json; \
		exit 1; \
	}
	@python3 -c "import json; d=json.load(open('/tmp/movie-dub-proof/checks.json')); d['audio_builder_smoke']='pass'; d['remux_smoke']='pass'; json.dump(d, open('/tmp/movie-dub-proof/checks.json','w'))"
	@echo "=== 4/4 test-pyvt ===" && cd pyvideotrans-main && .venv/bin/python -m pytest -q || { \
		python3 scripts/write_proof_report.py --result fail \
			--failed-check pyvideotrans_tests \
			--message "pyVideoTrans pytest failed" \
			--checks-json /tmp/movie-dub-proof/checks.json \
			--output reports/proof_report.json; \
		exit 1; \
	}
	@python3 -c "import json; d=json.load(open('/tmp/movie-dub-proof/checks.json')); d['pyvideotrans_tests']='pass'; json.dump(d, open('/tmp/movie-dub-proof/checks.json','w'))"
	@python3 scripts/write_proof_report.py --result pass \
		--checks-json /tmp/movie-dub-proof/checks.json \
		--output reports/proof_report.json
	@echo ""
	@echo "PROOF: PASS"
	@echo "Proof report: reports/proof_report.json"

proof-report:
	@cat reports/proof_report.json

setup-ffmpeg:
	bash scripts/setup_ffmpeg_mac.sh

setup-pyvt:
	bash scripts/setup_pyvideotrans_mac.sh

setup-openvoice:
	bash scripts/setup_openvoice_mac.sh

setup-qwen3:
	bash scripts/setup_qwen3_local.sh

setup-omnivoice:
	bash scripts/setup_omnivoice.sh

setup-checkpoints:
	python3 scripts/setup_openvoice_checkpoints.py

download-openvoice:
	bash scripts/download_openvoice_v2_checkpoints.sh

smoke-openvoice:
	python3 scripts/smoke_openvoice_bridge.py

smoke-qwen3:
	python3 scripts/smoke_qwen3.py

smoke-e2e:
	python3 scripts/smoke_e2e_short_clip.py

# Install dev dependencies (pytest, ruff) into the current interpreter.
setup-dev:
	python3 -m pip install -r requirements-dev.txt

# Run the full pytest unit test suite (tests/ dir). Fast — no model deps.
# Checks that pytest is installed first and gives a helpful error if not.
test: test-unit

test-unit:
	@python3 -c "import pytest" 2>/dev/null || { \
		echo "ERROR: pytest is not installed for $$(python3 --version 2>&1)."; \
		echo "  Fix: make setup-dev"; \
		echo "  Or:  python3 -m pip install pytest"; \
		exit 1; \
	}
	@echo "Running tests with $$(python3 --version 2>&1) from $$(pwd)..."
	python3 -m pytest tests/ -q

test-verbose:
	@python3 -c "import pytest" 2>/dev/null || { \
		echo "ERROR: pytest is not installed for $$(python3 --version 2>&1)."; \
		echo "  Fix: make setup-dev"; \
		exit 1; \
	}
	python3 -m pytest tests/ -v --tb=short

test-openvoice:
	cd pyvideotrans-main && .venv/bin/python -m pytest tests/test_openvoice_bridge.py -q

test-pyvt:
	cd pyvideotrans-main && .venv/bin/python -m pytest

generate-voices:
	@echo "Generating male and female reference voices..."
	@cd OpenVoice-main && .venv/bin/python ../scripts/generate_reference_voices.py

# Speaker profiling: diarization + per-speaker reference extraction.
# Usage:
#   make analyze-speakers INPUT=~/Movies/test.mp4 OUTPUT=tmp/speakers
# Optional: DIARIZATION=pyannote NUM_SPEAKERS=auto HF_TOKEN=$(HF_TOKEN)
analyze-speakers:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make analyze-speakers INPUT=<input.mp4> OUTPUT=<speakers_dir>"; \
		echo "  DIARIZATION=pyannote NUM_SPEAKERS=auto HF_TOKEN=\$$HF_TOKEN"; \
		exit 1; \
	fi
	pyvideotrans-main/.venv/bin/python scripts/analyze_speakers.py \
		--input "$(INPUT)" \
		--output-dir "$(OUTPUT)" \
		--diarization $(or $(DIARIZATION),pyannote) \
		--num-speakers $(or $(NUM_SPEAKERS),auto) \
		$(if $(HF_TOKEN),--hf-token $(HF_TOKEN))

# Post-generation pitch verification.
# Usage:
#   make verify-pitch MANIFEST=tmp/job/openvoice_manifest.json \
#     PROFILES=tmp/job/speakers/speaker_profiles.json \
#     OUTPUT=tmp/job/pitch_verification.json
verify-pitch:
	@if [ -z "$(MANIFEST)" ] || [ -z "$(PROFILES)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make verify-pitch MANIFEST=<manifest.json> PROFILES=<profiles.json> OUTPUT=<out.json>"; \
		exit 1; \
	fi
	pyvideotrans-main/.venv/bin/python scripts/verify_segment_pitch.py \
		--manifest "$(MANIFEST)" \
		--speaker-profiles "$(PROFILES)" \
		--output "$(OUTPUT)"

# Personal dubbing wrapper. Usage:
#   make dub INPUT=~/Movies/test.mp4 OUTPUT=~/Movies/dubbed-test.mp4
# Optional: SOURCE_LANG=auto TARGET_LANG=en GENDER=auto ASR_MODEL=whisper-large-v3-turbo
#           BACKGROUND_VOLUME=0.15 VOICE_VOLUME=1.2 VOCAL_SEPARATION=1
#           VOCAL_SEPARATION_METHOD=demucs DUCKING=1 TARGET_LUFS=-16
#           NO_NORMALIZE=1 FINAL_GAIN=1.0
#           BACKGROUND_TIMEOUT=900 LUFS_TIMEOUT=1200 FAIL_IF_BACKGROUND_MIX_FAILS=1
#           DEMUCS_TIMEOUT=900 ALLOW_DEMUCS_DOWNLOAD=1
#           SPEAKER_PROFILING=1 DIARIZATION=pyannote NUM_SPEAKERS=auto
#           TTS_ENGINE=openvoice FALLBACK_TTS_ENGINE=none
#           VERIFY_PITCH=1 FAIL_ON_PITCH_MISMATCH=1
#           PREFER_EMBEDDED_SUBTITLES=1 (NO_OCR_IMAGE_SUBS=1 to skip OCR)
#           RESUME=1 FROM_STAGE=tts ONLY_SEGMENT=42 SKIP_EXISTING=1 NO_CACHE=1
dub:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make dub INPUT=<input.mp4> OUTPUT=<output.mp4>"; \
		echo "  SOURCE_LANG=auto TARGET_LANG=en GENDER=auto ASR_MODEL=whisper-large-v3-turbo"; \
		echo "  BACKGROUND_VOLUME=0.15 VOCAL_SEPARATION=1 VOCAL_SEPARATION_METHOD=ffmpeg"; \
		echo "  For AI separation: VOCAL_SEPARATION_METHOD=demucs ALLOW_DEMUCS_DOWNLOAD=1"; \
		echo "  DUCKING=1"; \
		echo "  SPEAKER_PROFILING=1 DIARIZATION=pyannote TTS_ENGINE=openvoice"; \
		echo "  For OmniVoice: TTS_ENGINE=omnivoice OMNIVOICE_URL=http://localhost:3900"; \
		echo "  VERIFY_PITCH=1 FAIL_ON_PITCH_MISMATCH=1"; \
		echo "  PREFER_EMBEDDED_SUBTITLES=1 (use embedded subs as text+timing source)"; \
		exit 1; \
	fi
	python3 scripts/run_personal_dub.py \
		--input "$(INPUT)" \
		--source-language $(or $(SOURCE_LANG),auto) \
		--target-language $(or $(TARGET_LANG),en) \
		--model $(or $(ASR_MODEL),openai/whisper-large-v3-turbo) \
		$(if $(GENDER),--gender $(GENDER)) \
		$(if $(REFERENCE),--reference $(REFERENCE)) \
		--output "$(OUTPUT)" \
		$(if $(BACKGROUND_VOLUME),--background-volume $(BACKGROUND_VOLUME)) \
		$(if $(VOICE_VOLUME),--voice-volume $(VOICE_VOLUME)) \
		$(if $(FINAL_GAIN),--final-gain $(FINAL_GAIN)) \
		$(if $(NO_NORMALIZE),--no-normalize) \
		$(if $(VOCAL_SEPARATION),--vocal-separation) \
		$(if $(VOCAL_SEPARATION_METHOD),--vocal-separation-method $(VOCAL_SEPARATION_METHOD)) \
		$(if $(DUCKING),--ducking) \
		$(if $(TARGET_LUFS),--target-lufs $(TARGET_LUFS)) \
		$(if $(BACKGROUND_TIMEOUT),--background-timeout $(BACKGROUND_TIMEOUT)) \
		$(if $(LUFS_TIMEOUT),--lufs-timeout $(LUFS_TIMEOUT)) \
		$(if $(DEMUCS_TIMEOUT),--demucs-timeout $(DEMUCS_TIMEOUT)) \
		$(if $(ALLOW_DEMUCS_DOWNLOAD),--allow-demucs-download) \
		$(if $(CPU_ONLY),--cpu-only) \
		$(if $(FAIL_IF_BACKGROUND_MIX_FAILS),--fail-if-background-mix-fails) \
		$(if $(SPEAKER_PROFILING),--speaker-profiling) \
		$(if $(DIARIZATION),--speaker-diarization $(DIARIZATION)) \
		$(if $(NUM_SPEAKERS),--num-speakers $(NUM_SPEAKERS)) \
		$(if $(SPEAKER_PROFILE_JSON),--speaker-profile-json $(SPEAKER_PROFILE_JSON)) \
		$(if $(HF_TOKEN),--hf-token $(HF_TOKEN)) \
		$(if $(TTS_ENGINE),--tts-engine $(TTS_ENGINE)) \
		$(if $(FALLBACK_TTS_ENGINE),--fallback-tts-engine $(FALLBACK_TTS_ENGINE)) \
		$(if $(BASE_SPEAKER),--base-speaker $(BASE_SPEAKER)) \
		$(if $(VERIFY_PITCH),--verify-pitch) \
		$(if $(FAIL_ON_PITCH_MISMATCH),--fail-on-pitch-mismatch) \
		$(if $(PREFER_EMBEDDED_SUBTITLES),--prefer-embedded-subtitles) \
		$(if $(NO_OCR_IMAGE_SUBS),--no-ocr-image-subs) \
		$(if $(AGE_MODEL),--age-model $(AGE_MODEL)) \
		$(if $(AGE_MODEL_PATH),--age-model-path $(AGE_MODEL_PATH)) \
		$(if $(REFERENCE_CLIP_SCORING),--reference-clip-scoring $(REFERENCE_CLIP_SCORING)) \
		$(if $(CHARACTER_PROFILES),--character-profiles) \
		$(if $(RESUME),--resume) \
		$(if $(FROM_STAGE),--from-stage $(FROM_STAGE)) \
		$(if $(ONLY_SEGMENT),--only-segment $(ONLY_SEGMENT)) \
		$(if $(SKIP_EXISTING),--skip-existing) \
		$(if $(NO_CACHE),--no-cache) \
		$(if $(OMNIVOICE_URL),--omnivoice-url $(OMNIVOICE_URL)) \
		$(if $(TTS_SPEED),--tts-speed $(TTS_SPEED))

# Build character_profiles.json from speaker_profiles.json.
# Usage: make character-profiles PROFILES=tmp/job/speakers/speaker_profiles.json \
#   OUTPUT=tmp/job/character_profiles.json TTS_ENGINE=qwen3-local
character-profiles:
	@if [ -z "$(PROFILES)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make character-profiles PROFILES=<speaker_profiles.json> OUTPUT=<character_profiles.json>"; \
		echo "  TTS_ENGINE=qwen3-local (optional)"; \
		exit 1; \
	fi
	python3 scripts/character_profiles.py \
		--speaker-profiles "$(PROFILES)" \
		--output "$(OUTPUT)" \
		$(if $(TTS_ENGINE),--tts-engine $(TTS_ENGINE))

# Rename / lock / review a character in character_profiles.json.
# Usage: make character-rename PROFILE_FILE=<cp.json> CHAR_ID=CHAR_001 CHAR_NAME="Alice"
#        make character-lock PROFILE_FILE=<cp.json> CHAR_ID=CHAR_001 REF=voices/alice.wav
#        make character-review PROFILE_FILE=<cp.json> CHAR_ID=CHAR_001 STATUS=approved
character-rename:
	@if [ -z "$(PROFILE_FILE)" ] || [ -z "$(CHAR_ID)" ] || [ -z "$(CHAR_NAME)" ]; then \
		echo "Usage: make character-rename PROFILE_FILE=<cp.json> CHAR_ID=CHAR_001 CHAR_NAME=\"Alice\""; exit 1; \
	fi
	python3 scripts/character_profiles.py --rename $(CHAR_ID) "$(CHAR_NAME)" --output "$(PROFILE_FILE)"

character-lock:
	@if [ -z "$(PROFILE_FILE)" ] || [ -z "$(CHAR_ID)" ] || [ -z "$(REF)" ]; then \
		echo "Usage: make character-lock PROFILE_FILE=<cp.json> CHAR_ID=CHAR_001 REF=<wav>"; exit 1; \
	fi
	python3 scripts/character_profiles.py --lock-voice $(CHAR_ID) $(REF) --output "$(PROFILE_FILE)"

character-review:
	@if [ -z "$(PROFILE_FILE)" ] || [ -z "$(CHAR_ID)" ] || [ -z "$(STATUS)" ]; then \
		echo "Usage: make character-review PROFILE_FILE=<cp.json> CHAR_ID=CHAR_001 STATUS=approved"; exit 1; \
	fi
	python3 scripts/character_profiles.py --set-review $(CHAR_ID) $(STATUS) --output "$(PROFILE_FILE)"

# Regenerate one segment from a review job.
# Usage: make regenerate JOB=tmp/personal_dub/<id> SEGMENT=12 REMUX=1 \
#   TTS_ENGINE=qwen3-local [CHANGE_SPEAKER=SPEAKER_01] [CHANGE_REF=voices/x.wav] [SHORTEN=1]
regenerate:
	@if [ -z "$(JOB)" ] || [ -z "$(SEGMENT)" ]; then \
		echo "Usage: make regenerate JOB=<job_dir> SEGMENT=<id> [REMUX=1]"; \
		echo "  TTS_ENGINE=openvoice CHANGE_SPEAKER=SPEAKER_01 CHANGE_REF=<wav> SHORTEN=1"; \
		exit 1; \
	fi
	python3 scripts/regenerate_segment.py \
		--job "$(JOB)" --segment-id $(SEGMENT) \
		$(if $(REMUX),--remux) \
		$(if $(TTS_ENGINE),--tts-engine $(TTS_ENGINE)) \
		$(if $(CHANGE_SPEAKER),--change-speaker $(CHANGE_SPEAKER)) \
		$(if $(CHANGE_REF),--change-reference $(CHANGE_REF)) \
		$(if $(SHORTEN),--shorten) \
		$(if $(QWEN3_MODEL),--qwen3-model $(QWEN3_MODEL))

# Write failed_segments.json from a job's review/manifest/pitch artifacts.
# Usage: make failed-segments JOB=tmp/personal_dub/<id>
failed-segments:
	@if [ -z "$(JOB)" ]; then echo "Usage: make failed-segments JOB=<job_dir>"; exit 1; fi
	python3 scripts/review_loop.py failed \
		--review "$(JOB)/review_segments.json" \
		--manifest "$(JOB)/openvoice_manifest.json" \
		--pitch-verification "$(JOB)/pitch_verification.json" \
		--output "$(JOB)/failed_segments.json"

# Re-assign a segment (or all segments of a speaker) to a new speaker.
# Usage: make change-speaker JOB=<job> TO_SPEAKER=SPEAKER_01 [SEGMENT=12] [FROM_SPEAKER=SPEAKER_00]
change-speaker:
	@if [ -z "$(JOB)" ] || [ -z "$(TO_SPEAKER)" ]; then \
		echo "Usage: make change-speaker JOB=<job> TO_SPEAKER=SPEAKER_01 [SEGMENT=12] [FROM_SPEAKER=SPEAKER_00]"; exit 1; \
	fi
	python3 scripts/review_loop.py change-speaker \
		--review "$(JOB)/review_segments.json" \
		--speaker-profiles "$(JOB)/speakers/speaker_profiles.json" \
		--to-speaker $(TO_SPEAKER) \
		$(if $(SEGMENT),--segment-id $(SEGMENT)) \
		$(if $(FROM_SPEAKER),--from-speaker $(FROM_SPEAKER))

# Repeatable benchmark harness. Runs the full pipeline on a clip and writes
# a metrics report. Supply your own 5-10 min clip.
# Usage: make benchmark INPUT=~/Movies/bench.mp4 OUTPUT_DIR=reports/benchmarks/run1 \
#   TTS_ENGINE=qwen3-local SPEAKER_PROFILING=1 HF_TOKEN=$$HF_TOKEN
benchmark:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT_DIR)" ]; then \
		echo "Usage: make benchmark INPUT=<clip> OUTPUT_DIR=<dir>"; \
		echo "  TTS_ENGINE=qwen3-local SPEAKER_PROFILING=1 HF_TOKEN=\$$HF_TOKEN"; \
		echo "  TARGET_LUFS=-16 VERIFY_PITCH=1 PREFER_EMBEDDED_SUBTITLES=1"; \
		echo "  AGE_MODEL=auto CHARACTER_PROFILES=1"; \
		exit 1; \
	fi
	python3 scripts/benchmark.py \
		--input "$(INPUT)" \
		--output-dir "$(OUTPUT_DIR)" \
		--name $(or $(RUN_NAME),bench) \
		--target-language $(or $(TARGET_LANG),en) \
		--source-language $(or $(SOURCE_LANG),auto) \
		--tts-engine $(or $(TTS_ENGINE),openvoice) \
		$(if $(SPEAKER_PROFILING),--speaker-profiling) \
		$(if $(DIARIZATION),--diarization $(DIARIZATION)) \
		$(if $(NUM_SPEAKERS),--num-speakers $(NUM_SPEAKERS)) \
		$(if $(HF_TOKEN),--hf-token $(HF_TOKEN)) \
		$(if $(VERIFY_PITCH),--verify-pitch) \
		$(if $(PREFER_EMBEDDED_SUBTITLES),--prefer-embedded-subtitles) \
		$(if $(TARGET_LUFS),--target-lufs $(TARGET_LUFS)) \
		$(if $(BACKGROUND_VOLUME),--background-volume $(BACKGROUND_VOLUME)) \
		$(if $(DUCKING),--ducking) \
		$(if $(QWEN3_MODEL),--qwen3-model $(QWEN3_MODEL)) \
		$(if $(AGE_MODEL),--age-model $(AGE_MODEL)) \
		$(if $(AGE_MODEL_PATH),--age-model-path $(AGE_MODEL_PATH)) \
		$(if $(RESUME),--resume) \
		$(if $(SKIP_EXISTING),--skip-existing)

# Inspect a job's segment cache index.
# Usage: make cache-stats CACHE_DIR=tmp/personal_dub/<id>/cache
cache-stats:
	@if [ -z "$(CACHE_DIR)" ]; then echo "Usage: make cache-stats CACHE_DIR=<job>/cache"; exit 1; fi
	python3 scripts/cache.py --cache-dir "$(CACHE_DIR)" --summary

# Age model plugin status (checks .venv-age + local model dir).
age-model-status:
	@python3 scripts/age_model.py; true

# Install the optional age-regression plugin into a dedicated .venv-age and
# download the model into models/ (gitignored). Uses hf download, not git
# clone, so no model weights enter the repo.
# Usage: make setup-age [PYTHON_BIN=python3.10]
setup-age:
	@bash scripts/setup_age_model.sh

# Verify the age model loads from .venv-age + models/.
doctor-age:
	@if [ ! -d .venv-age ]; then \
		echo "FAIL: .venv-age missing. Run: make setup-age"; exit 1; \
	fi
	@. .venv-age/bin/activate && python -c "\
from pathlib import Path;\
from voice_age_regressor import AgeRegressionPipeline;\
model_path = Path('models/age_reg_ann_ecapa_librosa_combined');\
assert model_path.exists(), 'FAIL: model dir missing: ' + str(model_path);\
AgeRegressionPipeline.from_pretrained(str(model_path));\
print('Age model OK: ' + str(model_path))"

# Smoke-test the age model on a reference WAV. Writes tmp/smoke_age.json.
# Usage: make smoke-age [AUDIO=voices/male_reference.wav] [MODEL_PATH=models/...]
smoke-age:
	@if [ ! -d .venv-age ]; then \
		echo "FAIL: .venv-age missing. Run: make setup-age"; exit 1; \
	fi
	@mkdir -p tmp
	@. .venv-age/bin/activate && python scripts/estimate_speaker_age.py \
		--audio $(or $(AUDIO),voices/male_reference.wav) \
		--model-path $(or $(MODEL_PATH),models/age_reg_ann_ecapa_librosa_combined) \
		--output tmp/smoke_age.json
