.PHONY: doctor setup-ffmpeg setup-pyvt setup-openvoice setup-checkpoints download-openvoice smoke-openvoice smoke-e2e test-openvoice test-pyvt dub proof generate-voices

doctor:
	python3 scripts/doctor.py

# Run all four pass gates in sequence. All must pass.
# Note: doctor output is redirected to a file first, then tailed,
# so the doctor exit code is not masked by the pipe to tail.
proof:
	@echo "=== 1/4 doctor ==="
	@python3 scripts/doctor.py > /tmp/movie-dub-doctor.out || { cat /tmp/movie-dub-doctor.out; exit 1; }
	@tail -3 /tmp/movie-dub-doctor.out
	@echo "=== 2/4 smoke-openvoice ===" && python3 scripts/smoke_openvoice_bridge.py
	@echo "=== 3/4 smoke-e2e ===" && python3 scripts/smoke_e2e_short_clip.py
	@echo "=== 4/4 test-pyvt ===" && cd pyvideotrans-main && .venv/bin/python -m pytest -q
	@echo ""
	@echo "PROOF: PASS"

setup-ffmpeg:
	bash scripts/setup_ffmpeg_mac.sh

setup-pyvt:
	bash scripts/setup_pyvideotrans_mac.sh

setup-openvoice:
	bash scripts/setup_openvoice_mac.sh

setup-checkpoints:
	python3 scripts/setup_openvoice_checkpoints.py

download-openvoice:
	bash scripts/download_openvoice_v2_checkpoints.sh

smoke-openvoice:
	python3 scripts/smoke_openvoice_bridge.py

smoke-e2e:
	python3 scripts/smoke_e2e_short_clip.py

test-openvoice:
	cd pyvideotrans-main && .venv/bin/python -m pytest tests/test_openvoice_bridge.py -q

test-pyvt:
	cd pyvideotrans-main && .venv/bin/python -m pytest

generate-voices:
	@echo "Generating male and female reference voices..."
	@cd OpenVoice-main && .venv/bin/python ../scripts/generate_reference_voices.py

# Personal dubbing wrapper. Usage:
#   make dub INPUT=~/Movies/test.mp4 OUTPUT=~/Movies/dubbed-test.mp4
# Optional: SOURCE=auto TARGET=en GENDER=auto MODEL=whisper-large-v3-turbo
#           BACKGROUND_VOLUME=0.15 VOICE_VOLUME=1.2 VOCAL_SEPARATION=1
#           VOCAL_SEPARATION_METHOD=demucs DUCKING=1 TARGET_LUFS=-16
#           NO_NORMALIZE=1 FINAL_GAIN=1.0
#           BACKGROUND_TIMEOUT=900 LUFS_TIMEOUT=1200 FAIL_IF_BACKGROUND_MIX_FAILS=1
#           DEMUCS_TIMEOUT=900
dub:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make dub INPUT=<input.mp4> OUTPUT=<output.mp4>"; \
		echo "  SOURCE=auto TARGET=en GENDER=auto MODEL=whisper-large-v3-turbo"; \
		echo "  BACKGROUND_VOLUME=0.15 VOCAL_SEPARATION=1 VOCAL_SEPARATION_METHOD=demucs"; \
		echo "  DUCKING=1"; \
		exit 1; \
	fi
	python3 scripts/run_personal_dub.py \
		--input "$(INPUT)" \
		--source-language $(or $(SOURCE),auto) \
		--target-language $(or $(TARGET),en) \
		--model $(or $(MODEL),openai/whisper-large-v3-turbo) \
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
		$(if $(CPU_ONLY),--cpu-only) \
		$(if $(FAIL_IF_BACKGROUND_MIX_FAILS),--fail-if-background-mix-fails)
