.PHONY: doctor setup-ffmpeg setup-pyvt setup-openvoice setup-checkpoints download-openvoice smoke-openvoice smoke-e2e test-openvoice test-pyvt dub proof

doctor:
	python3 scripts/doctor.py

# Run all four pass gates in sequence. All must pass.
proof:
	@echo "=== 1/4 doctor ===" && python3 scripts/doctor.py | tail -3
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

# Personal dubbing wrapper. Usage:
#   make dub INPUT=~/Movies/test.mp4 OUTPUT=~/Movies/dubbed-test.mp4
# Optional: SOURCE=en TARGET=en REFERENCE=voices/openvoice_default_reference.wav
#           BACKGROUND_VOLUME=0.15 VOICE_VOLUME=1.2 VOCAL_SEPARATION=1
#           DUCKING=1 TARGET_LUFS=-16 NO_NORMALIZE=1 FINAL_GAIN=1.0
dub:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make dub INPUT=<input.mp4> OUTPUT=<output.mp4>"; \
		exit 1; \
	fi
	python3 scripts/run_personal_dub.py \
		--input "$(INPUT)" \
		--source-language $(or $(SOURCE),en) \
		--target-language $(or $(TARGET),en) \
		--reference $(or $(REFERENCE),voices/openvoice_default_reference.wav) \
		--output "$(OUTPUT)" \
		$(if $(BACKGROUND_VOLUME),--background-volume $(BACKGROUND_VOLUME)) \
		$(if $(VOICE_VOLUME),--voice-volume $(VOICE_VOLUME)) \
		$(if $(FINAL_GAIN),--final-gain $(FINAL_GAIN)) \
		$(if $(NO_NORMALIZE),--no-normalize) \
		$(if $(VOCAL_SEPARATION),--vocal-separation) \
		$(if $(DUCKING),--ducking) \
		$(if $(TARGET_LUFS),--target-lufs $(TARGET_LUFS))
