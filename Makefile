.PHONY: doctor doctor-strict doctor-demucs doctor-checkpoints setup-ffmpeg setup-pyvt setup-openvoice setup-checkpoints download-openvoice smoke-openvoice smoke-e2e test-openvoice test-pyvt dub proof proof-report generate-voices clean

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
#           DEMUCS_TIMEOUT=900 ALLOW_DEMUCS_DOWNLOAD=1
dub:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make dub INPUT=<input.mp4> OUTPUT=<output.mp4>"; \
		echo "  SOURCE=auto TARGET=en GENDER=auto MODEL=whisper-large-v3-turbo"; \
		echo "  BACKGROUND_VOLUME=0.15 VOCAL_SEPARATION=1 VOCAL_SEPARATION_METHOD=ffmpeg"; \
		echo "  For AI separation: VOCAL_SEPARATION_METHOD=demucs ALLOW_DEMUCS_DOWNLOAD=1"; \
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
		$(if $(ALLOW_DEMUCS_DOWNLOAD),--allow-demucs-download) \
		$(if $(CPU_ONLY),--cpu-only) \
		$(if $(FAIL_IF_BACKGROUND_MIX_FAILS),--fail-if-background-mix-fails)
