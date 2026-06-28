.PHONY: doctor setup-ffmpeg setup-pyvt setup-openvoice setup-checkpoints download-openvoice smoke-openvoice smoke-e2e test-openvoice test-pyvt

doctor:
	python3 scripts/doctor.py

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
