.PHONY: doctor setup-ffmpeg setup-pyvt setup-openvoice smoke-openvoice smoke-e2e test-openvoice test-pyvt

doctor:
	python3 scripts/doctor.py

setup-ffmpeg:
	bash scripts/setup_ffmpeg_mac.sh

setup-pyvt:
	bash scripts/setup_pyvideotrans_mac.sh

setup-openvoice:
	bash scripts/setup_openvoice_mac.sh

smoke-openvoice:
	python3 scripts/smoke_openvoice_bridge.py

smoke-e2e:
	python3 scripts/smoke_e2e_short_clip.py

test-openvoice:
	cd pyvideotrans-main && .venv/bin/python -m pytest tests/test_openvoice_bridge.py -q

test-pyvt:
	cd pyvideotrans-main && .venv/bin/python -m pytest
