PYTHON ?= .venv/bin/python
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
export PREFECT_HOME := $(CURDIR)/.prefect
export PREFECT_LOCAL_STORAGE_PATH := $(CURDIR)/.prefect/results
export PREFECT_SERVER_ANALYTICS_ENABLED := false
.DEFAULT_GOAL := help

.PHONY: help lock install install-model check check-core check-http check-target check-previous prove break-demo process-demo versions-demo prepare-real fetch-candidate prepare-candidate release serve smoke replay benchmark

help:
	@echo "Preparation only. Nothing starts automatically. See README.md."
	@echo "Future checks: make install; make check; make prove"
	@echo "Real model (separate): make install-model; make prepare-real; make serve"

lock:
	uv lock

install:
	uv sync --frozen

install-model:
	uv sync --frozen --extra model

check: check-core check-http check-target check-previous

check-core:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) -m unittest discover -s verification -v -b

check-http:
	HW8910_PROJECT=$(CURDIR) HW8910_RUNNER=lab.py HW8910_SERVICE=service.py $(PYTHON) scripts/run_integrations.py --json evidence/checks/http.json

check-target:
	HW8910_PROJECT=$(CURDIR) HW8910_RUNNER=flow_lab.py HW8910_SERVICE=api.py $(PYTHON) scripts/run_integrations.py --json evidence/checks/prefect-fastapi.json

check-previous:
	cd previous/hw67 && $(abspath $(PYTHON)) -m unittest discover -s tests -p 'test_*.py' -v

prove:
	$(PYTHON) scripts/prove_regressions.py

break-demo:
	$(PYTHON) scripts/break_demo.py

process-demo:
	$(PYTHON) scripts/process_demo.py

versions-demo:
	$(PYTHON) scripts/versions_demo.py

prepare-real:
	$(PYTHON) materialize.py

fetch-candidate:
	$(PYTHON) scripts/fetch_candidate.py

prepare-candidate:
	$(PYTHON) materialize.py --candidate-root .runtime/artifacts/candidate

release:
	$(PYTHON) flow.py

serve:
	$(PYTHON) api.py --workspace .runtime --backend transformers --port 8091

smoke:
	$(PYTHON) scripts/real_smoke.py

replay:
	$(PYTHON) scripts/replay_baseline.py

benchmark:
	$(PYTHON) scripts/benchmark.py --n 8
