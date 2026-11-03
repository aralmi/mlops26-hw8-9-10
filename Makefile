PYTHON ?= .venv/bin/python
PORT ?= 8091
API_URL ?= http://127.0.0.1:$(PORT)
CANDIDATE_DIR ?= .runtime/artifacts/candidate
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
export PREFECT_HOME := $(CURDIR)/.prefect
export PREFECT_LOCAL_STORAGE_PATH := $(CURDIR)/.prefect/results
export PREFECT_SERVER_ANALYTICS_ENABLED := false
.DEFAULT_GOAL := help

.PHONY: help lock install install-model check check-core check-http check-target check-previous prove break-demo process-demo versions-demo prepare-real fetch-candidate prepare-candidate release serve smoke replay benchmark status health predict invalid gate real-process restart export

help:
	@echo "Checks: make check; make prove; make status"
	@echo "Real API: make serve (one terminal), make health / predict / invalid (another)"
	@echo "Candidate: make gate or make release; the current candidate is rejected."

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
	cd previous/hw67 && $(abspath $(PYTHON)) -m unittest discover -s tests -p 'test_*.py' -v -b

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
	$(PYTHON) materialize.py --candidate-root $(CANDIDATE_DIR)

release:
	$(PYTHON) flow.py

serve:
	$(PYTHON) api.py --workspace .runtime --backend transformers --port $(PORT)

smoke:
	$(PYTHON) scripts/real_smoke.py

replay:
	$(PYTHON) scripts/replay_baseline.py --url $(API_URL)

benchmark:
	$(PYTHON) scripts/benchmark.py --n 8 --url $(API_URL)

status:
	$(PYTHON) scripts/verify_evidence.py

health:
	$(PYTHON) scripts/request.py --url $(API_URL) --health

predict:
	$(PYTHON) scripts/request.py --url $(API_URL)

invalid:
	$(PYTHON) scripts/request.py --url $(API_URL) --invalid

gate:
	$(PYTHON) scripts/gate.py

real-process:
	$(PYTHON) scripts/real_process.py

restart:
	$(PYTHON) scripts/restart_real.py

export:
	$(PYTHON) scripts/export_results.py

.PHONY: fetch-base package-candidate
fetch-base:
	$(PYTHON) scripts/fetch_base.py

package-candidate:
	$(PYTHON) scripts/package_candidate.py
