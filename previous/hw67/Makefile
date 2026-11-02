export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export DVC_SITE_CACHE_DIR := $(CURDIR)/.cache/dvc/site
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export TOKENIZERS_PARALLELISM := false
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING := false
export MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR := false

.PHONY: install up down sweep promote predict check check-storage
install:
	uv sync --frozen
up:
	docker compose -p mlops-hw6 up --build -d --wait
down:
	docker compose -p mlops-hw6 down
sweep:
	uv run python -m src.sweep
promote:
	uv run python -m evaluation.cli promote
predict:
	uv run python -m src.predict --uri models:/pandas-qwen-adapter@champion --prompt "In pandas, how do I count missing values?"
check:
	bash tests/check.sh
check-storage:
	uv run python tests/check_storage.py

.PHONY: report candidate baseline-init evaluate eval check-eval check-negative check-all baseline-update predict-candidate
report:
	uv run python -m src.report
candidate:
	uv run python -m evaluation.cli candidate
baseline-init:
	uv run python -m evaluation.cli baseline-init
evaluate:
	uv run python -m evaluation.cli evaluate
eval:
	uv run python -m evaluation.cli gate
check-eval:
	uv run python -m unittest discover -s tests -p 'test_*.py' -v -b
check-negative:
	uv run python -m evaluation.cli negative
check-all: check check-eval
baseline-update:
	uv run python -m evaluation.cli baseline-update
predict-candidate:
	uv run python -m src.predict --uri models:/pandas-qwen-adapter@candidate --prompt "In pandas, how do I count missing values?"

.PHONY: prepare-data verify status submission
prepare-data:
	uv run python -m preparation.prepare_data
verify:
	uv run python -m unittest discover -s verification -p 'test_*.py' -v -b
status:
	uv run dvc status
submission:
	uv run python -m preparation.export_submission
