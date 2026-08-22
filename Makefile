# Developer workflow targets. Override variables on the command line, e.g.
# `make sweep CHECKPOINT=checkpoints/best.pt` or `make plot LOG_FILE=run.jsonl`.

LOG_FILE ?= checkpoints/log.jsonl
CHECKPOINT ?= checkpoints/checkpoint.pt

.DEFAULT_GOAL := help
.PHONY: help sync test coverage lint format train-small sample bench plot sweep clean

help: ## Show available targets
	@awk 'BEGIN {FS = ": ## "} /^[a-z][a-z-]*:.*## /{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Sync dev environment and dependencies (uv)
	uv sync --dev

test: ## Run the test suite
	uv run pytest -q

coverage: ## Run tests with line-coverage report
	uv run pytest -q --cov=gpt_from_scratch --cov-report=term-missing

lint: ## Ruff lint plus format check
	uv run ruff check . && uv run ruff format --check .

format: ## Auto-format code and apply safe ruff fixes
	uv run ruff format . && uv run ruff check --fix .

train-small: ## Tiny CPU training run (end-to-end smoke test)
	uv run gpt-from-scratch-train --device cpu --max-iters 20 --batch-size 8 \
		--block-size 64 --n-layer 2 --n-head 2 --n-embd 64 \
		--eval-interval 10 --eval-iters 5 --log-file $(LOG_FILE)

sample: ## Generate text from a trained checkpoint
	uv run gpt-from-scratch-sample --checkpoint $(CHECKPOINT)

bench: ## Benchmark generation throughput
	uv run gpt-from-scratch-benchmark

plot: ## Plot loss curves from a training log
	uv run gpt-from-scratch-plot --log-file $(LOG_FILE)

sweep: ## Sweep temperature/top-k/top-p over a checkpoint
	uv run gpt-from-scratch-sweep --checkpoint $(CHECKPOINT)

clean: ## Remove caches, data and checkpoint artifacts (gitignored files)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist wheels data checkpoints
	find . -path ./.venv -prune -o -type d \( -name __pycache__ -o -name '*.egg-info' \
		-o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f \( -name '*.py[oc]' -o -name '*.pt' \) \
		-exec rm -f {} +
