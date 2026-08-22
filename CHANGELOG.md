# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-22

First tagged release: a small from-scratch GPT in PyTorch with training, sampling, evaluation, and benchmarking CLIs.

### Added

- Byte-level BPE tokenizer (`BPETokenizer`) trained from scratch with `train`, `encode`, `decode`, `save`, and JSON `load`.
- Model size presets (`nano` / `micro` / `small` / `medium`) via `GPT_PRESETS`, plus fully configurable `GPTConfig`.
- Warmup + cosine decay learning-rate schedule (`schedule.get_lr`) with configurable warmup iterations and minimum-LR ratio.
- Gradient accumulation for effective large batches, gradient clipping by global norm, and mixed precision (bf16 autocast; fp16 with GradScaler).
- Resumable training: checkpoints store model state, optimizer state, config, vocab, iteration, and val loss.
- Best-checkpoint tracking (`checkpoint-best.pt`) on validation-loss improvement.
- JSONL loss logging via `--log-file` with iteration, train/val loss, learning rate, tokens/sec, and printed ETA.
- Evaluation CLI (`gpt-from-scratch-evaluate`) reporting held-out perplexity and bits-per-character.
- Throughput benchmark CLI (`gpt-from-scratch-benchmark`) comparing cached vs uncached generation speed.
- Top-p (nucleus) sampling alongside temperature and top-k.
- Google Colab notebook for one-click free-GPU training (`notebooks/gpt_from_scratch_colab.ipynb`).
- Project documentation suite: README overhaul, ROADMAP.md, ARCHITECTURE.md, CONTRIBUTING.md, CHANGELOG.md, MIT LICENSE.
- CI running lint and tests on Python 3.12 and 3.13 with pytest-cov coverage reporting.
- pre-commit hooks running ruff check (autofix) and ruff format.

### Changed

- Causal self-attention now uses `F.scaled_dot_product_attention` (Flash Attention / memory-efficient path) instead of manual matmul, masking, and softmax.
- Generation runs on incremental KV-cache decoding by default, with an explicit causal mask when queries are shorter than the cached context.
- Weight initialization follows GPT-2: normal(0, 0.02) everywhere with residual-output projections scaled down by `1/sqrt(2 * n_layer)` for training stability at depth.

[0.1.0]: https://github.com/VictorMerk/gpt-from-scratch/releases/tag/v0.1.0
