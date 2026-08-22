# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-22

Second milestone: configurable architecture options, richer training diagnostics, a much more capable generation CLI, and analysis tooling.

### Added

- Architecture switches on `GPTConfig`: `pos_encoding` (`"learned"` / `"rope"`) for rotary position embeddings with non-persistent cos/sin tables sliced at the cache offset during decoding; `norm_type` (`"layernorm"` / `"rmsnorm"`) backed by a new `RMSNorm` module; `mlp_type` (`"gelu"` / `"swiglu"`) with a bias-free LLaMA-style gated MLP whose hidden size is `8/3 * n_embd` rounded up to a multiple of 8; and `tie_embeddings` (default true).
- Min-p sampling: `generate(min_p=...)` and the sampling CLI's `--min-p` drop tokens whose probability falls below a fraction of the top token's probability.
- Training extras: `--ema` / `--ema-decay` (0.999) maintain an exponential moving average of weights that is used for every validation pass and stored in checkpoints as `ema_state`; `--save-top-k` rotates the k best checkpoints by validation loss as `checkpoint-top1.pt`, ...; `--deterministic` seeds Python/NumPy/torch/CUDA RNGs and forces deterministic cuDNN/cuBLAS kernels (including `CUBLAS_WORKSPACE_CONFIG`); `--overfit-batches N` trains on a fixed pool of pre-sampled batches as a memorization diagnostic.
- Generation CLI upgrades: `--stream` prints each token as it is generated, holding back a tail so stop strings are never partially emitted; repeatable `--stop STR` halts generation at the earliest match; `--prompts-file PATH` generates for one prompt per line (blank lines and `#` comments skipped); `--interactive` starts a REPL supporting `/temp X`, `/top-k X`, `/top-p X`, `/min-p X` live-parameter changes plus `/quit` and `/exit`.
- Analysis tools: `gpt-from-scratch-plot` renders train/val loss curves from a `train.py --log-file` JSONL log (requires the optional `plot` dependency group: `uv sync --extra plot`), and `gpt-from-scratch-sweep` grid-searches temperature x top-k x top-p over a checkpoint, scoring continuations by teacher-forced NLL and distinct-1/distinct-2 diversity in a table sorted by NLL.
- Tooling: `Makefile` with sync/test/coverage/lint/format/train-small/sample/bench/plot/sweep/clean targets, a CPU-only-torch `Dockerfile` with the training CLI as entrypoint, and Dependabot weekly updates for pip dependencies and GitHub Actions.
- Documentation: MATH.md with plain-math derivations (attention scaling and causal masking, cross-entropy gradient, perplexity/bpc, AdamW decoupled decay, warmup+cosine schedule, RoPE rotation, RMSNorm vs LayerNorm, sampling filters) and README additions — grouped features, new CLI rows, GPTConfig options table, FAQ, and an honest nanoGPT/minGPT comparison table.

### Changed

- Checkpoint payloads may now carry additional optional fields: `ema_state` when trained with `--ema`, and `rank` / `val_loss` / `iteration` metadata on top-k checkpoints; older checkpoints remain loadable unchanged.

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

[0.2.0]: https://github.com/VictorMerk/gpt-from-scratch/releases/tag/v0.2.0
[0.1.0]: https://github.com/VictorMerk/gpt-from-scratch/releases/tag/v0.1.0
