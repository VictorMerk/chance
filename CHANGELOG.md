# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-22

Third milestone: GPT-2-grade tokenization, binary data caching with dataset tooling, reproducible experiment drivers, checkpoint export to standard formats, an OpenAI-compatible serving endpoint, and expanded CI automation.

### Added

- BPE pre-tokenization: GPT-2-style chunking (contractions, letter runs, digit runs, other-symbol runs; a single leading space attaches to the following chunk) hand-rolled from unicode property checks so merges never cross chunk boundaries during training or encoding.
- Special tokens on `BPETokenizer(special_tokens=...)`: ids reserved immediately after all merges, `encode_with_special` maps literal occurrences greedy-longest-first (plain `encode` leaves them as text), `decode` restores their literal forms, vocab size accounting reserves room for them, and they persist through `save`/`load`.
- Dataset tooling (`dataset.py`): `save_tokens_bin` writes raw little-endian uint16/uint32 token files, `load_tokens_bin` memory-maps them read-only and widens zero-copy via `torch.frombuffer` into a `torch.long` tensor, and `dataset_stats` summarizes splits (token counts, id ranges, top-10 tokens with decoded previews when `<prefix>train.bin.vocab.json` exists) — exposed as `python -m gpt_from_scratch.dataset --bin-prefix data/`.
- `data.load_corpus` reads any UTF-8 `.txt`/`.md` corpus file with a clear error for unsupported suffixes.
- Training on pre-tokenized data: `gpt-from-scratch-train --data-format bin` loads `{data-dir}/train.bin`, `val.bin`, and the sibling `train.bin.vocab.json` instead of downloading tiny Shakespeare (`text` remains the default).
- Architecture flag `pre_norm` on `GPTConfig` (default `True`): `False` switches every block to original Transformer/GPT-1 post-norm for ablation studies.
- Experiments package: a shared seeded short-run runner (`experiments/_runner.py`) reusing the trainer's data loading, batching, and evaluation helpers, plus two standalone CLIs — `python -m gpt_from_scratch.experiments.scaling` (val loss vs preset size across `GPT_PRESETS`, optional log-x plot) and `python -m gpt_from_scratch.experiments.lr_ablation` (constant vs cosine vs warmup+cosine twin runs differing only in schedule, schedule-overlay + loss-bar plot). Both print tables, write JSON results (`--out`), and need the plot extra only for `--plot`; see EXPERIMENTS.md.
- Checkpoint export (`export.py`, run as `python -m gpt_from_scratch.export --format hf|onnx`): Hugging Face `GPT2LMHeadModel` directories (`config.json`, GPT-2-named `pytorch_model.bin` with attention/MLP weights transposed to Conv1D layout, token-to-id `vocab.json`) and ONNX graphs (opset 17, dynamic batch/sequence axes, `input_ids -> logits`). Configs outside the GPT-2-representable subset (learned positions, LayerNorm, GELU MLP, pre-norm, tied embeddings) raise `NotImplementedError`.
- Serving (`serve.py`, requires the new `serve` extra): FastAPI app exposing OpenAI-compatible `POST /v1/completions` accepting `prompt`, `max_tokens`, `temperature`, `top_k`, `top_p`, and `stop`, returning `text_completion` objects with `finish_reason` (`stop`/`length`) and prompt/completion usage counts; run with `python -m gpt_from_scratch.serve --checkpoint ...`.
- Model card template at `docs/model_card.md` with placeholder fields, HF-loading snippets, and a filled example row-set for the default shakespeare-char configuration.
- CI/workflows: a mypy type-check job in CI (configuration under `[tool.mypy]`), `release.yml` running tests and uploading built sdist/wheel artifacts on `v*` tags, and `smoke.yml`, a manually triggered tiny end-to-end train + sample run that uploads its checkpoint.

### Changed

- Optional dependency groups expanded alongside `plot`: `serve` (fastapi + uvicorn) and `onnx` (onnxruntime); install together, e.g. `uv sync --extra plot --extra serve --extra onnx`.
- `BPETokenizer.save` payloads now include `special_tokens`; loading older files without the key continues to work.

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

[0.3.0]: https://github.com/VictorMerk/gpt-from-scratch/releases/tag/v0.3.0
[0.2.0]: https://github.com/VictorMerk/gpt-from-scratch/releases/tag/v0.2.0
[0.1.0]: https://github.com/VictorMerk/gpt-from-scratch/releases/tag/v0.1.0
