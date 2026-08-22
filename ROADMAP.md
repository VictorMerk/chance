# Roadmap

One hundred planned improvements for gpt-from-scratch, grouped by area and roughly ordered by impact; checked items shipped in milestone 1.

## Tokenization & Data

- [x] BPE tokenizer trained from scratch with train/encode/decode/save/load *(milestone 1)*
- [ ] GPT-2 style regex pre-tokenization before BPE merges
- [ ] Special-token support (<|endoftext|>) with reserved IDs and controlled generation
- [ ] Property-based fuzz tests for tokenizer roundtrips over random unicode
- [ ] Load and use a pretrained GPT-2 vocab/merges file
- [ ] Tokenizer throughput benchmark (bytes/sec, compression ratio) across corpora
- [ ] Generic corpus loader for any .txt/.md dataset with train/val/test split
- [ ] Binary .bin token cache with uint8/uint16 dtype and np.memmap for larger-than-RAM data
- [ ] Dataset statistics report (char/token counts, vocab histogram, length distribution)
- [ ] Optional Hugging Face datasets integration for an OpenWebText-scale demo
- [ ] Hash-based near-duplicate filtering for the training corpus
- [ ] Byte-level vs char-level vs BPE comparison study (val loss on Shakespeare)

## Architecture

- [x] Replace manual attention with F.scaled_dot_product_attention (Flash Attention path) *(milestone 1)*
- [x] KV-cache incremental decoding *(milestone 1)*
- [x] Top-p (nucleus) sampling *(milestone 1)*
- [ ] Min-p sampling
- [ ] Beam search decoding with length penalty
- [x] GPT-2 style residual-scaled initialization *(milestone 1)*
- [x] Model size presets (nano/micro/small/medium) *(milestone 1)*
- [ ] Rotary position embeddings (RoPE) as alternative to learned positions
- [ ] ALiBi position-bias option
- [ ] RMSNorm option behind a config flag
- [ ] SwiGLU MLP variant
- [ ] Grouped-Query / Multi-Query Attention for cheaper inference
- [ ] Mixture-of-Experts MLP with top-k routing (experimental)
- [ ] Weight-tying toggle and ablation support
- [ ] Stochastic depth (layer drop) for regularizing deep configs
- [ ] torch.compile integration with a config flag and fallback
- [ ] Per-layer/per-head attention entropy logging
- [ ] Attention pattern visualization (heatmaps) saved to disk

## Training

- [x] Warmup + cosine decay LR schedule *(milestone 1)*
- [x] Gradient accumulation for effective large batches *(milestone 1)*
- [x] Mixed precision (bf16 autocast; fp16 with GradScaler) *(milestone 1)*
- [x] Gradient clipping by global norm *(milestone 1)*
- [x] Resumable training (model + optimizer + iteration state) *(milestone 1)*
- [ ] EMA of weights with EMA-based evaluation
- [ ] torch.compile the training step for throughput
- [ ] DDP multi-GPU training via torchrun
- [ ] FSDP sharding for larger models
- [x] JSONL/CSV loss logging with tokens/sec and ETA *(milestone 1)*
- [ ] TensorBoard / W&B optional logging backends
- [x] Best-checkpoint tracking and early stopping *(milestone 1)*
- [ ] Checkpoint rotation (keep top-k by val loss)
- [ ] Fully deterministic mode (seed everything, --deterministic flag)
- [ ] Overfit-one-batch diagnostic mode and block-size curriculum

## Evaluation

- [x] Perplexity on held-out text *(milestone 1)*
- [x] Bits-per-character (bpc) metric *(milestone 1)*
- [x] Generation throughput benchmark (batch size, cache on/off) *(milestone 1)*
- [ ] Loss curves plotted from JSONL logs
- [ ] Embedding space visualization (PCA/UMAP of token embeddings)
- [ ] Sampling parameter sweep (temperature x top-k/top-p) with grid report
- [ ] Reproduce nanoGPT shakespeare-char val loss as a regression benchmark
- [ ] Synthetic micro-benchmarks: arithmetic and copy/reverse tasks
- [ ] Automated eval suite in CI on a tiny model (fast, deterministic)

## Inference & Serving

- [ ] Streaming generation CLI (token-by-token output)
- [ ] Interactive REPL with prompt history and reset
- [ ] Batch generation from a prompts file
- [ ] OpenAI-compatible HTTP API (FastAPI, optional dependency)
- [ ] Checkpoint export to Hugging Face Transformers format
- [ ] ONNX export + onnxruntime inference path
- [ ] Dynamic int8 quantization experiment with quality/speed report
- [ ] Speculative decoding with a tiny draft model
- [ ] Structured generation (prefix constraints, stop sequences)
- [ ] Model card template and example published checkpoint

## Tooling & CI

- [x] pre-commit hooks (ruff check + format) *(milestone 1)*
- [x] Coverage reporting with pytest-cov and badge *(milestone 1)*
- [x] Multi-Python CI matrix (3.12 / 3.13) *(milestone 1)*
- [ ] Type checking (mypy or pyright) in CI on src/
- [ ] Release workflow: build sdist/wheel on tag, publish to PyPI
- [ ] Dockerfile for reproducible training runs
- [ ] Dependabot config for dependency updates
- [ ] GPU smoke-test workflow (manual trigger, tiny training run)
- [ ] Benchmark regression job (fail PR on large throughput drop)
- [ ] Makefile or justfile with common dev commands

## Docs

- [x] README overhaul with badges, diagram, CLI reference, honest results section *(milestone 1)*
- [x] Google Colab one-click training notebook *(milestone 1)*
- [x] ARCHITECTURE.md tensor-shape walkthrough of a forward pass *(milestone 1)*
- [ ] MATH.md derivations: attention, cross-entropy, AdamW update rule
- [ ] Tutorial notebook building the model cell-by-cell from an empty file
- [x] CONTRIBUTING.md with dev setup and conventions *(milestone 1)*
- [ ] CODE_OF_CONDUCT.md and SECURITY.md
- [ ] mkdocs-material docs site published to GitHub Pages
- [ ] API reference generated from docstrings
- [ ] FAQ: why char-level, why pre-norm, why tied embeddings
- [ ] Comparison table vs nanoGPT / minGPT / lit-gpt
- [x] CHANGELOG with tagged semver releases *(milestone 1)*
- [ ] Demo GIF of sampling in the README
- [ ] Blog-style write-up of training results with plots

## Experiments & Research

- [ ] Scaling study: val loss vs parameter count on Shakespeare (4-5 sizes, plotted)
- [ ] Ablation: pre-norm vs post-norm at this scale
- [ ] Ablation: tied vs untied embeddings
- [ ] Ablation: LR schedule variants (constant vs cosine vs warmup+cosine)
- [ ] Grokking experiment on modular arithmetic
- [ ] Induction-head analysis: attention patterns across training checkpoints
- [ ] Byte-level model on an enwik8 subset with bpc compared to literature
- [ ] Temperature/top-k/top-p sample-quality evaluation grid
- [ ] Knowledge distillation: teacher to tiny student on Shakespeare
- [ ] Efficiency target: val loss < 1.5 on Shakespeare-char in under 5 minutes on a T4
- [ ] Loss-spike diagnosis and recovery experiment (LR restarts, clip ablation)
- [ ] Research-log series (docs/research/*.md) documenting each experiment
