# gpt-from-scratch

A small GPT-style language model built from scratch with PyTorch: character tokenizer, decoder-only transformer, training loop, and text generation. No transformer library used — attention, masking, weight tying, and the optimizer are implemented by hand.

## What is implemented

- Character-level tokenizer (`src/gpt_from_scratch/tokenizer.py`)
- Decoder-only transformer: multi-head causal self-attention, MLP blocks, pre-norm residual stream, tied input/output embeddings (`src/gpt_from_scratch/model.py`)
- AdamW with correct weight-decay grouping (decay for matrices, none for biases and layernorms)
- Training loop with train/val loss estimation and checkpointing (`src/gpt_from_scratch/train.py`)
- Sampling with temperature and top-k (`src/gpt_from_scratch/sample.py`)
- Tests covering tokenization roundtrips, causal masking, generation shape, and overfit behavior

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run gpt-from-scratch-train --max-iters 5000
uv run gpt-from-scratch-sample --prompt "ROMEO:" --max-new-tokens 500
```

On a machine without a GPU use a smaller model:

```bash
uv run gpt-from-scratch-train --n-layer 4 --n-head 4 --n-embd 256 --batch-size 32 --max-iters 2000
```

## Training on Google Colab (free GPU)

Open [colab.research.google.com](https://colab.research.google.com), create a notebook, set the runtime to T4 GPU, then run:

```python
!git clone https://github.com/VictorMerk/gpt-from-scratch.git
%cd gpt-from-scratch
!pip install -e . -q
!gpt-from-scratch-train --device cuda
```

Copy the checkpoint back or sample directly in the same session:

```python
!gpt-from-scratch-sample --checkpoint checkpoints/checkpoint.pt --prompt "ROMEO:"
```

## Development

```bash
uv sync --dev
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

CI runs lint and tests on every push via GitHub Actions.

## Roadmap

- Byte-pair encoding tokenizer
- Learning-rate schedule (warmup + cosine decay)
- Gradient accumulation and mixed precision for larger runs
- Perplexity evaluation on held-out data
