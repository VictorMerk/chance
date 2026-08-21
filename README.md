# chance

A small GPT-style language model built from scratch with PyTorch: character tokenizer, decoder-only transformer, training loop, and text generation. No transformer library used — attention, masking, weight tying, and the optimizer are implemented by hand.

## What is implemented

- Character-level tokenizer (`src/chance/tokenizer.py`)
- Decoder-only transformer: multi-head causal self-attention, MLP blocks, pre-norm residual stream, tied input/output embeddings (`src/chance/model.py`)
- AdamW with correct weight-decay grouping (decay for matrices, none for biases and layernorms)
- Training loop with train/val loss estimation and checkpointing (`src/chance/train.py`)
- Sampling with temperature and top-k (`src/chance/sample.py`)
- Tests covering tokenization roundtrips, causal masking, generation shape, and overfit behavior

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run chance-train --max-iters 5000
uv run chance-sample --prompt "ROMEO:" --max-new-tokens 500
```

On a machine without a GPU use a smaller model:

```bash
uv run chance-train --n-layer 4 --n-head 4 --n-embd 256 --batch-size 32 --max-iters 2000
```

## Training on Google Colab (free GPU)

Open [colab.research.google.com](https://colab.research.google.com), create a notebook, set the runtime to T4 GPU, then run:

```python
!git clone https://github.com/<your-user>/chance.git
%cd chance
!pip install -e . -q
!chance-train --device cuda
```

Copy the checkpoint back or sample directly in the same session:

```python
!chance-sample --checkpoint checkpoints/checkpoint.pt --prompt "ROMEO:"
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
