"""Shared short-training runner and formatting helpers for experiment scripts.

Reuses the data loading, batching, and evaluation pieces of ``train.py`` (read
only) so every experiment trains under identical, comparable conditions without
invoking ``train.main()``.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.schedule import get_lr
from gpt_from_scratch.tokenizer import CharTokenizer
from gpt_from_scratch.train import estimate_loss, get_batch, load_training_data, resolve_device

RunResult = dict[str, float | int]

_DATA_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, list[str]]] = {}


def load_shakespeare_char(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load (and tokenize) tiny Shakespeare once per process, then serve from cache."""
    key = (str(data_dir.resolve()), "text")
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_training_data(data_dir, "text")
    return _DATA_CACHE[key]


def build_config(
    vocab_size: int, config_overrides: dict[str, Any], *, block_size: int, dropout: float
) -> GPTConfig:
    """Merge validated ``config_overrides`` on top of the runner's base config."""
    allowed = {field.name for field in dataclasses.fields(GPTConfig)}
    unknown = sorted(set(config_overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown GPTConfig field(s): {unknown}; allowed: {sorted(allowed)}")
    params: dict[str, Any] = {"block_size": block_size, "dropout": dropout}
    params.update(config_overrides)
    return GPTConfig(vocab_size=vocab_size, **params)


def run_short_train(
    config_overrides: dict[str, Any],
    max_iters: int = 300,
    device: str | torch.device | None = None,
    *,
    lr_fn: Callable[[int], float] | None = None,
    lr: float = 1e-3,
    lr_min_ratio: float = 0.1,
    warmup_iters: int = 100,
    batch_size: int = 32,
    block_size: int = 128,
    dropout: float = 0.1,
    eval_iters: int = 25,
    grad_clip: float = 1.0,
    weight_decay: float = 0.1,
    seed: int = 1337,
    data_dir: Path | None = None,
) -> RunResult:
    """Train one short char-level run and report final losses, parameter count, wall time.

    The schedule defaults to train.py's warmup+cosine; pass ``lr_fn(step) -> lr``
    to override it entirely. Seeding happens before model construction, so two
    calls with the same seed and config see identical initial weights and batch
    order (the basis for twin comparisons in the ablations).

    Returns ``{"train_loss", "val_loss", "params", "seconds"}``.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    resolved = resolve_device(device)
    data_dir = Path("data") if data_dir is None else Path(data_dir)
    train_data, val_data, vocab = load_shakespeare_char(data_dir)
    tokenizer = CharTokenizer(vocab=vocab)
    splits = {"train": train_data, "val": val_data}

    torch.manual_seed(seed)
    config = build_config(
        tokenizer.vocab_size, config_overrides, block_size=block_size, dropout=dropout
    )
    model = GPT(config).to(resolved)
    optimizer = model.configure_optimizers(lr=lr, weight_decay=weight_decay)

    min_lr = lr * lr_min_ratio
    start = time.perf_counter()
    for iteration in range(1, max_iters + 1):
        if lr_fn is not None:
            step_lr = lr_fn(iteration)
        else:
            step_lr = get_lr(
                iteration,
                max_lr=lr,
                min_lr=min_lr,
                warmup_iters=warmup_iters,
                max_iters=max_iters,
            )
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        x, y = get_batch(train_data, batch_size, block_size, resolved)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    seconds = time.perf_counter() - start

    eval_args = argparse.Namespace(
        batch_size=batch_size, block_size=block_size, eval_iters=eval_iters
    )
    losses = estimate_loss(model, splits, eval_args, resolved)
    return {
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "params": model.num_parameters(),
        "seconds": seconds,
    }


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render right-aligned markdown-style rows; all cells must be pre-formatted strings."""
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cell.rjust(widths[i]) for i, cell in enumerate(cells)) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def require_matplotlib() -> ModuleType:
    """Return pyplot, or raise with the exact fix for the optional plot extra."""
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting: uv sync --extra plot") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
