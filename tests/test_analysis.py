from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.plots import load_losses, plot_loss
from gpt_from_scratch.sweep import distinct_n, mean_nll, run_grid
from gpt_from_scratch.tokenizer import CharTokenizer


def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=11,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )


def write_log(path: Path) -> Path:
    records = [
        {"iter": 10, "train_loss": 4.5, "val_loss": None, "lr": 1e-3, "tokens_per_sec": 9000.5},
        {"iter": 20, "train_loss": 2.25, "val_loss": 2.5, "lr": 5e-4, "tokens_per_sec": 10000.0},
    ]
    text = "".join(json.dumps(record) + "\n" for record in records) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_losses_roundtrip_with_null_val(tmp_path: Path) -> None:
    series = load_losses(write_log(tmp_path / "log.jsonl"))
    assert list(series) == ["iter", "train_loss", "val_loss", "lr", "tokens_per_sec"]
    assert series["iter"] == [10.0, 20.0]
    assert series["train_loss"] == [4.5, 2.25]
    assert math.isnan(series["val_loss"][0])
    assert series["val_loss"][1] == 2.5
    assert series["lr"] == [1e-3, 5e-4]
    assert series["tokens_per_sec"] == [9000.5, 10000.0]


def test_distinct_n_exact_values_on_toy_inputs() -> None:
    assert distinct_n([1, 2, 1, 2, 1], 1) == 0.4
    assert distinct_n([1, 2, 1, 2, 1], 2) == 0.5
    assert distinct_n([7, 7, 7, 7], 1) == 0.25
    assert distinct_n([1, 2, 3], 3) == 1.0
    assert distinct_n([1, 2], 3) == 0.0
    assert distinct_n([], 1) == 0.0


def test_mean_nll_is_finite_and_positive_on_random_model() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    ids = torch.randint(0, 11, (40,)).tolist()
    value = mean_nll(model, ids, block_size=16, device=torch.device("cpu"))
    assert math.isfinite(value)
    assert value > 0.0


def test_mean_nll_matches_direct_cross_entropy_within_one_block() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    ids = torch.randint(0, 11, (10,)).tolist()
    data = torch.tensor(ids)
    _, expected = model(data[:-1].unsqueeze(0), data[1:].unsqueeze(0))
    value = mean_nll(model, ids, block_size=16, device=torch.device("cpu"))
    assert abs(value - expected.item()) < 1e-6


def test_run_grid_smoke_single_combo_returns_finite_row() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    tokenizer = CharTokenizer.from_text("abcdefghijk")
    rows = run_grid(
        model,
        tokenizer,
        prompt_ids=[0, 1],
        temps=[0.8],
        top_ks=[0],
        top_ps=[0.0],
        samples_per_combo=2,
        max_new_tokens=8,
        device=torch.device("cpu"),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["temp"] == 0.8
    assert row["top_k"] is None
    assert row["top_p"] is None
    assert math.isfinite(row["nll"])
    assert row["nll"] >= 0.0
    for key in ("distinct1", "distinct2"):
        assert 0.0 <= row[key] <= 1.0


def test_run_grid_emits_one_unconstrained_row_per_temperature() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    tokenizer = CharTokenizer.from_text("abcdefghijk")
    rows = run_grid(
        model,
        tokenizer,
        prompt_ids=[0],
        temps=[1.0],
        top_ks=[0, 0, 50],
        top_ps=[0.0],
        samples_per_combo=1,
        max_new_tokens=2,
        device=torch.device("cpu"),
    )
    assert len(rows) == 2
    unconstrained = [row for row in rows if row["top_k"] is None and row["top_p"] is None]
    assert len(unconstrained) == 1


def test_plot_loss_writes_nonempty_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    series = {
        "iter": [0.0, 10.0, 20.0],
        "train_loss": [5.0, 3.0, 2.0],
        "val_loss": [float("nan"), 4.0, 3.5],
        "lr": [1e-3, 5e-4, 1e-4],
        "tokens_per_sec": [0.0, 0.0, 0.0],
    }
    out = tmp_path / "curves.png"
    assert plot_loss(series, out) == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_main_functions_exist() -> None:
    from gpt_from_scratch import plots, sweep

    assert callable(plots.main)
    assert callable(sweep.main)
