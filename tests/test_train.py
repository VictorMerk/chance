from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.train import (
    ModelEMA,
    TopKCheckpoint,
    build_overfit_batches,
    enable_determinism,
    estimate_loss,
    get_batch,
    update_top_k_checkpoints,
)


def tiny_args() -> argparse.Namespace:
    return argparse.Namespace(batch_size=2, block_size=8, eval_iters=2)


def tiny_model() -> GPT:
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=16, dropout=0.0)
    return GPT(config)


def test_get_batch_shapes_and_dtype_on_cpu() -> None:
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8, device=torch.device("cpu"))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_get_batch_targets_are_shifted_inputs() -> None:
    data = torch.arange(50)
    x, y = get_batch(data, batch_size=2, block_size=8, device=torch.device("cpu"))
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_estimate_loss_returns_finite_floats() -> None:
    model = tiny_model()
    data = torch.randint(0, 11, (64,))
    losses = estimate_loss(model, {"train": data}, tiny_args(), torch.device("cpu"))
    assert set(losses) == {"train"}
    for value in losses.values():
        assert isinstance(value, float)
        assert math.isfinite(value)


def _linear_model() -> nn.Linear:
    return nn.Linear(3, 2)


def test_ema_matches_manual_computation_for_three_steps() -> None:
    model = _linear_model()
    ema = ModelEMA(model, decay=0.9)
    manual = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        for value in (1.0, 2.0, 3.0):
            for param in model.parameters():
                param.fill_(value)
            ema.update(model)
            for name, tensor in model.state_dict().items():
                manual[name].mul_(0.9).add_(tensor.detach(), alpha=0.1)
    assert set(ema.shadow) == set(manual)
    for name, shadow in ema.shadow.items():
        assert torch.allclose(shadow, manual[name])


def test_ema_converges_toward_constant_and_leaves_model_untouched() -> None:
    model = _linear_model()
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(5.0)
        for _ in range(300):
            ema.update(model)
    for shadow in ema.shadow.values():
        assert torch.allclose(shadow, torch.full_like(shadow, 5.0), atol=1e-3)
    # apply_to/restore_from round trip must not perturb the live weights
    backup = ema.apply_to(model)
    ema.restore_from(model, backup)
    for tensor in model.state_dict().values():
        assert torch.equal(tensor, torch.full_like(tensor, 5.0))


def test_enable_determinism_seeds_reproducibly() -> None:
    prev_algorithms = torch.are_deterministic_algorithms_enabled()
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        enable_determinism(123)
        first = torch.randint(0, 1000, (8,))
        enable_determinism(123)
        second = torch.randint(0, 1000, (8,))
        assert torch.equal(first, second)
        enable_determinism(124)
        third = torch.randint(0, 1000, (8,))
        assert not torch.equal(first, third)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
    finally:
        torch.use_deterministic_algorithms(prev_algorithms)
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic


def _build_payload() -> dict[str, object]:
    return {"model_state": {}}


def test_update_top_k_writes_and_prunes_by_val_loss(tmp_path: Path) -> None:
    entries: list[TopKCheckpoint] | None = []
    for step, val_loss in enumerate((3.0, 1.0, 2.0, 0.5, 4.0), start=1):
        entries = update_top_k_checkpoints(tmp_path, val_loss, step, 2, _build_payload, entries)
    files = sorted(path.name for path in tmp_path.glob("checkpoint-top*.pt"))
    assert files == ["checkpoint-top1.pt", "checkpoint-top2.pt"]
    best = torch.load(tmp_path / "checkpoint-top1.pt", weights_only=False)
    runner_up = torch.load(tmp_path / "checkpoint-top2.pt", weights_only=False)
    assert best["val_loss"] == 0.5
    assert best["rank"] == 1
    assert best["iteration"] == 4
    assert best["model_state"] == {}
    assert runner_up["val_loss"] == 1.0
    assert runner_up["rank"] == 2
    assert runner_up["iteration"] == 2


def test_update_top_k_rediscovery_matches_disk_state(tmp_path: Path) -> None:
    update_top_k_checkpoints(tmp_path, 1.0, 10, 2, _build_payload, [])
    # entries=None simulates a fresh process discovering checkpoints on disk
    entries = update_top_k_checkpoints(tmp_path, 0.5, 11, 2, _build_payload, None)
    assert [entry.val_loss for entry in entries] == [0.5, 1.0]
    assert entries[0].path.name == "checkpoint-top1.pt"
    # a worse eval leaves the kept set untouched
    again = update_top_k_checkpoints(tmp_path, 9.0, 12, 2, _build_payload, None)
    assert [(entry.val_loss, entry.iteration) for entry in again] == [(0.5, 11), (1.0, 10)]
    assert sorted(path.name for path in tmp_path.glob("checkpoint-top*.pt")) == [
        "checkpoint-top1.pt",
        "checkpoint-top2.pt",
    ]


def test_overfit_batches_are_cached_and_repeat_identically() -> None:
    data = torch.arange(64)
    torch.manual_seed(7)
    cache = build_overfit_batches(data, 3, 2, 8, torch.device("cpu"))
    assert len(cache) == 3
    for x, y in cache:
        assert x.shape == (2, 8)
        assert y.shape == (2, 8)
        assert torch.equal(y[:, :-1], x[:, 1:])
    # training indexes cache[i % len(cache)], so later iterations see identical tensors
    first_x, first_y = cache[0 % len(cache)]
    cycled_x, cycled_y = cache[len(cache) % len(cache)]
    assert torch.equal(cycled_x, first_x)
    assert torch.equal(cycled_y, first_y)
    # resampling with the same seed reproduces the exact cache contents
    torch.manual_seed(7)
    again = build_overfit_batches(data, 3, 2, 8, torch.device("cpu"))
    for (x0, y0), (x1, y1) in zip(cache, again, strict=True):
        assert torch.equal(x0, x1)
        assert torch.equal(y0, y1)
