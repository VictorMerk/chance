import argparse
import math

import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.train import estimate_loss, get_batch


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
