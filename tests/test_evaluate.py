import math

import torch

from gpt_from_scratch import benchmark, evaluate
from gpt_from_scratch.benchmark import benchmark_generation
from gpt_from_scratch.evaluate import bits_per_character, perplexity
from gpt_from_scratch.model import GPT, GPTConfig


def tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=11,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )


def test_perplexity_is_finite_and_above_one() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    data = torch.randint(0, 11, (256,))
    value = perplexity(
        model, data, block_size=8, batch_size=4, device=torch.device("cpu"), max_batches=3
    )
    assert math.isfinite(value)
    assert value > 1.0


def test_perplexity_restores_prior_mode() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    data = torch.randint(0, 11, (64,))
    model.train()
    perplexity(model, data, block_size=8, batch_size=2, device=torch.device("cpu"))
    assert model.training
    model.eval()
    perplexity(model, data, block_size=8, batch_size=2, device=torch.device("cpu"))
    assert not model.training


def test_bits_per_character_conversion() -> None:
    assert abs(bits_per_character(2.0, 1.0) - 2.0 / math.log(2)) < 1e-12


def test_benchmark_generation_returns_positive_finite_speed() -> None:
    torch.manual_seed(0)
    model = GPT(tiny_config())
    for use_cache in (True, False):
        tokens_per_sec = benchmark_generation(
            model,
            block_size=tiny_config().block_size,
            batch_size=2,
            max_new_tokens=5,
            use_cache=use_cache,
            device=torch.device("cpu"),
        )
        assert math.isfinite(tokens_per_sec)
        assert tokens_per_sec > 0.0


def test_main_functions_are_importable() -> None:
    assert callable(evaluate.main)
    assert callable(benchmark.main)
