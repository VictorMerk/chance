from typing import Any

import pytest

from gpt_from_scratch.schedule import get_lr


def lr_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = {
        "max_lr": 1.0,
        "min_lr": 0.1,
        "warmup_iters": 100,
        "max_iters": 1000,
    }
    kwargs.update(overrides)
    return kwargs


def test_step_zero_returns_zero() -> None:
    assert get_lr(0, **lr_kwargs()) == 0.0


def test_warmup_is_linear() -> None:
    assert get_lr(50, **lr_kwargs()) == pytest.approx(0.5)
    assert get_lr(100, **lr_kwargs()) == pytest.approx(1.0)


def test_end_of_decay_reaches_min_lr() -> None:
    assert get_lr(1000, **lr_kwargs()) == pytest.approx(0.1, abs=1e-6)


def test_after_max_iters_is_constant_min_lr() -> None:
    assert get_lr(1100, **lr_kwargs()) == pytest.approx(0.1)


def test_decay_midpoint_is_average_of_extremes() -> None:
    value = get_lr(550, **lr_kwargs())
    assert abs(value - (1.0 + 0.1) / 2) < 1e-6


def test_zero_warmup_starts_at_max_lr() -> None:
    kwargs = lr_kwargs(warmup_iters=0)
    assert get_lr(0, **kwargs) == pytest.approx(1.0)


def test_equal_min_and_max_lr_is_constant() -> None:
    kwargs = lr_kwargs(max_lr=0.5, min_lr=0.5, warmup_iters=0)
    for step in (0, 5, 20, 2000):
        assert get_lr(step, **kwargs) == pytest.approx(0.5)


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        get_lr(-1, **lr_kwargs())
    with pytest.raises(ValueError):
        get_lr(10, **lr_kwargs(min_lr=2.0))
    with pytest.raises(ValueError):
        get_lr(10, **lr_kwargs(warmup_iters=-5))
    with pytest.raises(ValueError):
        get_lr(10, **lr_kwargs(max_iters=-5))
