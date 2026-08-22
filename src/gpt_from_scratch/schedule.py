from __future__ import annotations

import math


def get_lr(
    step: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    max_iters: int,
) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_iters < 0 or max_iters < 0:
        raise ValueError("warmup_iters and max_iters must be non-negative")
    if min_lr > max_lr:
        raise ValueError("min_lr must not exceed max_lr")
    if warmup_iters > 0 and step <= warmup_iters:
        return step / warmup_iters * max_lr
    if step >= max_iters:
        return min_lr
    progress = (step - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)
