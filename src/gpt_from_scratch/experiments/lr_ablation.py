"""LR schedule ablation: constant vs cosine-only vs warmup+cosine, same tiny config.

Usage::

    python -m gpt_from_scratch.experiments.lr_ablation --max-iters 300 --plot

All three modes train an identical tiny model with identical seeds and batch
order; only the learning-rate schedule differs. Prints a table and (optionally)
saves the three schedules next to the resulting val losses.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gpt_from_scratch.experiments._runner import (
    RunResult,
    format_table,
    require_matplotlib,
    run_short_train,
)
from gpt_from_scratch.schedule import get_lr

LR_MODES = ("constant", "cosine", "warmup_cosine")


def _make_lr_fn(schedule: list[float]) -> Callable[[int], float]:
    """Bind a precomputed schedule table into an ``lr_fn(step) -> lr`` callback."""

    def lr_fn(step: int) -> float:
        return schedule[step - 1]

    return lr_fn


# Validated categorical slots 1-3 in fixed mode order (dataviz reference palette).
MODE_COLORS: dict[str, str] = {
    "constant": "#2a78d6",
    "cosine": "#eb6834",
    "warmup_cosine": "#1baf7a",
}
TEXT_SECONDARY = "#52514e"


def lr_for_step(
    mode: str,
    step: int,
    *,
    max_iters: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
) -> float:
    """Learning rate at ``step`` (1-based) for the given schedule mode."""
    if not 1 <= step <= max_iters:
        raise ValueError(f"step {step} outside [1, {max_iters}]")
    if mode == "constant":
        return max_lr
    if mode == "cosine":
        return get_lr(step, max_lr=max_lr, min_lr=min_lr, warmup_iters=0, max_iters=max_iters)
    if mode == "warmup_cosine":
        return get_lr(
            step, max_lr=max_lr, min_lr=min_lr, warmup_iters=warmup_iters, max_iters=max_iters
        )
    raise ValueError(f"unknown lr mode {mode!r}; expected one of {list(LR_MODES)}")


def build_schedule(
    mode: str, max_iters: int, *, max_lr: float, min_lr: float, warmup_iters: int
) -> list[float]:
    """Schedule points for steps 1..max_iters, pure so it can be plotted or tested."""
    return [
        lr_for_step(
            mode, step, max_iters=max_iters, max_lr=max_lr, min_lr=min_lr, warmup_iters=warmup_iters
        )
        for step in range(1, max_iters + 1)
    ]


def schedule_fn(points: list[float]) -> Callable[[int], float]:
    """Turn schedule points into the runner's ``lr_fn(step)`` (steps are 1-based)."""
    return lambda step: points[step - 1]


def build_rows(results: list[dict[str, Any]]) -> list[list[str]]:
    """Format result dicts as table cell strings (mode, final lr, val_loss, seconds)."""
    rows: list[list[str]] = []
    for result in results:
        rows.append(
            [
                str(result["mode"]),
                f"{float(result['final_lr']):.2e}",
                f"{float(result['val_loss']):.4f}",
                f"{float(result['seconds']):.1f}",
            ]
        )
    return rows


def plot_ablation(
    schedules: dict[str, list[float]], results: list[dict[str, Any]], out_path: Path
) -> Path:
    """Save a two-panel figure: LR schedules overlay and final val-loss bars."""
    plt = require_matplotlib()
    fig, (ax_sched, ax_loss) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    val_by_mode = {str(result["mode"]): float(result["val_loss"]) for result in results}
    for mode, points in schedules.items():
        ax_sched.plot(
            range(1, len(points) + 1), points, label=mode, color=MODE_COLORS[mode], linewidth=2.0
        )
    ax_sched.set_xlabel("iteration")
    ax_sched.set_ylabel("learning rate")
    ax_sched.set_title("LR schedules")
    ax_sched.legend(frameon=False)
    ax_sched.grid(True, alpha=0.3)

    modes = [str(result["mode"]) for result in results]
    bars = ax_loss.bar(
        modes, [val_by_mode[mode] for mode in modes], color=[MODE_COLORS[mode] for mode in modes]
    )
    ax_loss.bar_label(bars, fmt="%.4f", fontsize=9)
    ax_loss.set_ylabel("validation loss")
    ax_loss.set_title("Final val loss")
    ax_loss.grid(True, axis="y", alpha=0.3)
    ax_loss.tick_params(axis="x", colors=TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare constant vs cosine vs warmup+cosine on identical tiny runs"
    )
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("lr_ablation_results.json"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-out", type=Path, default=Path("lr_ablation.png"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    overrides = {"n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd}

    results: list[dict[str, Any]] = []
    schedules: dict[str, list[float]] = {}
    for mode in LR_MODES:
        print(f"training with {mode} schedule for {args.max_iters} iters ...")
        schedule = build_schedule(
            mode,
            args.max_iters,
            max_lr=args.lr,
            min_lr=args.lr * 0.1,
            warmup_iters=args.warmup_iters,
        )
        schedules[mode] = schedule
        result: RunResult = run_short_train(
            overrides,
            args.max_iters,
            args.device,
            lr_fn=_make_lr_fn(schedule),
            lr=args.lr,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        results.append({"mode": mode, **result, "final_lr": schedule[-1]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(format_table(("mode", "final_lr", "val_loss", "seconds"), build_rows(results)))
    print(f"wrote results to {args.out}")

    if args.plot:
        path = plot_ablation(schedules, results, args.plot_out)
        print(f"saved plot to {path}")


if __name__ == "__main__":
    main()
