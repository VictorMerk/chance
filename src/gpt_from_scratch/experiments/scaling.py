"""Scaling probe: train nano/micro/small presets on shakespeare-char and compare.

Usage::

    python -m gpt_from_scratch.experiments.scaling --max-iters 300 --plot

Writes ``[{name, params, val_loss, seconds}]`` to a JSON file, prints a table,
and (optionally) plots validation loss against parameter count on a log-x axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpt_from_scratch.experiments._runner import (
    RunResult,
    format_table,
    require_matplotlib,
    run_short_train,
)
from gpt_from_scratch.model import GPT_PRESETS

DEFAULT_SIZES = ("nano", "micro", "small")

# Validated categorical slot 1 (dataviz reference palette); single series needs no legend.
SERIES_COLOR = "#2a78d6"
TEXT_SECONDARY = "#52514e"


def parse_sizes(spec: str) -> list[str]:
    """Parse a comma-separated preset-name list; every name must exist in GPT_PRESETS."""
    names = [part.strip() for part in spec.split(",") if part.strip()]
    if not names:
        raise ValueError(f"--sizes must name at least one preset from {sorted(GPT_PRESETS)}")
    unknown = [name for name in names if name not in GPT_PRESETS]
    if unknown:
        raise ValueError(f"unknown size(s) {unknown}; available presets: {sorted(GPT_PRESETS)}")
    return names


def preset_overrides(name: str) -> dict[str, int]:
    """Translate a GPT_PRESETS entry into config overrides for the shared runner."""
    preset = GPT_PRESETS[name]
    return {"n_layer": preset.n_layer, "n_head": preset.n_head, "n_embd": preset.n_embd}


def build_rows(results: list[dict[str, Any]]) -> list[list[str]]:
    """Format result dicts as table cell strings (name, params, val_loss, seconds)."""
    rows: list[list[str]] = []
    for result in results:
        params = int(result["params"])
        rows.append(
            [
                str(result["name"]),
                f"{params:,}",
                f"{float(result['val_loss']):.4f}",
                f"{float(result['seconds']):.1f}",
            ]
        )
    return rows


def plot_scaling(results: list[dict[str, Any]], out_path: Path) -> Path:
    """Save val-loss-vs-params (log-x) with one labeled point per size."""
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xs = [int(result["params"]) for result in results]
    ys = [float(result["val_loss"]) for result in results]
    ax.plot(xs, ys, color=SERIES_COLOR, linewidth=2.0, marker="o", markersize=8)
    for x, y, result in zip(xs, ys, results, strict=True):
        ax.annotate(str(result["name"]), (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("validation loss")
    ax.set_title("Scaling: val loss vs model size (shakespeare-char)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(colors=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPT_PRESETS sizes on tiny Shakespeare-char and compare val losses"
    )
    parser.add_argument("--sizes", type=str, default=",".join(DEFAULT_SIZES))
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("scaling_results.json"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-out", type=Path, default=Path("scaling.png"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)

    results: list[dict[str, Any]] = []
    for name in sizes:
        print(f"training {name} for {args.max_iters} iters ...")
        result: RunResult = run_short_train(
            preset_overrides(name),
            args.max_iters,
            args.device,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        results.append({"name": name, **result})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(format_table(("name", "params", "val_loss", "seconds"), build_rows(results)))
    print(f"wrote results to {args.out}")

    if args.plot:
        path = plot_scaling(results, args.plot_out)
        print(f"saved plot to {path}")


if __name__ == "__main__":
    main()
