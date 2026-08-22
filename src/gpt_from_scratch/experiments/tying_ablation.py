"""Tied vs untied input/output embeddings on identical twin configs.

Usage::

    python -m gpt_from_scratch.experiments.tying_ablation --max-iters 300

Both runs share seed, data order, schedule, and dimensions; only
``tie_embeddings`` differs. Prints a table with both val losses, the delta,
and each run's parameter count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gpt_from_scratch.experiments._runner import (
    RunResult,
    format_table,
    run_short_train,
)

VARIANTS = ("tied", "untied")


def overrides_for(variant: str) -> dict[str, Any]:
    """Config overrides selecting the embedding-tying variant."""
    if variant == "tied":
        return {"tie_embeddings": True}
    if variant == "untied":
        return {"tie_embeddings": False}
    raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")


def build_rows(results: list[dict[str, Any]]) -> list[list[str]]:
    """Format result dicts as table cell strings (variant, params, val_loss, seconds)."""
    rows: list[list[str]] = []
    for result in results:
        rows.append(
            [
                str(result["variant"]),
                f"{int(result['params']):,}",
                f"{float(result['val_loss']):.4f}",
                f"{float(result['seconds']):.1f}",
            ]
        )
    return rows


def delta_val_loss(results: list[dict[str, Any]]) -> float | None:
    """val_loss(untied) - val_loss(tied); positive means tying helped."""
    by_variant = {str(result["variant"]): result for result in results}
    if set(by_variant) != set(VARIANTS):
        return None
    return float(by_variant["untied"]["val_loss"]) - float(by_variant["tied"]["val_loss"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tied vs untied embeddings on identical twin runs"
    )
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("tying_ablation_results.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    base = {"n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd}

    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"training {variant} variant for {args.max_iters} iters ...")
        result: RunResult = run_short_train(
            {**base, **overrides_for(variant)},
            args.max_iters,
            args.device,
            lr=args.lr,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        results.append({"variant": variant, **result})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(format_table(("variant", "params", "val_loss", "seconds"), build_rows(results)))
    delta = delta_val_loss(results)
    if delta is not None:
        verdict = "tying helped" if delta > 0 else "untied helped" if delta < 0 else "tie"
        print(f"delta val_loss (untied - tied): {delta:+.4f} ({verdict})")
    print(f"wrote results to {args.out}")


if __name__ == "__main__":
    main()
