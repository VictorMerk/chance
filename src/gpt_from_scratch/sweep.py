from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import torch

from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.tokenizer import CharTokenizer

GridRow = dict[str, float | int | None]


def distinct_n(ids: list[int], n: int) -> float:
    """Fraction of distinct n-grams among the ``len(ids) - n + 1`` sliding windows."""
    if n <= 0:
        raise ValueError("n must be positive")
    total = len(ids) - n + 1
    if total <= 0:
        return 0.0
    grams = {tuple(ids[i : i + n]) for i in range(total)}
    return len(grams) / total


@torch.no_grad()
def mean_nll(model: GPT, ids: list[int], block_size: int, device: torch.device) -> float:
    """Teacher-forced cross-entropy of ``ids`` under the model, chunked to block_size.

    Chunks are weighted by token count so the result equals the NLL over the whole
    sequence regardless of where the block boundaries fall.
    """
    if len(ids) < 2:
        return 0.0
    was_training = model.training
    model.eval()
    data = torch.tensor(ids, dtype=torch.long, device=device)
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, block_size):
        chunk = data[start : start + block_size + 1]
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        _, loss = model(x, y)
        total_nll += loss.item() * y.numel()
        total_tokens += y.numel()
    if was_training:
        model.train()
    return total_nll / total_tokens


@torch.no_grad()
def run_grid(
    model: GPT,
    tokenizer: CharTokenizer,
    *,
    prompt_ids: list[int],
    temps: list[float],
    top_ks: list[int],
    top_ps: list[float | None],
    samples_per_combo: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[GridRow]:
    """Generate and score samples for every (temperature, top_k, top_p) combination.

    ``top_k`` of 0 disables top-k filtering and ``top_p`` of None or 0 disables
    top-p; combos that normalize to the same setting are emitted once, so each
    temperature gets a single unconstrained row. Metrics (nll, distinct1,
    distinct2) are averaged over the generated continuations only.
    """
    if samples_per_combo < 1:
        raise ValueError("samples_per_combo must be at least 1")
    rows: list[GridRow] = []
    seen: set[tuple[float, int | None, float | None]] = set()
    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for temp, top_k, top_p in product(temps, top_ks, top_ps):
        k = top_k if top_k is not None and top_k > 0 else None
        p = top_p if top_p else None
        key = (temp, k, p)
        if key in seen:
            continue
        seen.add(key)
        nlls: list[float] = []
        d1s: list[float] = []
        d2s: list[float] = []
        for _ in range(samples_per_combo):
            out = model.generate(
                start,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=k,
                top_p=p,
            )
            sample_ids = out[0, len(prompt_ids) :].tolist()
            nlls.append(mean_nll(model, sample_ids, model.config.block_size, device))
            d1s.append(distinct_n(sample_ids, 1))
            d2s.append(distinct_n(sample_ids, 2))
        rows.append(
            {
                "temp": temp,
                "top_k": k,
                "top_p": p,
                "nll": sum(nlls) / len(nlls),
                "distinct1": sum(d1s) / len(d1s),
                "distinct2": sum(d2s) / len(d2s),
            }
        )
    return rows


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_float_list(spec: str) -> list[float]:
    values = [float(part) for part in spec.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _parse_int_list(spec: str) -> list[int]:
    values = [int(part) for part in spec.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def _format_cell(column: str, value: float | None) -> str:
    if column == "temp":
        return f"{value:.2f}"
    if column == "top_k":
        return "off" if value is None else str(value)
    if column == "top_p":
        return "off" if value is None else f"{value:.2f}"
    return f"{value:.3f}"


def format_table(rows: list[GridRow]) -> str:
    columns = ("temp", "top_k", "top_p", "nll", "distinct1", "distinct2")
    cells = [[_format_cell(col, row[col]) for col in columns] for row in rows]
    widths = [
        max(len(col), max((len(line[i]) for line in cells), default=0))
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(col.rjust(widths[i]) for i, col in enumerate(columns)) + " |"
    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    body = [
        "| " + " | ".join(cell.rjust(widths[i]) for i, cell in enumerate(line)) + " |"
        for line in cells
    ]
    return "\n".join([header, separator, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search sampling parameters of a checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--temps", type=_parse_float_list, default=[0.5, 0.8, 1.0])
    parser.add_argument("--top-ks", type=_parse_int_list, default=[0, 50, 200])
    parser.add_argument("--top-ps", type=_parse_float_list, default=[0.0, 0.9])
    parser.add_argument("--samples-per-combo", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(vocab=checkpoint["vocab"])
    config = GPTConfig.from_dict(checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("--prompt must encode to at least one token")

    rows = run_grid(
        model,
        tokenizer,
        prompt_ids=prompt_ids,
        temps=args.temps,
        top_ks=args.top_ks,
        top_ps=args.top_ps,
        samples_per_combo=args.samples_per_combo,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    rows.sort(key=lambda row: float(row["nll"]))
    print(format_table(rows))


if __name__ == "__main__":
    main()
