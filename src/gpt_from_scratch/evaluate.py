from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from gpt_from_scratch.data import download_tiny_shakespeare, load_text, train_val_split
from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.tokenizer import CharTokenizer


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def perplexity(
    model: GPT,
    data: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    was_training = model.training
    model.eval()
    n_batches = 1 if max_batches is None else max_batches
    total = 0.0
    for _ in range(n_batches):
        x, y = _get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return math.exp(total / n_batches)


def bits_per_character(loss_nats: float, chars_per_token: float) -> float:
    return loss_nats / math.log(2) / chars_per_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perplexity of a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    text_path = download_tiny_shakespeare(args.data_dir)
    _, val_text = train_val_split(load_text(text_path))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(vocab=checkpoint["vocab"])
    config = GPTConfig.from_dict(checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    ppl = perplexity(
        model,
        val_data,
        block_size=config.block_size,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_batches,
    )
    chars_per_token = len(val_text) / len(val_data)
    bpc = bits_per_character(math.log(ppl), chars_per_token)
    print(f"val perplexity: {ppl:.2f} | bits/char: {bpc:.4f}")


if __name__ == "__main__":
    main()
