from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gpt_from_scratch.model import GPT, GPTConfig


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def benchmark_generation(
    model: GPT,
    *,
    block_size: int,
    batch_size: int,
    max_new_tokens: int,
    use_cache: bool,
    device: torch.device,
) -> float:
    prompt_len = max(1, block_size - max_new_tokens)
    start = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len), device=device)
    was_training = model.training
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    model.generate(start, max_new_tokens=max_new_tokens, use_cache=use_cache)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    if was_training:
        model.train()
    return batch_size * max_new_tokens / elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cached vs uncached generation speed")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = GPTConfig.from_dict(checkpoint["config"])
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        config = GPTConfig(
            vocab_size=65, block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0
        )
        model = GPT(config).to(device)
    model.eval()
    cached = benchmark_generation(
        model,
        block_size=config.block_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        device=device,
    )
    uncached = benchmark_generation(
        model,
        block_size=config.block_size,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_cache=False,
        device=device,
    )
    print(f"cached   : {cached:,.0f} tokens/sec")
    print(f"uncached : {uncached:,.0f} tokens/sec")


if __name__ == "__main__":
    main()
