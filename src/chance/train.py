from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from chance.data import download_tiny_shakespeare, load_text, train_val_split
from chance.model import GPT, GPTConfig
from chance.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT on tiny Shakespeare")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, data in splits.items():
        total = 0.0
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / args.eval_iters
    model.train()
    return losses


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    text_path = download_tiny_shakespeare(args.data_dir)
    train_text, val_text = train_val_split(load_text(text_path))
    tokenizer = CharTokenizer.from_text(train_text + val_text)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    splits = {"train": train_data, "val": val_data}

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay)
    print(f"device={device} parameters={model.num_parameters():,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for iteration in range(1, args.max_iters + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            losses = estimate_loss(model, splits, args, device)
            elapsed = time.time() - t0
            print(
                f"iter {iteration:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {elapsed:.1f}s"
            )

    checkpoint_path = args.out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "vocab": tokenizer.vocab,
            "iteration": args.max_iters,
            "val_loss": losses["val"],
        },
        checkpoint_path,
    )
    meta = {"val_loss": losses["val"], "parameters": model.num_parameters()}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
