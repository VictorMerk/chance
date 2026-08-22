from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from gpt_from_scratch.data import download_tiny_shakespeare, load_text, train_val_split
from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.schedule import get_lr
from gpt_from_scratch.tokenizer import CharTokenizer

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


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
    parser.add_argument("--lr-warmup-iters", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
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
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def forward_loss(
    model: GPT, x: torch.Tensor, y: torch.Tensor, amp_dtype: torch.dtype | None
) -> torch.Tensor:
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )
    with ctx:
        _, loss = model(x, y)
    return loss


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split, data in splits.items():
        total = 0.0
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            loss = forward_loss(model, x, y, amp_dtype)
            total += loss.item()
        losses[split] = total / args.eval_iters
    model.train()
    return losses


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    vocab: list[str],
    iteration: int,
    val_loss: float,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "vocab": vocab,
            "iteration": iteration,
            "val_loss": val_loss,
        },
        path,
    )


def format_eta(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main() -> None:
    args = parse_args()
    if args.grad_accum < 1:
        raise ValueError("grad_accum must be at least 1")
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

    start_iter = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iter = int(checkpoint["iteration"])
        best_val_loss = float(checkpoint.get("val_loss", float("inf")))
        print(f"resumed from {args.resume} at iteration {start_iter}")

    min_lr = args.lr * args.lr_min_ratio
    amp_dtype = DTYPES[args.dtype] if device.type == "cuda" else None
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"device={device} parameters={model.num_parameters():,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    eval_start = time.time()
    last_eval_iter = start_iter
    tokens_since_eval = 0
    losses: dict[str, float] = {}
    for iteration in range(start_iter + 1, args.max_iters + 1):
        lr = get_lr(
            iteration,
            max_lr=args.lr,
            min_lr=min_lr,
            warmup_iters=args.lr_warmup_iters,
            max_iters=args.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.block_size, device)
            loss = forward_loss(model, x, y, amp_dtype)
            scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        tokens_since_eval += tokens_per_step

        if iteration % args.eval_interval == 0 or iteration == args.max_iters:
            losses = estimate_loss(model, splits, args, device, amp_dtype)
            elapsed = time.time() - eval_start
            steps_since_eval = iteration - last_eval_iter
            tokens_per_sec = tokens_since_eval / elapsed if elapsed > 0 else 0.0
            seconds_per_step = elapsed / steps_since_eval if steps_since_eval > 0 else 0.0
            eta_seconds = (args.max_iters - iteration) * seconds_per_step
            print(
                f"iter {iteration:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:,.0f} tok/s | eta {format_eta(eta_seconds)} | {elapsed:.1f}s"
            )
            if args.log_file is not None:
                record = {
                    "iter": iteration,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                }
                with args.log_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_path = args.out_dir / "checkpoint-best.pt"
                save_checkpoint(
                    model, optimizer, config, tokenizer.vocab, iteration, best_val_loss, best_path
                )
                print(f"saved checkpoint to {best_path}")
            last_eval_iter = iteration
            tokens_since_eval = 0
            eval_start = time.time()

    if not losses:
        losses = estimate_loss(model, splits, args, device, amp_dtype)

    final_path = args.out_dir / "checkpoint.pt"
    save_checkpoint(
        model, optimizer, config, tokenizer.vocab, args.max_iters, losses["val"], final_path
    )
    meta = {"val_loss": losses["val"], "parameters": model.num_parameters()}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved checkpoint to {final_path}")


if __name__ == "__main__":
    main()
