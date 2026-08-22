from __future__ import annotations

import argparse
import json
from pathlib import Path

LOG_KEYS = ("iter", "train_loss", "val_loss", "lr", "tokens_per_sec")


def load_losses(log_file: Path) -> dict[str, list[float]]:
    """Parse the JSONL log written by ``train.py --log-file`` into aligned series.

    Records logged before the first eval carry ``"val_loss": null``; those points
    become NaN so every series keeps the same length and matplotlib skips them.
    """
    series: dict[str, list[float]] = {key: [] for key in LOG_KEYS}
    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for key in LOG_KEYS:
                value = record.get(key)
                series[key].append(float("nan") if value is None else float(value))
    return series


def plot_loss(series: dict[str, list[float]], out_path: Path) -> Path:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: uv sync --extra plot") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(series["iter"], series["train_loss"], label="train")
    ax.plot(series["iter"], series["val_loss"], label="val", linestyle="--", marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from a train.py JSONL log")
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("losses.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = plot_loss(load_losses(args.log_file), args.out)
    print(f"saved plot to {path}")


if __name__ == "__main__":
    main()
