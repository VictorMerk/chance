from __future__ import annotations

import urllib.request
from pathlib import Path

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

CORPUS_SUFFIXES = frozenset({".txt", ".md"})


def download_tiny_shakespeare(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "input.txt"
    if not path.exists():
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_corpus(path: Path) -> str:
    """Read a UTF-8 .txt or .md corpus file."""
    suffix = path.suffix.lower()
    if suffix not in CORPUS_SUFFIXES:
        supported = ", ".join(sorted(CORPUS_SUFFIXES))
        raise ValueError(f"unsupported corpus type {suffix!r}; expected one of: {supported}")
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"corpus file not found: {path}") from exc


def train_val_split(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    split = int(len(text) * (1.0 - val_fraction))
    return text[:split], text[split:]
