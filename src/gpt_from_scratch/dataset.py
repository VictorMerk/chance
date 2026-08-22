"""Pre-tokenized dataset utilities: raw binary token files, corpus stats, and a CLI."""

from __future__ import annotations

import argparse
import json
import mmap
import sys
import warnings
from array import array
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean
from typing import NotRequired, TypedDict

import torch

TYPECODES: dict[str, str] = {"uint16": "H", "uint32": "I"}
ITEMSIZES: dict[str, int] = {"uint16": 2, "uint32": 4}
MAX_VALUES: dict[str, int] = {"uint16": 2**16 - 1, "uint32": 2**32 - 1}
TORCH_DTYPES: dict[str, torch.dtype] = {"uint16": torch.uint16, "uint32": torch.uint32}


class SplitSummary(TypedDict):
    tokens: int
    min_id: int | None
    max_id: int | None
    mean_id: float | None


class TopToken(TypedDict):
    id: int
    count: int
    preview: NotRequired[str]


class DatasetStats(TypedDict):
    train: SplitSummary
    val: SplitSummary
    total_tokens: int
    top10: list[TopToken]


def _resolve_dtype(dtype: str) -> str:
    if dtype not in TYPECODES:
        supported = ", ".join(sorted(TYPECODES))
        raise ValueError(f"unsupported dtype {dtype!r}; expected one of: {supported}")
    return dtype


def save_tokens_bin(ids: Sequence[int] | torch.Tensor, path: Path, dtype: str = "uint16") -> Path:
    """Write token ids as raw little-endian binary; parent dirs are created as needed."""
    _resolve_dtype(dtype)
    values = ids.detach().reshape(-1).tolist() if isinstance(ids, torch.Tensor) else list(ids)
    try:
        payload = array(TYPECODES[dtype], values)  # rejects out-of-range ids itself
    except OverflowError as exc:
        raise ValueError(f"token ids must fit in {dtype}, i.e. 0..{MAX_VALUES[dtype]}") from exc
    if payload.itemsize != ITEMSIZES[dtype]:  # guard against exotic C platforms
        raise RuntimeError(f"typecode {TYPECODES[dtype]!r} is not {ITEMSIZES[dtype]} bytes here")
    if sys.byteorder == "big":
        payload.byteswap()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload.tobytes())
    return path


def load_tokens_bin(path: Path, dtype: str = "uint16") -> torch.Tensor:
    """Load a raw binary token file into an owned ``torch.long`` tensor.

    The file is memory-mapped read-only and parsed zero-copy via
    ``torch.frombuffer``; the single widening copy to ``torch.long`` detaches the
    result from the mapping (safe after close) and makes it directly usable for
    embedding lookups. The file has no header, so ``dtype`` must match how it
    was written.
    """
    _resolve_dtype(dtype)
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        return torch.empty(0, dtype=torch.long)
    if size % ITEMSIZES[dtype] != 0:
        raise ValueError(
            f"{path}: file size {size} is not a multiple of the {dtype} "
            f"itemsize ({ITEMSIZES[dtype]} bytes)"
        )
    with path.open("rb", buffering=0) as handle:
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            with warnings.catch_warnings():
                # The view is never written through, so PyTorch's notice about
                # non-writable buffers is expected noise here.
                warnings.filterwarnings(
                    "ignore", message="The given buffer is not writable", category=UserWarning
                )
                view = torch.frombuffer(mapping, dtype=TORCH_DTYPES[dtype])
            return view.long()
        finally:
            mapping.close()


def _as_id_list(ids: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(ids, torch.Tensor):
        return ids.detach().reshape(-1).tolist()
    return list(ids)


def _split_summary(ids: list[int]) -> SplitSummary:
    if not ids:
        return {"tokens": 0, "min_id": None, "max_id": None, "mean_id": None}
    return {
        "tokens": len(ids),
        "min_id": min(ids),
        "max_id": max(ids),
        "mean_id": fmean(ids),
    }


def dataset_stats(
    train_ids: Sequence[int] | torch.Tensor,
    val_ids: Sequence[int] | torch.Tensor,
    decode: Callable[[list[int]], str] | None = None,
) -> DatasetStats:
    """Pure summary of two splits: totals, id ranges, and top-10 id frequencies.

    When ``decode`` is given, each top-10 entry also carries a decoded preview.
    """
    train = _as_id_list(train_ids)
    val = _as_id_list(val_ids)
    counts: Counter[int] = Counter(train)
    counts.update(val)
    top10: list[TopToken] = []
    for token_id, count in counts.most_common(10):
        entry: TopToken = {"id": token_id, "count": count}
        if decode is not None:
            entry["preview"] = decode([token_id])[:32]
        top10.append(entry)
    return {
        "train": _split_summary(train),
        "val": _split_summary(val),
        "total_tokens": len(train) + len(val),
        "top10": top10,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report statistics for pre-tokenized .bin data")
    parser.add_argument(
        "--bin-prefix",
        type=str,
        default="data/",
        help="path prefix of the token files; loads <prefix>train.bin and <prefix>val.bin, "
        "plus <prefix>train.bin.vocab.json for decoded previews when present",
    )
    return parser.parse_args(argv)


def _load_vocab(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    vocab = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(vocab, list) or not all(isinstance(token, str) for token in vocab):
        raise ValueError(f"{path} must be a JSON array of strings")
    return vocab


def _decoder_from_vocab(vocab: list[str]) -> Callable[[list[int]], str]:
    def decode(ids: list[int]) -> str:
        return "".join(vocab[i] if 0 <= i < len(vocab) else "?" for i in ids)

    return decode


def _format_stat(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}"


def main() -> None:
    args = parse_args()
    prefix: str = args.bin_prefix
    train_path = Path(prefix + "train.bin")
    val_path = Path(prefix + "val.bin")
    vocab_path = Path(prefix + "train.bin.vocab.json")
    vocab = _load_vocab(vocab_path)
    decode = _decoder_from_vocab(vocab) if vocab is not None else None
    stats = dataset_stats(load_tokens_bin(train_path), load_tokens_bin(val_path), decode)

    print(f"train file : {train_path}")
    print(f"val file   : {val_path}")
    print(f"{'split':<6} {'tokens':>12} {'min_id':>8} {'max_id':>8} {'mean_id':>9}")
    for name in ("train", "val"):
        row = stats[name]
        print(
            f"{name:<6} {row['tokens']:>12,} {_format_stat(row['min_id']):>8} "
            f"{_format_stat(row['max_id']):>8} {_format_stat(row['mean_id']):>9}"
        )
    print(f"{'total':<6} {stats['total_tokens']:>12,}")
    print("top-10 most frequent token ids:")
    for entry in stats["top10"]:
        line = f"  id {entry['id']:>6}  count {entry['count']:>10,}"
        if "preview" in entry:
            line += f"  {entry['preview']!r}"
        print(line)
    if vocab is None:
        print(f"(no vocab at {vocab_path}; decoded previews skipped)")


if __name__ == "__main__":
    main()
