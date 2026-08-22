from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from gpt_from_scratch import dataset
from gpt_from_scratch.dataset import (
    dataset_stats,
    load_tokens_bin,
    parse_args,
    save_tokens_bin,
)
from gpt_from_scratch.train import load_training_data
from gpt_from_scratch.train import parse_args as train_parse_args


def _write_bin_dataset(tmp_path: Path) -> Path:
    # uint16 is the documented on-disk dtype for train.py --data-format bin.
    save_tokens_bin([0, 1, 2, 2], tmp_path / "train.bin", dtype="uint16")
    save_tokens_bin([1], tmp_path / "val.bin", dtype="uint16")
    (tmp_path / "train.bin.vocab.json").write_text(json.dumps(["x", "y", "z"]), encoding="utf-8")
    return tmp_path


def test_save_load_roundtrip_uint16(tmp_path: Path) -> None:
    ids = [0, 1, 65, 65535, 42]
    path = save_tokens_bin(ids, tmp_path / "train.bin")
    assert path == tmp_path / "train.bin"
    assert path.stat().st_size == 2 * len(ids)
    loaded = load_tokens_bin(path)
    assert loaded.dtype == torch.long
    assert loaded.tolist() == ids


def test_save_load_roundtrip_uint32_large_values(tmp_path: Path) -> None:
    ids = [0, 65535, 65536, 123456789, 2**32 - 1]
    path = save_tokens_bin(ids, tmp_path / "tokens.bin", dtype="uint32")
    assert path.stat().st_size == 4 * len(ids)
    assert load_tokens_bin(path, dtype="uint32").tolist() == ids


def test_save_rejects_overflow_and_negatives(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fit in uint16"):
        save_tokens_bin([70000], tmp_path / "over.bin", dtype="uint16")
    with pytest.raises(ValueError):
        save_tokens_bin([-1], tmp_path / "neg.bin")


def test_save_accepts_tensor_input(tmp_path: Path) -> None:
    path = save_tokens_bin(torch.tensor([3, 1, 2]), tmp_path / "t.bin")
    assert load_tokens_bin(path).tolist() == [3, 1, 2]


@pytest.mark.parametrize(("dtype", "itemsize"), [("uint16", 2), ("uint32", 4)])
def test_load_corrupt_size_raises(tmp_path: Path, dtype: str, itemsize: int) -> None:
    path = tmp_path / f"corrupt-{dtype}.bin"
    path.write_bytes(b"\x01" * (itemsize + 1))
    with pytest.raises(ValueError, match="not a multiple"):
        load_tokens_bin(path, dtype=dtype)


def test_load_empty_file_returns_empty_tensor(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    loaded = load_tokens_bin(path)
    assert loaded.numel() == 0
    assert loaded.dtype == torch.long
    saved_empty = save_tokens_bin([], tmp_path / "saved-empty.bin")
    assert load_tokens_bin(saved_empty).numel() == 0


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tokens_bin(tmp_path / "nope.bin")


def test_unsupported_dtype_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        save_tokens_bin([1], tmp_path / "x.bin", dtype="int8")
    with pytest.raises(ValueError, match="unsupported dtype"):
        load_tokens_bin(tmp_path / "x.bin", dtype="int8")


def test_dataset_stats_sane_on_toy_data() -> None:
    stats = dataset_stats(torch.tensor([1, 1, 2, 3]), [2, 3, 3])
    assert stats["total_tokens"] == 7
    assert stats["train"]["tokens"] == 4
    assert stats["train"]["min_id"] == 1
    assert stats["train"]["max_id"] == 3
    assert stats["train"]["mean_id"] == pytest.approx(1.75)
    assert stats["val"]["tokens"] == 3
    top = stats["top10"]
    assert top[0]["id"] == 3
    assert top[0]["count"] == 3
    assert "preview" not in top[0]


def test_dataset_stats_with_decode_previews() -> None:
    vocab = ["a", "b", "c"]

    def decode(ids: list[int]) -> str:
        return "".join(vocab[i] for i in ids)

    stats = dataset_stats([0, 0, 1], [2], decode=decode)
    previews = {entry["id"]: entry["preview"] for entry in stats["top10"]}
    assert previews[0] == "a"


def test_dataset_stats_handles_empty_split() -> None:
    stats = dataset_stats([1, 2], [])
    assert stats["val"]["tokens"] == 0
    assert stats["val"]["mean_id"] is None
    assert stats["total_tokens"] == 2


def test_parse_args_default_prefix() -> None:
    args = parse_args([])
    assert args.bin_prefix == "data/"


def test_cli_main_prints_stats_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_bin_dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", ["gpt-from-scratch-dataset", "--bin-prefix", f"{tmp_path}/"])
    dataset.main()
    out = capsys.readouterr().out
    assert "train" in out
    assert "total" in out
    assert "'y'" in out  # decoded preview for the most frequent id


# --- train.py --data-format bin, exercised through the extracted loading helper ---


def test_load_training_data_bin_mode(tmp_path: Path) -> None:
    root = _write_bin_dataset(tmp_path)
    train_ids, val_ids, vocab = load_training_data(root, "bin")
    assert train_ids.dtype == torch.long
    assert train_ids.tolist() == [0, 1, 2, 2]
    assert val_ids.tolist() == [1]
    assert vocab == ["x", "y", "z"]


def test_load_training_data_bin_mode_requires_token_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pre-tokenized"):
        load_training_data(tmp_path, "bin")


def test_load_training_data_bin_mode_requires_vocab(tmp_path: Path) -> None:
    save_tokens_bin([0], tmp_path / "train.bin")
    save_tokens_bin([0], tmp_path / "val.bin")
    with pytest.raises(FileNotFoundError, match="vocab"):
        load_training_data(tmp_path, "bin")


def test_load_training_data_bin_mode_rejects_bad_vocab(tmp_path: Path) -> None:
    _write_bin_dataset(tmp_path)
    (tmp_path / "train.bin.vocab.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array of strings"):
        load_training_data(tmp_path, "bin")


def test_load_training_data_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data_format"):
        load_training_data(tmp_path, "parquet")


def test_train_parse_args_has_data_format_default_text() -> None:
    args = train_parse_args([])
    assert args.data_format == "text"
