from pathlib import Path

import pytest

from gpt_from_scratch.data import load_corpus, train_val_split


def test_split_returns_expected_proportions() -> None:
    text = "abcdefghij"
    train, val = train_val_split(text, val_fraction=0.2)
    assert train == "abcdefgh"
    assert val == "ij"


def test_split_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        train_val_split("abc", val_fraction=0.0)
    with pytest.raises(ValueError):
        train_val_split("abc", val_fraction=1.0)


def test_load_corpus_reads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "corpus.md"
    path.write_text("# Title\n\nbody text\n", encoding="utf-8")
    assert load_corpus(path) == "# Title\n\nbody text\n"


def test_load_corpus_reads_txt(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("hello", encoding="utf-8")
    assert load_corpus(path) == "hello"


def test_load_corpus_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="corpus"):
        load_corpus(tmp_path / "missing.txt")


def test_load_corpus_rejects_other_suffixes(tmp_path: Path) -> None:
    path = tmp_path / "corpus.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported corpus type"):
        load_corpus(path)
