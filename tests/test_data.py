import pytest

from gpt_from_scratch.data import train_val_split


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
