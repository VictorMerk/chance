import pytest

from gpt_from_scratch.tokenizer import CharTokenizer


def test_from_text_builds_sorted_unique_vocab() -> None:
    tokenizer = CharTokenizer.from_text("cbaabc")
    assert tokenizer.vocab == ["a", "b", "c"]
    assert tokenizer.vocab_size == 3


def test_encode_decode_roundtrip() -> None:
    tokenizer = CharTokenizer.from_text("hello world")
    ids = tokenizer.encode("hello")
    assert ids == [tokenizer.vocab.index(ch) for ch in "hello"]
    assert tokenizer.decode(ids) == "hello"


def test_encode_rejects_unknown_characters() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="unknown characters"):
        tokenizer.encode("abd")


def test_decode_roundtrip_on_full_vocab() -> None:
    text = "The quick brown fox jumps over the lazy dog.\n"
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_duplicate_vocab_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CharTokenizer(vocab=["a", "b", "a"])
