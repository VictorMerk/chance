import random

import pytest

from gpt_from_scratch.bpe import BPETokenizer

TRAIN_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks and the fox runs away. "
    "over and over the quick fox jumps the lazy dog. "
    "aaabdaaabac aaabdaaabac aaabdaaabac\n"
)


def _trained(vocab_size: int = 280, text: str = TRAIN_TEXT) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size)
    return tokenizer


def test_train_reaches_requested_vocab_size() -> None:
    tokenizer = _trained(280)
    assert tokenizer.vocab_size == 280
    assert len(tokenizer.merges) == 24


def test_train_encodes_compressible_text_shorter_than_bytes() -> None:
    tokenizer = _trained(300)
    ids = tokenizer.encode(TRAIN_TEXT)
    assert len(ids) < len(TRAIN_TEXT.encode("utf-8"))


def test_roundtrip_on_training_text_and_random_substrings() -> None:
    tokenizer = _trained(276)
    assert tokenizer.decode(tokenizer.encode(TRAIN_TEXT)) == TRAIN_TEXT
    rng = random.Random(0)
    for _ in range(20):
        start = rng.randrange(len(TRAIN_TEXT))
        end = rng.randrange(start, len(TRAIN_TEXT)) + 1
        substring = TRAIN_TEXT[start:end]
        assert tokenizer.decode(tokenizer.encode(substring)) == substring


def test_save_load_roundtrip(tmp_path) -> None:
    tokenizer = _trained(288)
    path = tmp_path / "bpe.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    inputs = [TRAIN_TEXT, "aaabdaaabac", "the quick fox", "", "unseen text!"]
    assert all(loaded.encode(s) == tokenizer.encode(s) for s in inputs)


def test_vocab_size_below_256_raises() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        _trained(255)


def test_unreachable_vocab_size_raises() -> None:
    with pytest.raises(ValueError, match="achievable"):
        _trained(300, text="ab")


def test_encode_unseen_characters_falls_back_to_bytes() -> None:
    tokenizer = _trained()
    unseen = "héllo wörld — 你好，世界 🌍"
    ids = tokenizer.encode(unseen)
    assert all(i < tokenizer.vocab_size for i in ids)
    assert tokenizer.decode(ids) == unseen
