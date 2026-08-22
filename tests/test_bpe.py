import json
import random

import pytest

from gpt_from_scratch.bpe import BPETokenizer, _pretokenize

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


# --- GPT-2 style pre-tokenization -------------------------------------------


def test_pretokenize_contractions_and_runs() -> None:
    assert _pretokenize("I'm sure you'll agree, it's fine") == [
        "I",
        "'m",
        " sure",
        " you",
        "'ll",
        " agree",
        ",",
        " it",
        "'s",
        " fine",
    ]
    assert _pretokenize("abc123world45") == ["abc", "123", "world", "45"]
    assert _pretokenize("don't we'll've") == ["don", "'t", " we", "'ll", "'ve"]


def test_pretokenize_whitespace_attachment() -> None:
    # All but the last char of a whitespace run stand alone; the last literal
    # space attaches to the following chunk.
    assert _pretokenize("  abc  def  ") == [" ", " abc", " ", " def", "  "]
    assert _pretokenize("a b") == ["a", " b"]
    # Only a literal space attaches; other whitespace always stands alone.
    assert _pretokenize("a\n\nb") == ["a", "\n", "\n", "b"]
    assert _pretokenize("a \nb") == ["a", " ", "\n", "b"]
    assert _pretokenize("\t\n x") == ["\t\n", " x"]


def test_pretokenize_edge_cases() -> None:
    assert _pretokenize("") == []
    assert _pretokenize("   ") == ["   "]
    assert _pretokenize("'s") == ["'s"]
    assert _pretokenize("'S") == ["'", "S"]
    assert _pretokenize("!'s") == ["!'", "s"]
    assert _pretokenize(" 's") == [" '", "s"]  # no contraction right after a space
    assert _pretokenize("你好世界 🌍") == ["你好世界", " 🌍"]


def test_merges_respect_chunk_boundaries() -> None:
    tokenizer = _trained(300)
    for new_id in range(256, 256 + len(tokenizer.merges)):
        blob = tokenizer.decode([new_id])
        # Within a chunk a space can only be an attached leading space, so a
        # merged token with an interior space would have crossed a boundary.
        assert blob.isspace() or " " not in blob[1:], blob


# --- Special tokens ----------------------------------------------------------


def test_default_tokenizer_has_no_special_tokens() -> None:
    tokenizer = _trained(280)
    assert tokenizer.special_tokens == ()
    assert tokenizer.encode_with_special("a <|x|> b") == tokenizer.encode("a <|x|> b")


def test_special_tokens_get_ids_after_base_vocab_and_merges() -> None:
    tokenizer = BPETokenizer(special_tokens=("<|endoftext|>", "<|pad|>"))
    tokenizer.train(TRAIN_TEXT, 280)
    eot_id = 256 + len(tokenizer.merges)
    pad_id = eot_id + 1
    assert tokenizer.vocab_size == 280

    ids = tokenizer.encode_with_special("hi<|pad|>!<|endoftext|>")
    assert ids[-1] == eot_id
    assert pad_id in ids
    assert tokenizer.decode([eot_id]) == "<|endoftext|>"
    assert tokenizer.decode([pad_id]) == "<|pad|>"
    assert tokenizer.decode(ids) == "hi<|pad|>!<|endoftext|>"


def test_encode_leaves_special_tokens_as_plain_text() -> None:
    tokenizer = BPETokenizer(special_tokens=("<|endoftext|>",))
    tokenizer.train(TRAIN_TEXT, 270)
    text = "hello <|endoftext|> world"
    plain = tokenizer.encode(text)
    assert max(plain) < 256 + len(tokenizer.merges)
    assert tokenizer.decode(plain) == text


def test_duplicate_or_empty_special_tokens_raise() -> None:
    with pytest.raises(ValueError, match="unique"):
        BPETokenizer(special_tokens=("<|x|>", "<|x|>"))
    with pytest.raises(ValueError, match="non-empty"):
        BPETokenizer(special_tokens=("",))


def test_vocab_size_must_leave_room_for_special_tokens() -> None:
    tokenizer = BPETokenizer(special_tokens=("<|a|>", "<|b|>"))
    with pytest.raises(ValueError, match="room"):
        tokenizer.train(TRAIN_TEXT, 257)


def test_save_load_roundtrip_with_special_tokens(tmp_path) -> None:
    tokenizer = BPETokenizer(special_tokens=("<|endoftext|>",))
    tokenizer.train(TRAIN_TEXT, 270)
    path = tmp_path / "bpe.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)
    assert loaded.special_tokens == ("<|endoftext|>",)
    assert loaded.merges == tokenizer.merges
    text = "the fox<|endoftext|>"
    assert loaded.encode_with_special(text) == tokenizer.encode_with_special(text)
    assert loaded.decode(loaded.encode_with_special(text)) == text


def test_load_legacy_payload_without_special_tokens(tmp_path) -> None:
    tokenizer = _trained(280)
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"merges": [[left, right] for left, right in tokenizer.merges]}),
        encoding="utf-8",
    )
    loaded = BPETokenizer.load(path)
    assert loaded.special_tokens == ()
    assert loaded.vocab_size == 280
    assert loaded.encode(TRAIN_TEXT) == tokenizer.encode(TRAIN_TEXT)


# --- Fuzz --------------------------------------------------------------------

_FUZZ_POOLS = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()-",
    "áéíóúüñçßæøåÀÉÎÕÜŁŻŠ",
    "你好世界東京試験テスト한국어",
    "🌍🚀🎉👍🔥",
    " \t\n\r\x0b\x0c",
    "0123456789",
    "'s't're've'm'll'd",
)


def _fuzz_strings(rng: random.Random, count: int) -> list[str]:
    strings = []
    for _ in range(count):
        pool = "".join(rng.sample(_FUZZ_POOLS, rng.randrange(1, 4)))
        strings.append("".join(rng.choice(pool) for _ in range(rng.randrange(0, 40))))
    return strings


def test_fuzz_roundtrip_ids_in_range_and_save_load(tmp_path) -> None:
    rng = random.Random(20260822)
    strings = _fuzz_strings(rng, 60)
    corpus = " ".join(strings) * 3
    tokenizer = BPETokenizer(special_tokens=("<|endoftext|>", "<|pad|>"))
    tokenizer.train(corpus, 350)

    for s in strings:
        ids = tokenizer.encode(s)
        assert all(0 <= i < tokenizer.vocab_size for i in ids)
        assert tokenizer.decode(ids) == s

    mixed = f"{strings[0]}<|pad|>{strings[1]} 🌍<|endoftext|><|pad|>"
    special_ids = tokenizer.encode_with_special(mixed)
    assert special_ids.count(257 + len(tokenizer.merges)) == 2  # <|pad|>
    assert special_ids.count(256 + len(tokenizer.merges)) == 1  # <|endoftext|>
    assert tokenizer.decode(special_ids) == mixed
    plain = tokenizer.encode(mixed)
    assert max(plain) < 256 + len(tokenizer.merges)
    assert tokenizer.decode(plain) == mixed

    path = tmp_path / "fuzz.json"
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.vocab_size == tokenizer.vocab_size
    for s in strings[:10]:
        assert loaded.encode_with_special(s) == tokenizer.encode_with_special(s)
        assert loaded.decode(loaded.encode(s)) == s
