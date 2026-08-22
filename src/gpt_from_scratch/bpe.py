from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

# GPT-2 contraction alternatives; matched case-sensitively, longest-prefix safe.
_CONTRACTIONS: tuple[str, ...] = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _contraction_length(text: str, i: int) -> int:
    """Length of the contraction suffix starting at ``i``, or 0 if none."""
    for suffix in _CONTRACTIONS:
        if text.startswith(suffix, i):
            return len(suffix)
    return 0


def _pretokenize(text: str) -> list[str]:
    """Split ``text`` into chunks following GPT-2 pre-tokenization semantics.

    Contractions ('s, 't, 're, 've, 'm, 'll, 'd), letter runs, digit runs and
    other-symbol runs each form their own chunk. A single leading literal
    space attaches to the chunk that follows it; any remaining whitespace
    (other whitespace characters, and whitespace runs minus their last
    character) forms its own chunk, so merges can never cross these
    boundaries. stdlib ``re`` lacks unicode property classes, so
    classification hand-rolls ``\\p{L}``/``\\p{N}``/``\\p{Z}`` via
    ``str.isalpha`` / ``str.isdigit`` / ``str.isspace``.
    """
    chunks: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if j == n:  # trailing whitespace run stays whole
                chunks.append(text[i:j])
                break
            if j - i > 1:  # all but the run's last char stand alone
                chunks.append(text[i : j - 1])
            i = j - 1
            if text[i] != " ":  # only a literal space may lead the next chunk
                chunks.append(text[i])
                i += 1
                continue
        start = i
        length = _contraction_length(text, i)
        if length:
            i += length
        else:
            if text[i] == " ":  # leading space joins the following chunk
                i += 1
            if text[i].isalpha():
                while i < n and text[i].isalpha():
                    i += 1
            elif text[i].isdigit():
                while i < n and text[i].isdigit():
                    i += 1
            else:
                while i < n and not (text[i].isspace() or text[i].isalpha() or text[i].isdigit()):
                    i += 1
        chunks.append(text[start:i])
    return chunks


def _pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in pairwise(ids):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE tokenizer with GPT-2 style pre-tokenization.

    Ids 0-255 are raw UTF-8 bytes, id ``256 + rank`` is the merge at that
    rank, and optional special tokens take the ids immediately after all
    merges. Special tokens are never produced by :meth:`encode`; use
    :meth:`encode_with_special` to recognize their literal forms.
    """

    def __init__(self, special_tokens: Sequence[str] = ()) -> None:
        specials = tuple(special_tokens)
        if len(set(specials)) != len(specials):
            raise ValueError("special_tokens must be unique")
        if any(not token for token in specials):
            raise ValueError("special_tokens must be non-empty strings")
        self.special_tokens: tuple[str, ...] = specials
        self.merges: list[tuple[int, int]] = []
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._ranks: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    def _special_ids(self) -> dict[str, int]:
        base = 256 + len(self.merges)
        return {token: base + offset for offset, token in enumerate(self.special_tokens)}

    def train(self, text: str, vocab_size: int) -> None:
        num_special = len(self.special_tokens)
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        if vocab_size < 256 + num_special:
            raise ValueError(
                f"vocab_size {vocab_size} leaves no room for {num_special} special "
                f"tokens; need at least {256 + num_special}"
            )
        # Merge counting is per pre-tokenized chunk, so merges never cross
        # chunk boundaries during training either.
        words = [list(chunk.encode("utf-8")) for chunk in _pretokenize(text)]
        merges: list[tuple[int, int]] = []
        vocab = {i: bytes([i]) for i in range(256)}
        while 256 + len(merges) + num_special < vocab_size:
            counts: dict[tuple[int, int], int] = {}
            for word in words:
                for pair in pairwise(word):
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                raise ValueError(
                    f"text is too small to reach vocab_size {vocab_size}; "
                    f"maximum achievable is {256 + len(merges) + num_special}"
                )
            best = max(counts.items(), key=lambda item: item[1])[0]
            new_id = 256 + len(merges)
            words = [_merge(word, best, new_id) for word in words]
            merges.append(best)
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        self.merges = merges
        self._vocab = vocab
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            pairs = set(pairwise(ids))
            best = min(pairs, key=lambda pair: self._ranks.get(pair, len(self._ranks)))
            if best not in self._ranks:
                break
            ids = _merge(ids, best, 256 + self._ranks[best])
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _pretokenize(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode_with_special(self, text: str) -> list[int]:
        """Like :meth:`encode`, but literal special-token occurrences map to
        their reserved ids. Matching is greedy longest-first."""
        special_ids = self._special_ids()
        if not special_ids:
            return self.encode(text)
        ordered = sorted(special_ids, key=len, reverse=True)
        ids: list[int] = []
        i, n = 0, len(text)
        while i < n:
            token = next((t for t in ordered if text.startswith(t, i)), None)
            if token is not None:
                ids.append(special_ids[token])
                i += len(token)
                continue
            end = n
            for t in ordered:
                pos = text.find(t, i + 1)
                if pos != -1:
                    end = min(end, pos)
            ids.extend(self.encode(text[i:end]))
            i = end
        return ids

    def decode(self, ids: list[int]) -> str:
        special_values = {v: k for k, v in self._special_ids().items()}
        parts = [
            special_values[i].encode("utf-8") if i in special_values else self._vocab[i]
            for i in ids
        ]
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {
            "merges": [[left, right] for left, right in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(payload.get("special_tokens", ()))
        tokenizer.merges = [(int(left), int(right)) for left, right in payload["merges"]]
        vocab = {i: bytes([i]) for i in range(256)}
        for new_id, pair in enumerate(tokenizer.merges):
            vocab[256 + new_id] = vocab[pair[0]] + vocab[pair[1]]
        tokenizer._vocab = vocab
        tokenizer._ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
        return tokenizer
