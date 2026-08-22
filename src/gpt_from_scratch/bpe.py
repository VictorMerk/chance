from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path


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
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._ranks: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        vocab = {i: bytes([i]) for i in range(256)}
        while len(vocab) < vocab_size:
            counts = _pair_counts(ids)
            if not counts:
                raise ValueError(
                    f"text is too small to reach vocab_size {vocab_size}; "
                    f"maximum achievable is {len(vocab)}"
                )
            best = max(counts.items(), key=lambda item: item[1])[0]
            new_id = len(vocab)
            ids = _merge(ids, best, new_id)
            merges.append(best)
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        self.merges = merges
        self._vocab = vocab
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(pairwise(ids))
            best = min(pairs, key=lambda pair: self._ranks.get(pair, len(self._ranks)))
            if best not in self._ranks:
                break
            ids = _merge(ids, best, 256 + self._ranks[best])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        payload = {"merges": [[left, right] for left, right in self.merges]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.merges = [(int(left), int(right)) for left, right in payload["merges"]]
        vocab = {i: bytes([i]) for i in range(256)}
        for new_id, pair in enumerate(tokenizer.merges):
            vocab[256 + new_id] = vocab[pair[0]] + vocab[pair[1]]
        tokenizer._vocab = vocab
        tokenizer._ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
        return tokenizer
